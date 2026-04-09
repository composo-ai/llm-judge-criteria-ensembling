"""Unified data collection for RB2 judge experiments.

Collects k scores from mini and/or full models for every example in RB2,
using a specified prompt variant. Results are written incrementally as JSONL
and support resume on interruption.

Prompt variants:
  base         Vanilla RB2 prompt
  criteria     RB2 + task-specific criteria
  cal-low      RB2 + calibration context (rejected response)
  cal-high     RB2 + calibration context (chosen response)
  cal-both     RB2 + calibration context (one high + one low)
  cal-cross    RB2 + calibration context (different category)
  combined     RB2 + criteria + calibration (low)

Usage:
  python collect.py --prompt base --models both --k 8
  python collect.py --prompt base --models all --k 8
  python collect.py --prompt criteria --models both --k 8
  python collect.py --prompt base --models full --k 1 --temperature 0.0
"""

import argparse
import asyncio
import json
import os
import sys
import random
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import datasets
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio

from judge import TASK_CRITERIA, score_response, score_response_n

PROMPT_VARIANTS = [
    "base", "criteria",
    "cal-low", "cal-high", "cal-both", "cal-cross",
    "combined",
]
CAL_MAP = {
    "cal-low": "low",
    "cal-high": "high",
    "cal-both": "both",
    "cal-cross": "cross-category",
    "combined": "low",
}


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_completed_ids(path: Path) -> set[str]:
    ids = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                ids.add(json.loads(line)["id"])
    return ids


def output_filename(prompt: str, models: str, k: int, temperature: float) -> str:
    name = f"{prompt}_{models}_k{k}"
    if temperature != 1.0:
        name += f"_t{temperature}"
    return name + ".jsonl"


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

_cal_cache: dict[str, dict] = {}


async def _score_calibration(
    client: AsyncAzureOpenAI,
    cal: dict,
    subset: str,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    criteria: str | None = None,
) -> dict:
    cache_key = cal["id"]
    if cache_key in _cal_cache:
        return _cal_cache[cache_key]
    async with semaphore:
        result = await score_response(
            client, cal["prompt"], cal["response"], subset,
            model=model, temperature=temperature, criteria=criteria,
        )
    _cal_cache[cache_key] = result
    return result


def _pick_calibration(
    example: dict,
    queue: list[dict],
    variant: str,
    rng: random.Random,
    all_queues: dict[str, list[dict]],
) -> dict:
    subset = example["subset"]

    if variant == "cross-category":
        other = [s for s in all_queues if s != subset]
        cal = rng.choice(all_queues[rng.choice(other)])
        return {"id": cal["id"], "prompt": cal["prompt"], "response": cal["chosen"][0]}

    candidates = [c for c in queue if c["id"] != example["id"]]
    cal = rng.choice(candidates)

    if variant == "high":
        return {"id": cal["id"], "prompt": cal["prompt"], "response": cal["chosen"][0]}
    elif variant == "low":
        return {
            "id": cal["id"], "prompt": cal["prompt"],
            "response": cal["rejected"][rng.randrange(len(cal["rejected"]))],
        }
    elif variant == "both":
        cal2 = rng.choice(candidates)
        return {
            "id": f"{cal['id']}+{cal2['id']}",
            "prompt_high": cal["prompt"], "response_high": cal["chosen"][0],
            "prompt_low": cal2["prompt"],
            "response_low": cal2["rejected"][rng.randrange(len(cal2["rejected"]))],
        }
    # fallback
    return {"id": cal["id"], "prompt": cal["prompt"], "response": cal["chosen"][0]}


async def _resolve_calibration(
    cal_cand: dict,
    variant: str,
    subset: str,
    client: AsyncAzureOpenAI,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    criteria: str | None = None,
) -> tuple[dict | None, dict]:
    """Score calibration example(s). Returns (cal_example | None, metadata)."""
    if variant == "both":
        hi = await _score_calibration(
            client,
            {"id": cal_cand["id"].split("+")[0],
             "prompt": cal_cand["prompt_high"], "response": cal_cand["response_high"]},
            subset, model, temperature, semaphore, criteria,
        )
        lo = await _score_calibration(
            client,
            {"id": cal_cand["id"].split("+")[1],
             "prompt": cal_cand["prompt_low"], "response": cal_cand["response_low"]},
            subset, model, temperature, semaphore, criteria,
        )
        cost = {
            "input_tokens": hi["usage"]["input_tokens"] + lo["usage"]["input_tokens"],
            "output_tokens": hi["usage"]["output_tokens"] + lo["usage"]["output_tokens"],
        }
        meta = {"cal_id": cal_cand["id"], "cal_score": None, "cal_cost": cost}
        if hi["raw_score"] is None or lo["raw_score"] is None:
            return None, meta
        meta["cal_score"] = {"high": hi["raw_score"], "low": lo["raw_score"]}
        return {
            "prompt_high": cal_cand["prompt_high"], "response_high": cal_cand["response_high"],
            "score_high": hi["raw_score"],
            "prompt_low": cal_cand["prompt_low"], "response_low": cal_cand["response_low"],
            "score_low": lo["raw_score"],
        }, meta
    else:
        r = await _score_calibration(
            client, cal_cand, subset, model, temperature, semaphore, criteria,
        )
        cost = {"input_tokens": r["usage"]["input_tokens"],
                "output_tokens": r["usage"]["output_tokens"]}
        meta = {"cal_id": cal_cand["id"], "cal_score": None, "cal_cost": cost}
        if r["raw_score"] is None:
            return None, meta
        meta["cal_score"] = r["raw_score"]
        return {
            "prompt": cal_cand["prompt"], "response": cal_cand["response"],
            "score": r["raw_score"],
        }, meta


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

async def score_example(
    clients: dict[str, AsyncAzureOpenAI],
    model_names: dict[str, str],
    example: dict,
    k: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    prompt_variant: str,
    criteria: str | None = None,
    calibration_example: dict | None = None,
    cal_metadata: dict | None = None,
    cal_model_key: str | None = None,
) -> dict:
    subset = example["subset"]
    prompt = example["prompt"]
    all_responses = [example["chosen"][0]] + list(example["rejected"])
    model_keys = list(clients.keys())

    async with semaphore:
        tasks = []
        for mk in model_keys:
            for resp in all_responses:
                tasks.append(score_response_n(
                    clients[mk], prompt, resp, subset,
                    n=k, model=model_names[mk], temperature=temperature,
                    criteria=criteria, calibration_example=calibration_example,
                ))
        all_results = await asyncio.gather(*tasks)

    result: dict = {
        "id": example["id"],
        "subset": subset,
        "prompt_variant": prompt_variant,
        "k": k,
    }
    cost: dict[str, int] = {}
    idx = 0
    for mk in model_keys:
        scores, errors = [], []
        inp_tok, out_tok = 0, 0
        for _ in all_responses:
            r = all_results[idx]
            scores.append(r["scores"])
            errors.append(r["errors"])
            inp_tok += r["usage"]["input_tokens"]
            out_tok += r["usage"]["output_tokens"]
            idx += 1
        result[f"{mk}_scores"] = scores
        result[f"{mk}_errors"] = errors
        cost[f"{mk}_input_tokens"] = inp_tok
        cost[f"{mk}_output_tokens"] = out_tok

    # Attribute calibration cost to whichever model scored the calibration example
    if cal_metadata and "cal_cost" in cal_metadata and cal_model_key:
        cost[f"{cal_model_key}_input_tokens"] = cost.get(f"{cal_model_key}_input_tokens", 0) + cal_metadata["cal_cost"]["input_tokens"]
        cost[f"{cal_model_key}_output_tokens"] = cost.get(f"{cal_model_key}_output_tokens", 0) + cal_metadata["cal_cost"]["output_tokens"]

    result["cost"] = cost
    if cal_metadata:
        result["cal_id"] = cal_metadata["cal_id"]
        result["cal_score"] = cal_metadata["cal_score"]

    # Refused if ALL scores None for any response across ALL models
    refused = False
    for ri in range(len(all_responses)):
        if all(
            all(s is None for s in result[f"{mk}_scores"][ri])
            for mk in model_keys
        ):
            refused = True
            break
    result["refused"] = refused
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Collect RB2 judge scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prompt", required=True, choices=PROMPT_VARIANTS)
    parser.add_argument("--models", default="both",
                        choices=["nano", "mini", "full", "both", "all"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=999,
                        help="Max examples per subset (999 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--full-model",
                        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"))
    parser.add_argument("--mini-model",
                        default=os.environ.get("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-5.4-mini"))
    parser.add_argument("--nano-model",
                        default=os.environ.get("AZURE_OPENAI_NANO_DEPLOYMENT", "gpt-5.4-nano"))
    args = parser.parse_args()

    # --- Dataset ---
    print("Loading RewardBench 2...")
    ds = datasets.load_dataset("allenai/reward-bench-2", split="test")
    ds = ds.filter(lambda x: x["subset"] != "Ties")
    print(f"  {len(ds)} examples (Ties excluded)")
    if args.subset:
        ds = ds.filter(lambda x: x["subset"] == args.subset)
        print(f"  Filtered to '{args.subset}': {len(ds)}")
    ds = ds.shuffle(seed=args.seed)

    subset_queues: dict[str, list[dict]] = defaultdict(list)
    for ex in ds:
        subset_queues[ex["subset"]].append(ex)
    subsets = sorted(subset_queues.keys())
    targets = {s: min(args.sample_size, len(subset_queues[s])) for s in subsets}

    # --- Config ---
    use_criteria = args.prompt in ("criteria", "combined")
    cal_variant = CAL_MAP.get(args.prompt)

    print(f"\n  prompt={args.prompt}  models={args.models}  k={args.k}  temp={args.temperature}")
    for s in subsets:
        print(f"  {s}: {targets[s]} / {len(subset_queues[s])}")

    if cal_variant:
        for s in subsets:
            if len(subset_queues[s]) < 2:
                print(f"ERROR: '{s}' needs >= 2 examples for calibration")
                return

    # --- Output ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / output_filename(args.prompt, args.models, args.k, args.temperature)
    completed = load_completed_ids(out_file)
    if completed:
        print(f"  Resuming: {len(completed)} done")

    # --- Clients ---
    # Expand "both" and "all" to the set of models to create
    models_to_run = {
        "both": {"mini", "full"},
        "all": {"nano", "mini", "full"},
    }.get(args.models, {args.models})

    clients: dict[str, AsyncAzureOpenAI] = {}
    model_names: dict[str, str] = {}
    if "full" in models_to_run:
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            print("ERROR: AZURE_OPENAI_API_KEY not set"); sys.exit(1)
        clients["full"] = AsyncAzureOpenAI(api_version="2025-04-01-preview")
        model_names["full"] = args.full_model
    if "mini" in models_to_run:
        if not os.environ.get("AZURE_OPENAI_MINI_API_KEY"):
            print("ERROR: AZURE_OPENAI_MINI_API_KEY not set"); sys.exit(1)
        clients["mini"] = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_MINI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_MINI_ENDPOINT"],
            api_version="2025-04-01-preview",
        )
        model_names["mini"] = args.mini_model
    if "nano" in models_to_run:
        if not os.environ.get("AZURE_OPENAI_NANO_API_KEY"):
            print("ERROR: AZURE_OPENAI_NANO_API_KEY not set"); sys.exit(1)
        clients["nano"] = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_NANO_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_NANO_ENDPOINT"],
            api_version="2025-04-01-preview",
        )
        model_names["nano"] = args.nano_model

    # Calibration always uses the most capable available model
    cal_model_key = "full" if "full" in clients else ("mini" if "mini" in clients else "nano")
    cal_client = clients[cal_model_key]
    cal_model = model_names[cal_model_key]

    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)

    # --- Resume state ---
    valid_per_subset: dict[str, int] = defaultdict(int)
    if completed and out_file.exists():
        with open(out_file) as f:
            for line in f:
                valid_per_subset[json.loads(line)["subset"]] += 1

    print(f"\n  Collecting...")
    t0 = time.time()
    outf = open(out_file, "a")
    n_valid = len(completed)
    n_refused = 0

    for subset in subsets:
        queue = subset_queues[subset]
        qi = 0
        target = targets[subset]
        criteria = TASK_CRITERIA[subset] if use_criteria else None

        while valid_per_subset[subset] < target and qi < len(queue):
            # Build batch of uncompleted examples
            batch = []
            while len(batch) < (target - valid_per_subset[subset]) and qi < len(queue):
                ex = queue[qi]; qi += 1
                if ex["id"] not in completed:
                    batch.append(ex)
            if not batch:
                break

            # --- Calibration ---
            cal_examples: list[dict | None] = [None] * len(batch)
            cal_metas: list[dict | None] = [None] * len(batch)
            if cal_variant:
                cands = [_pick_calibration(ex, queue, cal_variant, rng, subset_queues)
                         for ex in batch]
                keep = []
                for i, (ex, cand) in enumerate(zip(batch, cands)):
                    cal_ex, cal_meta = await _resolve_calibration(
                        cand, cal_variant, subset,
                        cal_client, cal_model, args.temperature, sem, criteria,
                    )
                    if cal_ex is None:
                        n_refused += 1
                        print(f"  Cal refused {ex['id']} in {subset}")
                    else:
                        cal_examples[i] = cal_ex
                        cal_metas[i] = cal_meta
                        keep.append(i)
                batch = [batch[i] for i in keep]
                cal_examples = [cal_examples[i] for i in keep]
                cal_metas = [cal_metas[i] for i in keep]
                if not batch:
                    continue

            # --- Score ---
            tasks = [
                score_example(
                    clients, model_names, ex, args.k, args.temperature, sem,
                    args.prompt, criteria=criteria,
                    calibration_example=cal_examples[i],
                    cal_metadata=cal_metas[i],
                    cal_model_key=cal_model_key,
                )
                for i, ex in enumerate(batch)
            ]
            results = await tqdm_asyncio.gather(*tasks, desc=subset)

            for r in results:
                if not r["refused"]:
                    outf.write(json.dumps(r) + "\n")
                    outf.flush()
                    n_valid += 1
                    valid_per_subset[subset] += 1
                else:
                    n_refused += 1
                    print(f"  Refused {r['id']} in {subset}")

        if valid_per_subset[subset] < target:
            print(f"  WARNING: {valid_per_subset[subset]}/{target} for {subset}")

    outf.close()
    elapsed = time.time() - t0

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"DONE: {args.prompt} | {args.models} | k={args.k} | temp={args.temperature}")
    print(f"{'='*60}")
    print(f"  {out_file}")
    print(f"  {n_valid} valid, {n_refused} refused, {elapsed:.0f}s")
    for s in subsets:
        print(f"  {s}: {valid_per_subset[s]}")
    if cal_variant:
        print(f"  Cal cache: {len(_cal_cache)}")

    totals: dict[str, int] = defaultdict(int)
    with open(out_file) as f:
        for line in f:
            for key, val in json.loads(line).get("cost", {}).items():
                totals[key] += val
    for key in sorted(totals):
        print(f"  {key}: {totals[key]:,}")


if __name__ == "__main__":
    asyncio.run(main())

import argparse
import asyncio
import json
import os
import random
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from collections import defaultdict
from pathlib import Path

import datasets
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio

from judge import score_response, score_response_n, TASK_CRITERIA


def load_completed_ids(output_file: Path) -> set[str]:
    completed = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                completed.add(json.loads(line)["id"])
    return completed


# Cache for calibration scores: {cal_id: result_dict}
cal_score_cache: dict[str, dict] = {}


async def get_cal_score(
    client: AsyncAzureOpenAI,
    cal_candidate: dict,
    subset: str,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    criteria: str | None = None,
) -> dict:
    """Score a calibration example, using cache if available."""
    cal_id = cal_candidate["id"]
    if cal_id in cal_score_cache:
        return cal_score_cache[cal_id]

    async with semaphore:
        result = await score_response(
            client,
            cal_candidate["prompt"],
            cal_candidate["response"],
            subset,
            model=model,
            temperature=temperature,
            criteria=criteria,
        )

    cal_score_cache[cal_id] = result
    return result


def pick_calibration_candidate(
    example: dict,
    queue: list[dict],
    variant: str,
    rng: random.Random,
) -> dict:
    """Pick a calibration candidate from the same subset."""
    candidates = [c for c in queue if c["id"] != example["id"]]

    if variant == "high":
        cal = rng.choice(candidates)
        return {"id": cal["id"], "prompt": cal["prompt"], "response": cal["chosen"][0]}
    elif variant == "low":
        cal = rng.choice(candidates)
        return {
            "id": cal["id"],
            "prompt": cal["prompt"],
            "response": cal["rejected"][rng.randrange(len(cal["rejected"]))],
        }
    else:  # default to low
        cal = rng.choice(candidates)
        return {
            "id": cal["id"],
            "prompt": cal["prompt"],
            "response": cal["rejected"][rng.randrange(len(cal["rejected"]))],
        }


async def score_example_combined(
    mini_client: AsyncAzureOpenAI,
    full_client: AsyncAzureOpenAI,
    example: dict,
    mini_model: str,
    full_model: str,
    temperature: float,
    mini_n: int,
    full_n: int,
    semaphore: asyncio.Semaphore,
    cal_candidate: dict,
    criteria: str,
) -> dict:
    subset = example["subset"]
    prompt = example["prompt"]

    # Step 1: Score calibration example to get a reference score
    cal_result = await get_cal_score(
        full_client, cal_candidate, subset, full_model, temperature, semaphore, criteria
    )

    if cal_result["raw_score"] is None:
        return {
            "id": example["id"],
            "subset": subset,
            "mini_n": mini_n,
            "full_n": full_n,
            "mini_scores": [[None] * mini_n] * 4,
            "mini_stds": [None] * 4,
            "full_scores": [[None] * full_n] * 4,
            "mini_errors": [["cal_refused"] * mini_n] * 4,
            "full_errors": [["cal_refused"] * full_n] * 4,
            "refused": True,
            "cal_id": cal_candidate["id"],
            "cal_score": None,
            "criteria": criteria,
            "cost": {
                "mini_input_tokens": 0,
                "mini_output_tokens": 0,
                "full_input_tokens": cal_result["usage"]["input_tokens"],
                "full_output_tokens": cal_result["usage"]["output_tokens"],
            },
        }

    cal_example = {
        "prompt": cal_candidate["prompt"],
        "response": cal_candidate["response"],
        "score": cal_result["raw_score"],
    }

    # Step 2: Score all 4 responses with both models
    all_responses = [example["chosen"][0]] + list(example["rejected"])
    num_responses = len(all_responses)

    async with semaphore:
        all_results = await asyncio.gather(
            *[
                score_response_n(
                    mini_client,
                    prompt,
                    resp,
                    subset,
                    n=mini_n,
                    model=mini_model,
                    temperature=temperature,
                    criteria=criteria,
                    calibration_example=cal_example,
                )
                for resp in all_responses
            ],
            *[
                score_response_n(
                    full_client,
                    prompt,
                    resp,
                    subset,
                    n=full_n,
                    model=full_model,
                    temperature=temperature,
                    criteria=criteria,
                    calibration_example=cal_example,
                )
                for resp in all_responses
            ],
        )

    mini_results = all_results[:num_responses]
    full_results = all_results[num_responses:]

    mini_scores = []
    mini_errors = []
    mini_stds = []
    mini_input_tokens = cal_result["usage"]["input_tokens"]
    mini_output_tokens = cal_result["usage"]["output_tokens"]

    for r in mini_results:
        mini_scores.append(r["scores"])
        mini_errors.append(r["errors"])
        valid = [s for s in r["scores"] if s is not None]
        mini_stds.append(float(np.std(valid)) if len(valid) > 1 else None)
        mini_input_tokens += r["usage"]["input_tokens"]
        mini_output_tokens += r["usage"]["output_tokens"]

    full_scores = []
    full_errors = []
    full_input_tokens = 0
    full_output_tokens = 0

    for r in full_results:
        full_scores.append(r["scores"])
        full_errors.append(r["errors"])
        full_input_tokens += r["usage"]["input_tokens"]
        full_output_tokens += r["usage"]["output_tokens"]

    # Refused if ALL mini AND full scores are None for any response
    any_response_fully_failed = any(
        all(s is None for s in mini_scores[i])
        and all(s is None for s in full_scores[i])
        for i in range(num_responses)
    )

    return {
        "id": example["id"],
        "subset": subset,
        "mini_n": mini_n,
        "full_n": full_n,
        "mini_scores": mini_scores,
        "mini_stds": mini_stds,
        "full_scores": full_scores,
        "mini_errors": mini_errors,
        "full_errors": full_errors,
        "refused": any_response_fully_failed,
        "cal_id": cal_candidate["id"],
        "cal_score": cal_result["raw_score"],
        "criteria": criteria,
        "cost": {
            "mini_input_tokens": mini_input_tokens,
            "mini_output_tokens": mini_output_tokens,
            "full_input_tokens": full_input_tokens,
            "full_output_tokens": full_output_tokens,
        },
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Run RB2 combined scoring (criteria + calibration + escalation)"
    )
    parser.add_argument(
        "--mini-model",
        default=os.environ.get("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-5.4-mini"),
        help="Mini model deployment name",
    )
    parser.add_argument(
        "--full-model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
        help="Full model deployment name",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50, help="Number of examples per subset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Max concurrent examples"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Judge temperature"
    )
    parser.add_argument("--output-dir", default="results/raw", help="Output directory")
    parser.add_argument("--subset", default=None, help="Filter to a specific subset")
    parser.add_argument(
        "--mini-n", type=int, default=8, help="Number of completions per mini call"
    )
    parser.add_argument(
        "--full-n", type=int, default=8, help="Number of completions per full call"
    )
    parser.add_argument(
        "--cal-variant",
        default="low",
        choices=["high", "low"],
        help="Calibration variant (default: low, which performed best in isolation)",
    )
    args = parser.parse_args()

    # Load dataset
    print("Loading RewardBench 2 dataset...")
    ds = datasets.load_dataset("allenai/reward-bench-2", split="test")
    ds = ds.filter(lambda x: x["subset"] != "Ties")
    print(f"Excluded Ties subset, {len(ds)} examples remaining")

    if args.subset:
        ds = ds.filter(lambda x: x["subset"] == args.subset)
        print(f"Filtered to subset '{args.subset}': {len(ds)} examples")

    ds = ds.shuffle(seed=args.seed)
    subset_counts = defaultdict(int)
    for ex in ds:
        subset_counts[ex["subset"]] += 1

    subsets_sorted = sorted(subset_counts.keys())
    subset_targets = {
        s: min(args.sample_size, subset_counts[s]) for s in subsets_sorted
    }

    print(f"\nSampling {args.sample_size} per subset:")
    print(f"  Mini: {args.mini_model} n={args.mini_n}")
    print(f"  Full: {args.full_model} n={args.full_n}")
    print(f"  Calibration variant: {args.cal_variant}")
    print(f"  Criteria: enabled (task-specific)")
    for s in subsets_sorted:
        print(f"  {s}: {subset_targets[s]} / {subset_counts[s]}")

    # Build per-subset queues
    subset_queues = defaultdict(list)
    for ex in ds:
        subset_queues[ex["subset"]].append(ex)

    for s in subsets_sorted:
        if len(subset_queues[s]) < 2:
            print(f"ERROR: Subset '{s}' needs at least 2 examples for calibration.")
            return

    # Output file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"combined_cal_{args.cal_variant}_mini_n{args.mini_n}_full_n{args.full_n}.jsonl"
    )

    completed_ids = load_completed_ids(output_file)
    if completed_ids:
        print(f"\nResuming: {len(completed_ids)} examples already completed")

    # Create clients
    if not os.environ.get("AZURE_OPENAI_MINI_API_KEY"):
        print("ERROR: AZURE_OPENAI_MINI_API_KEY not set in .env")
        return
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not set in .env")
        return

    mini_client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_MINI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_MINI_ENDPOINT"],
        api_version="2025-04-01-preview",
    )
    full_client = AsyncAzureOpenAI(api_version="2025-04-01-preview")
    semaphore = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)

    print(f"\nScoring (temp={args.temperature}, concurrency={args.concurrency})...")
    print(
        "Calibration scores cached. Results written incrementally. Safe to interrupt and resume.\n"
    )
    start_time = time.time()

    result_count = len(completed_ids)
    valid_per_subset = defaultdict(int)
    refused_total = 0

    if completed_ids and output_file.exists():
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                valid_per_subset[r["subset"]] += 1

    outfile = open(output_file, "a")

    for subset in subsets_sorted:
        queue = subset_queues[subset]
        queue_idx = 0
        subset_target = subset_targets[subset]
        criteria = TASK_CRITERIA[subset]

        while valid_per_subset[subset] < subset_target and queue_idx < len(queue):
            batch = []
            while len(batch) < (
                subset_target - valid_per_subset[subset]
            ) and queue_idx < len(queue):
                ex = queue[queue_idx]
                queue_idx += 1
                if ex["id"] not in completed_ids:
                    batch.append(ex)
            if not batch:
                break

            # Pick calibration candidates
            cal_candidates = [
                pick_calibration_candidate(ex, queue, args.cal_variant, rng)
                for ex in batch
            ]

            tasks = [
                score_example_combined(
                    mini_client,
                    full_client,
                    ex,
                    args.mini_model,
                    args.full_model,
                    args.temperature,
                    args.mini_n,
                    args.full_n,
                    semaphore,
                    cal,
                    criteria,
                )
                for ex, cal in zip(batch, cal_candidates)
            ]
            batch_results = await tqdm_asyncio.gather(*tasks, desc=subset)

            for r in batch_results:
                if not r["refused"]:
                    outfile.write(json.dumps(r) + "\n")
                    outfile.flush()
                    result_count += 1
                    valid_per_subset[subset] += 1
                else:
                    refused_total += 1
                    print(
                        f"  Refused example {r['id']} in {subset}, retrying with next..."
                    )

        if valid_per_subset[subset] < subset_target:
            print(
                f"  WARNING: Only got {valid_per_subset[subset]}/{subset_target} valid examples for {subset}"
            )

    outfile.close()
    elapsed = time.time() - start_time

    print(f"\nResults saved to {output_file}")
    print(f"Calibration cache: {len(cal_score_cache)} unique cal examples scored")

    total_mini_input = 0
    total_mini_output = 0
    total_full_input = 0
    total_full_output = 0
    with open(output_file) as f:
        for line in f:
            r = json.loads(line)
            total_mini_input += r["cost"]["mini_input_tokens"]
            total_mini_output += r["cost"]["mini_output_tokens"]
            total_full_input += r["cost"]["full_input_tokens"]
            total_full_output += r["cost"]["full_output_tokens"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nExamples: {result_count} valid, {refused_total} refused")
    print(f"Wall clock: {elapsed:.1f}s")

    for subset in subsets_sorted:
        print(f"  {subset}: {valid_per_subset[subset]}")

    print(f"\nTokens (mini):")
    print(f"  Input:  {total_mini_input:,}")
    print(f"  Output: {total_mini_output:,}")
    print(f"\nTokens (full):")
    print(f"  Input:  {total_full_input:,}")
    print(f"  Output: {total_full_output:,}")


if __name__ == "__main__":
    asyncio.run(main())

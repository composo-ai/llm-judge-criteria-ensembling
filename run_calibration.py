import argparse
import asyncio
import json
import os
import random
import time

from dotenv import load_dotenv

load_dotenv()
from collections import defaultdict
from pathlib import Path

import datasets
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio

from judge import score_response

# Cache for calibration scores: {cal_id: raw_score}
# Avoids re-scoring the same calibration example for different target examples
cal_score_cache: dict[str, dict] = {}


async def get_cal_score(
    client: AsyncAzureOpenAI,
    cal_candidate: dict,
    subset: str,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
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
        )

    cal_score_cache[cal_id] = result
    return result


def pick_calibration_candidate(
    example: dict,
    queue: list[dict],
    variant: str,
    rng: random.Random,
    subset: str,
    all_subset_queues: dict[str, list[dict]],
) -> dict:
    """Pick a calibration candidate based on the variant.

    For high/low/random: picks from the same subset.
    For cross-category: picks from a different subset (control).

    Note: high/low filtering is based on which response is chosen vs rejected,
    not on a pre-computed score. chosen[0] is the known-good response (high),
    rejected responses are known-bad (low).
    """
    if variant == "cross-category":
        # Pick from a different subset
        other_subsets = [s for s in all_subset_queues if s != subset]
        other_subset = rng.choice(other_subsets)
        other_queue = all_subset_queues[other_subset]
        cal = rng.choice(other_queue)
        return {
            "id": cal["id"],
            "prompt": cal["prompt"],
            "response": cal["chosen"][0],
        }

    # Same-subset candidates (excluding self)
    candidates = [c for c in queue if c["id"] != example["id"]]

    if variant == "high":
        # Use the chosen (correct) response as calibration
        cal = rng.choice(candidates)
        return {
            "id": cal["id"],
            "prompt": cal["prompt"],
            "response": cal["chosen"][0],
        }
    elif variant == "low":
        # Use a rejected (incorrect) response as calibration
        cal = rng.choice(candidates)
        return {
            "id": cal["id"],
            "prompt": cal["prompt"],
            "response": cal["rejected"][rng.randrange(len(cal["rejected"]))],
        }
    elif variant == "both":
        # Return two candidates — handled specially in score_example
        cal_high = rng.choice(candidates)
        cal_low = rng.choice(candidates)
        return {
            "id": f"{cal_high['id']}+{cal_low['id']}",
            "prompt_high": cal_high["prompt"],
            "response_high": cal_high["chosen"][0],
            "prompt_low": cal_low["prompt"],
            "response_low": cal_low["rejected"][
                rng.randrange(len(cal_low["rejected"]))
            ],
        }
    else:  # "random" — same as high (chosen response, random example)
        cal = rng.choice(candidates)
        return {
            "id": cal["id"],
            "prompt": cal["prompt"],
            "response": cal["chosen"][0],
        }


async def score_example(
    client: AsyncAzureOpenAI,
    example: dict,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    cal_candidate: dict,
    variant: str,
) -> dict:
    subset = example["subset"]
    prompt = example["prompt"]

    # Step 1: Score the calibration example(s) to get reference score(s)
    if variant == "both":
        # Score both high and low calibration examples
        cal_result_high = await get_cal_score(
            client,
            {
                "id": cal_candidate["id"].split("+")[0],
                "prompt": cal_candidate["prompt_high"],
                "response": cal_candidate["response_high"],
            },
            subset,
            model,
            temperature,
            semaphore,
        )
        cal_result_low = await get_cal_score(
            client,
            {
                "id": cal_candidate["id"].split("+")[1],
                "prompt": cal_candidate["prompt_low"],
                "response": cal_candidate["response_low"],
            },
            subset,
            model,
            temperature,
            semaphore,
        )

        if cal_result_high["raw_score"] is None or cal_result_low["raw_score"] is None:
            return {
                "id": example["id"],
                "subset": subset,
                "k": 1,
                "all_scores": [[None], [None], [None], [None]],
                "all_errors": [
                    ["cal_refused"],
                    ["cal_refused"],
                    ["cal_refused"],
                    ["cal_refused"],
                ],
                "refused": True,
                "cal_id": cal_candidate["id"],
                "cal_score": None,
                "variant": variant,
                "cost": {
                    "input_tokens": cal_result_high["usage"]["input_tokens"]
                    + cal_result_low["usage"]["input_tokens"],
                    "output_tokens": cal_result_high["usage"]["output_tokens"]
                    + cal_result_low["usage"]["output_tokens"],
                },
            }

        # Pass both high and low examples — build_user_message detects "prompt_high" key
        cal_example = {
            "prompt_high": cal_candidate["prompt_high"],
            "response_high": cal_candidate["response_high"],
            "score_high": cal_result_high["raw_score"],
            "prompt_low": cal_candidate["prompt_low"],
            "response_low": cal_candidate["response_low"],
            "score_low": cal_result_low["raw_score"],
        }
        cal_score_info = {
            "high": cal_result_high["raw_score"],
            "low": cal_result_low["raw_score"],
        }
        cal_cost_input = (
            cal_result_high["usage"]["input_tokens"]
            + cal_result_low["usage"]["input_tokens"]
        )
        cal_cost_output = (
            cal_result_high["usage"]["output_tokens"]
            + cal_result_low["usage"]["output_tokens"]
        )
    else:
        cal_result = await get_cal_score(
            client, cal_candidate, subset, model, temperature, semaphore
        )

        if cal_result["raw_score"] is None:
            return {
                "id": example["id"],
                "subset": subset,
                "k": 1,
                "all_scores": [[None], [None], [None], [None]],
                "all_errors": [
                    ["cal_refused"],
                    ["cal_refused"],
                    ["cal_refused"],
                    ["cal_refused"],
                ],
                "refused": True,
                "cal_id": cal_candidate["id"],
                "cal_score": None,
                "variant": variant,
                "cost": {
                    "input_tokens": cal_result["usage"]["input_tokens"],
                    "output_tokens": cal_result["usage"]["output_tokens"],
                },
            }

        cal_example = {
            "prompt": cal_candidate["prompt"],
            "response": cal_candidate["response"],
            "score": cal_result["raw_score"],
        }
        cal_score_info = cal_result["raw_score"]
        cal_cost_input = cal_result["usage"]["input_tokens"]
        cal_cost_output = cal_result["usage"]["output_tokens"]

    # Step 2: Score all 4 responses with calibration context
    all_responses = [example["chosen"][0]] + list(example["rejected"])

    async with semaphore:
        results = await asyncio.gather(
            *[
                score_response(
                    client,
                    prompt,
                    resp,
                    subset,
                    model=model,
                    temperature=temperature,
                    calibration_example=cal_example,
                )
                for resp in all_responses
            ]
        )

    all_scores = [[r["raw_score"]] for r in results]
    all_errors = [[r["error"]] for r in results]

    total_input_tokens = cal_cost_input + sum(
        r["usage"]["input_tokens"] for r in results
    )
    total_output_tokens = cal_cost_output + sum(
        r["usage"]["output_tokens"] for r in results
    )

    any_response_failed = any(r["raw_score"] is None for r in results)

    return {
        "id": example["id"],
        "subset": subset,
        "k": 1,
        "all_scores": all_scores,
        "all_errors": all_errors,
        "refused": any_response_failed,
        "cal_id": cal_candidate["id"],
        "cal_score": cal_score_info,
        "variant": variant,
        "cost": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
    }


async def main():
    parser = argparse.ArgumentParser(description="Run RB2 calibration context scoring")
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
        help="Model/deployment to use as judge",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50, help="Number of examples per subset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max concurrent API calls"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Judge temperature"
    )
    parser.add_argument(
        "--output-dir", default="results/raw", help="Output directory for results"
    )
    parser.add_argument("--subset", default=None, help="Filter to a specific subset")
    parser.add_argument(
        "--cal-variant",
        default="random",
        choices=["random", "high", "low", "both", "cross-category"],
        help="Calibration variant: high (chosen response), low (rejected response), "
        "both (one high + one low), cross-category (different subset as control), "
        "random (same as high, default)",
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

    # Sample and build per-subset queues
    ds = ds.shuffle(seed=args.seed)
    subset_counts = defaultdict(int)
    for ex in ds:
        subset_counts[ex["subset"]] += 1

    subsets_sorted = sorted(subset_counts.keys())
    subset_targets = {
        s: min(args.sample_size, subset_counts[s]) for s in subsets_sorted
    }

    # Build per-subset queues
    subset_queues = defaultdict(list)
    for ex in ds:
        subset_queues[ex["subset"]].append(ex)

    # Validate
    for s in subsets_sorted:
        if len(subset_queues[s]) < 2:
            print(
                f"ERROR: Subset '{s}' has only {len(subset_queues[s])} examples, need at least 2."
            )
            return

    if args.cal_variant == "cross-category" and len(subsets_sorted) < 2:
        print("ERROR: cross-category variant requires at least 2 subsets.")
        return

    print(
        f"\nSampling {args.sample_size} per subset (calibration variant: {args.cal_variant}):"
    )
    for s in subsets_sorted:
        print(f"  {s}: {subset_targets[s]} / {subset_counts[s]}")

    # Score
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not set.")
        return
    client = AsyncAzureOpenAI(api_version="2025-04-01-preview")
    semaphore = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)

    print(
        f"\nScoring with {args.model} (temp={args.temperature}, concurrency={args.concurrency})..."
    )
    print("Calibration scores are cached — each cal example is scored at most once.")
    print("Refused examples will be replaced with new ones from the same subset.\n")
    start_time = time.time()

    results = []
    valid_per_subset = defaultdict(int)
    refused_total = 0

    for subset in subsets_sorted:
        queue = subset_queues[subset]
        queue_idx = 0
        subset_target = subset_targets[subset]

        while valid_per_subset[subset] < subset_target and queue_idx < len(queue):
            needed = subset_target - valid_per_subset[subset]
            batch = queue[queue_idx : queue_idx + needed]
            queue_idx += len(batch)

            cal_candidates = [
                pick_calibration_candidate(
                    ex, queue, args.cal_variant, rng, subset, subset_queues
                )
                for ex in batch
            ]

            tasks = [
                score_example(
                    client,
                    ex,
                    args.model,
                    args.temperature,
                    semaphore,
                    cal,
                    args.cal_variant,
                )
                for ex, cal in zip(batch, cal_candidates)
            ]
            batch_results = await tqdm_asyncio.gather(*tasks, desc=subset)

            for r in batch_results:
                if not r["refused"]:
                    results.append(r)
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

    elapsed = time.time() - start_time
    print(
        f"\nCompleted in {elapsed:.1f}s ({refused_total} refused, {len(results)} valid)"
    )
    print(f"Calibration cache hits: {len(cal_score_cache)} unique cal examples scored")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = args.model.split("/")[-1].replace("-", "_")
    output_file = output_dir / f"calibration_{args.cal_variant}_{model_short}.jsonl"

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {output_file}")

    # Token usage
    total_input_tokens = sum(r["cost"]["input_tokens"] for r in results)
    total_output_tokens = sum(r["cost"]["output_tokens"] for r in results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nVariant: {args.cal_variant}")
    print(f"Examples: {len(results)} valid, {refused_total} refused")
    print(f"Wall clock: {elapsed:.1f}s")
    print(f"Avg time per example: {elapsed / len(results):.1f}s")

    for subset in subsets_sorted:
        count = sum(1 for r in results if r["subset"] == subset)
        print(f"  {subset}: {count}")

    print(f"\nTokens:")
    print(f"  Input:  {total_input_tokens:,}")
    print(f"  Output: {total_output_tokens:,}")


if __name__ == "__main__":
    asyncio.run(main())

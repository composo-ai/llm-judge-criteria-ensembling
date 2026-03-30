import argparse
import asyncio
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()
from collections import defaultdict
from pathlib import Path

import datasets
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio

from judge import score_response_n


async def score_example_ensemble(
    client: AsyncAzureOpenAI,
    example: dict,
    model: str,
    temperature: float,
    k: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    subset = example["subset"]
    prompt = example["prompt"]

    # All 4 responses: chosen[0] first, then all rejected
    all_responses = [example["chosen"][0]] + list(example["rejected"])

    # 4 API calls (one per response), each returning k completions
    async with semaphore:
        all_results = await asyncio.gather(
            *[
                score_response_n(
                    client,
                    prompt,
                    resp,
                    subset,
                    n=k,
                    model=model,
                    temperature=temperature,
                )
                for resp in all_responses
            ]
        )

    all_scores = []
    all_errors = []
    total_input_tokens = 0
    total_output_tokens = 0

    for r in all_results:
        all_scores.append(r["scores"])
        all_errors.append(r["errors"])
        total_input_tokens += r["usage"]["input_tokens"]
        total_output_tokens += r["usage"]["output_tokens"]

    # An example is refused if ALL completions for any response failed
    any_response_fully_failed = any(
        all(s is None for s in scores) for scores in all_scores
    )

    return {
        "id": example["id"],
        "subset": subset,
        "k": k,
        "all_scores": all_scores,
        "all_errors": all_errors,
        "refused": any_response_fully_failed,
        "cost": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
    }


async def main():
    parser = argparse.ArgumentParser(description="Run RB2 ensemble scoring")
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
        help="Model/deployment to use as judge",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of examples per subset",
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
        "--k", type=int, default=8, help="Number of completions per response"
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

    # Sample size is per category
    ds = ds.shuffle(seed=args.seed)
    subset_counts = defaultdict(int)
    for ex in ds:
        subset_counts[ex["subset"]] += 1

    subsets_sorted = sorted(subset_counts.keys())
    subset_targets = {
        s: min(args.sample_size, subset_counts[s]) for s in subsets_sorted
    }

    print(f"\nSampling {args.sample_size} per subset, k={args.k}:")
    for s in subsets_sorted:
        print(f"  {s}: {subset_targets[s]} / {subset_counts[s]}")

    total_examples = sum(subset_targets.values())
    total_calls = total_examples * 4 * args.k
    print(
        f"\nTotal: {total_examples} examples × 4 responses × {args.k} calls = {total_calls} API calls"
    )

    # Build per-subset example queues
    subset_queues = defaultdict(list)
    for ex in ds:
        subset_queues[ex["subset"]].append(ex)

    # Score
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not set. Export it before running.")
        return
    client = AsyncAzureOpenAI(api_version="2025-04-01-preview")
    semaphore = asyncio.Semaphore(args.concurrency)

    print(
        f"\nScoring with {args.model} (temp={args.temperature}, concurrency={args.concurrency})..."
    )
    print("Refused examples will be replaced with new ones from the same subset.\n")
    start_time = time.time()

    results = []
    valid_per_subset = defaultdict(int)
    refused_total = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for subset in subsets_sorted:
        queue = subset_queues[subset]
        queue_idx = 0
        subset_target = subset_targets[subset]

        while valid_per_subset[subset] < subset_target and queue_idx < len(queue):
            needed = subset_target - valid_per_subset[subset]
            batch = queue[queue_idx : queue_idx + needed]
            queue_idx += len(batch)

            tasks = [
                score_example_ensemble(
                    client, ex, args.model, args.temperature, args.k, semaphore
                )
                for ex in batch
            ]
            batch_results = await tqdm_asyncio.gather(*tasks, desc=subset)

            for r in batch_results:
                total_input_tokens += r["cost"]["input_tokens"]
                total_output_tokens += r["cost"]["output_tokens"]
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

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = args.model.split("/")[-1].replace("-", "_")
    output_file = output_dir / f"ensemble_k{args.k}_{model_short}.jsonl"

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nExamples: {len(results)} valid, {refused_total} refused")
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

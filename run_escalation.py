import argparse
import asyncio
import json
import os
import time
import numpy as np

from dotenv import load_dotenv

load_dotenv()
from collections import defaultdict
from pathlib import Path

import datasets
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio

from judge import score_response_n


async def score_example_escalation(
    mini_client: AsyncAzureOpenAI,
    full_client: AsyncAzureOpenAI,
    example: dict,
    mini_model: str,
    full_model: str,
    temperature: float,
    mini_n: int,
    full_n: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    subset = example["subset"]
    prompt = example["prompt"]

    # All 4 responses: chosen[0] first, then all rejected
    all_responses = [example["chosen"][0]] + list(example["rejected"])
    num_responses = len(all_responses)

    async with semaphore:
        # Score all responses with both models in parallel
        # 4 mini calls + 4 full calls = 8 API calls total (each with n completions)
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
                )
                for resp in all_responses
            ],
        )

    mini_results = all_results[:num_responses]
    full_results = all_results[num_responses:]

    # Extract scores and compute stds
    mini_scores = []
    mini_errors = []
    mini_stds = []
    mini_input_tokens = 0
    mini_output_tokens = 0

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
        "cost": {
            "mini_input_tokens": mini_input_tokens,
            "mini_output_tokens": mini_output_tokens,
            "full_input_tokens": full_input_tokens,
            "full_output_tokens": full_output_tokens,
        },
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Run RB2 adaptive escalation (mini → full)"
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

    print(f"\nSampling {args.sample_size} per subset:")
    print(f"  Mini: {args.mini_model} n={args.mini_n}")
    print(f"  Full: {args.full_model} n={args.full_n}")
    for s in subsets_sorted:
        print(f"  {s}: {subset_targets[s]} / {subset_counts[s]}")

    total_examples = sum(subset_targets.values())
    print(
        f"\nTotal: {total_examples} examples × 8 API calls (4 mini + 4 full), "
        f"each with n completions ({args.mini_n} mini, {args.full_n} full)"
    )

    # Build per-subset example queues
    subset_queues = defaultdict(list)
    for ex in ds:
        subset_queues[ex["subset"]].append(ex)

    # Create two clients
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

    print(f"\nScoring (temp={args.temperature}, concurrency={args.concurrency})...")
    print("Refused examples will be replaced with new ones from the same subset.\n")
    start_time = time.time()

    results = []
    valid_per_subset = defaultdict(int)
    refused_total = 0
    total_mini_input = 0
    total_mini_output = 0
    total_full_input = 0
    total_full_output = 0

    for subset in subsets_sorted:
        queue = subset_queues[subset]
        queue_idx = 0
        subset_target = subset_targets[subset]

        while valid_per_subset[subset] < subset_target and queue_idx < len(queue):
            needed = subset_target - valid_per_subset[subset]
            batch = queue[queue_idx : queue_idx + needed]
            queue_idx += len(batch)

            tasks = [
                score_example_escalation(
                    mini_client,
                    full_client,
                    ex,
                    args.mini_model,
                    args.full_model,
                    args.temperature,
                    args.mini_n,
                    args.full_n,
                    semaphore,
                )
                for ex in batch
            ]
            batch_results = await tqdm_asyncio.gather(*tasks, desc=subset)

            for r in batch_results:
                total_mini_input += r["cost"]["mini_input_tokens"]
                total_mini_output += r["cost"]["mini_output_tokens"]
                total_full_input += r["cost"]["full_input_tokens"]
                total_full_output += r["cost"]["full_output_tokens"]
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

    output_file = (
        output_dir / f"escalation_mini_n{args.mini_n}_full_n{args.full_n}.jsonl"
    )

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nExamples: {len(results)} valid, {refused_total} refused")
    print(f"Wall clock: {elapsed:.1f}s")
    if results:
        print(f"Avg time per example: {elapsed / len(results):.1f}s")

    for subset in subsets_sorted:
        count = sum(1 for r in results if r["subset"] == subset)
        print(f"  {subset}: {count}")

    print(f"\nTokens (mini):")
    print(f"  Input:  {total_mini_input:,}")
    print(f"  Output: {total_mini_output:,}")
    print(f"\nTokens (full):")
    print(f"  Input:  {total_full_input:,}")
    print(f"  Output: {total_full_output:,}")


if __name__ == "__main__":
    asyncio.run(main())

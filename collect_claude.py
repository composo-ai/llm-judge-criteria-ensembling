"""Data collection for RB2 judge experiments using Claude CLI.

Uses the `claude` CLI (Pro Max subscription) instead of the Anthropic API
to score examples with Claude Sonnet 4.6 (full) and Claude Haiku 4.5 (mini).
Each scoring call is a subprocess invocation of `claude -p` with JSON output.

Prompt variants (subset of collect.py):
  base         Vanilla RB2 prompt
  criteria     RB2 + task-specific criteria

Usage:
  python collect_claude.py --prompt base --k 8
  python collect_claude.py --prompt criteria --k 8
  python collect_claude.py --prompt base --k 8 --subset Math --sample-size 10
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import datasets

from judge import TASK_CRITERIA, build_user_message, parse_score

MODEL_MAP = {
    "full": "claude-sonnet-4-6",
    "mini": "claude-haiku-4-5-20251001",
}

# Resolve absolute path to claude CLI at import time so subprocess spawns
# don't depend on PATH inheritance across asyncio.
#
# The CLAUDE_BIN env var lets the user point at a frozen copy of the binary
# (e.g. `cp -r ~/.npm-global/lib/node_modules/@anthropic-ai/claude-code /tmp/
# claude-frozen` then `export CLAUDE_BIN=/tmp/claude-frozen/cli.js`).  This
# avoids race conditions where claude's auto-updater briefly deletes the
# binary mid-run while concurrent subprocess spawns are happening.
CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
if CLAUDE_BIN is None:
    print("ERROR: 'claude' CLI not found. Set CLAUDE_BIN env var or install on PATH.",
          file=sys.stderr)
    sys.exit(1)

# Disable claude CLI auto-updater for subprocess calls. Auto-updates can
# briefly delete and re-create the binary, causing FileNotFoundError in
# concurrent subprocess spawns.
CLAUDE_ENV = {
    **os.environ,
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_AUTO_UPDATE": "1",
    "CLAUDE_CONFIG_DIR": os.path.expanduser("~/.claude-bigdaddy"),
}


# ---------------------------------------------------------------------------
# Resume helpers (same as collect.py)
# ---------------------------------------------------------------------------

def load_completed_ids(path: Path) -> set[str]:
    ids = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                ids.add(json.loads(line)["id"])
    return ids


def output_filename(prompt: str, k: int) -> str:
    return f"{prompt}_claude_both_k{k}.jsonl"


# ---------------------------------------------------------------------------
# Claude CLI interface
# ---------------------------------------------------------------------------

RATE_LIMIT_KEYWORDS = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "429",
    "too many requests",
    "quota",
    "usage limit",
    "5-hour",
    "5 hour",
    "five hour",
    "reset",
    "try again",
    "exceeded",
)

# Permanent content-policy refusals from the Claude Code wrapper. These do
# not get better with retries, so we skip the call immediately and let the
# example be marked as refused.
POLICY_REFUSAL_KEYWORDS = (
    "claude code is unable to respond",
    "violate our usage policy",
    "violate our acceptable use",
    "violates our usage policy",
)


def _looks_like_rate_limit(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in RATE_LIMIT_KEYWORDS)


def _looks_like_policy_refusal(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in POLICY_REFUSAL_KEYWORDS)


async def call_claude_cli(
    user_message: str,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout_s: int = 120,
    max_transient_retries: int = 5,
    max_parse_retries: int = 3,
) -> dict:
    """Call the claude CLI once and return parsed result.

    Retry policy:
      - Rate-limit errors: retry indefinitely with backoff capped at 60s
      - Other transient errors (timeouts, json parse, non-zero exit without
        rate-limit keywords): retry up to max_transient_retries times
      - Score parse failures: retry up to max_parse_retries times
      - "Not logged in": fatal, exit immediately
    """
    transient_failures = 0
    parse_failures = 0
    rate_limit_attempts = 0

    while True:
        try:
            async with semaphore:
                proc = await asyncio.create_subprocess_exec(
                    CLAUDE_BIN, "-p",
                    "--output-format", "json",
                    "--model", model,
                    "--effort", "low",
                    "--system-prompt", "",
                    "--tools", "",
                    "--no-session-persistence",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=CLAUDE_ENV,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=user_message.encode()),
                    timeout=timeout_s,
                )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            transient_failures += 1
            if transient_failures > max_transient_retries:
                return {
                    "score": None,
                    "error": "timeout (max retries)",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            wait = min(2 ** transient_failures, 60)
            print(f"    [retry] timeout ({model}) "
                  f"[{transient_failures}/{max_transient_retries}], "
                  f"waiting {wait}s...", flush=True)
            await asyncio.sleep(wait)
            continue
        except FileNotFoundError:
            # Claude CLI auto-update may briefly delete the binary.
            # Treat as transient and retry.
            transient_failures += 1
            if transient_failures > max_transient_retries:
                return {
                    "score": None,
                    "error": f"binary missing: {CLAUDE_BIN}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            wait = min(2 ** transient_failures, 60)
            print(f"    [retry] claude binary missing (auto-update?), "
                  f"waiting {wait}s "
                  f"[{transient_failures}/{max_transient_retries}]...",
                  flush=True)
            await asyncio.sleep(wait)
            continue

        if proc.returncode != 0:
            # Try to extract a clean error message from the JSON output
            # (claude -p --output-format json puts errors in result field).
            stderr_text = stderr.decode().strip()
            stdout_text = stdout.decode().strip()
            err_text = stderr_text or stdout_text
            try:
                err_data = json.loads(stdout_text)
                if isinstance(err_data, dict):
                    err_text = err_data.get("result") or err_text
            except (json.JSONDecodeError, ValueError):
                pass
            # Policy refusals are permanent — give up immediately, no retry.
            if _looks_like_policy_refusal(err_text):
                return {
                    "score": None,
                    "error": "policy_refusal",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            if _looks_like_rate_limit(err_text):
                rate_limit_attempts += 1
                wait = min(2 ** min(rate_limit_attempts, 6), 60)
                print(f"    [rate-limit] cli exit {proc.returncode} ({model}): "
                      f"{err_text} -- waiting {wait}s "
                      f"(attempt {rate_limit_attempts})...", flush=True)
                await asyncio.sleep(wait)
                continue
            transient_failures += 1
            if transient_failures > max_transient_retries:
                return {
                    "score": None,
                    "error": f"cli_error rc={proc.returncode}: {err_text}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            wait = min(2 ** transient_failures, 60)
            print(f"    [retry] cli exit {proc.returncode} ({model}) "
                  f"[{transient_failures}/{max_transient_retries}]: "
                  f"{err_text} -- waiting {wait}s...", flush=True)
            await asyncio.sleep(wait)
            continue

        try:
            data = json.loads(stdout.decode())
        except json.JSONDecodeError:
            transient_failures += 1
            if transient_failures > max_transient_retries:
                return {
                    "score": None,
                    "error": "json_parse_error (max retries)",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            wait = min(2 ** transient_failures, 60)
            print(f"    [retry] json parse error ({model}) "
                  f"[{transient_failures}/{max_transient_retries}], "
                  f"waiting {wait}s...", flush=True)
            await asyncio.sleep(wait)
            continue

        if data.get("is_error"):
            error_msg = data.get("result", "unknown CLI error")
            if "Not logged in" in error_msg:
                print(f"FATAL: {error_msg}", file=sys.stderr, flush=True)
                sys.exit(1)
            # Policy refusals are permanent — give up immediately, no retry.
            if _looks_like_policy_refusal(error_msg):
                return {
                    "score": None,
                    "error": "policy_refusal",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            if _looks_like_rate_limit(error_msg):
                rate_limit_attempts += 1
                wait = min(2 ** min(rate_limit_attempts, 6), 60)
                print(f"    [rate-limit] cli is_error ({model}): {error_msg} "
                      f"-- waiting {wait}s (attempt {rate_limit_attempts})...",
                      flush=True)
                await asyncio.sleep(wait)
                continue
            transient_failures += 1
            if transient_failures > max_transient_retries:
                return {
                    "score": None,
                    "error": f"cli_error: {error_msg}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            wait = min(2 ** transient_failures, 60)
            print(f"    [retry] cli error ({model}) "
                  f"[{transient_failures}/{max_transient_retries}]: "
                  f"{error_msg} -- waiting {wait}s...", flush=True)
            await asyncio.sleep(wait)
            continue

        # Successful CLI call — extract score
        usage = data.get("usage", {})
        input_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        output_tokens = usage.get("output_tokens", 0)

        result_text = data.get("result", "")
        score = parse_score(result_text)

        if score is None:
            parse_failures += 1
            if parse_failures < max_parse_retries:
                await asyncio.sleep(1)
                continue
            return {
                "score": None,
                "error": "no_score_found",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        return {
            "score": score,
            "error": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


async def score_response_k(
    prompt: str,
    response: str,
    subset: str,
    k: int,
    model: str,
    semaphore: asyncio.Semaphore,
    criteria: str | None = None,
) -> dict:
    """Score a response k times via k independent CLI calls."""
    user_message = build_user_message(prompt, response, subset, criteria=criteria)

    tasks = [call_claude_cli(user_message, model, semaphore) for _ in range(k)]
    results = await asyncio.gather(*tasks)

    scores = [r["score"] for r in results]
    errors = [r["error"] for r in results]
    total_input = sum(r["input_tokens"] for r in results)
    total_output = sum(r["output_tokens"] for r in results)

    return {
        "scores": scores,
        "errors": errors,
        "usage": {"input_tokens": total_input, "output_tokens": total_output},
    }


# ---------------------------------------------------------------------------
# Core scoring (per example)
# ---------------------------------------------------------------------------

async def score_example(
    example: dict,
    k: int,
    semaphore: asyncio.Semaphore,
    prompt_variant: str,
    criteria: str | None = None,
) -> dict:
    prompt = example["prompt"]
    all_responses = [example["chosen"][0]] + list(example["rejected"])

    # Score all responses with both models concurrently
    tasks = []
    for model_key, model_name in MODEL_MAP.items():
        for resp in all_responses:
            tasks.append((model_key, score_response_k(
                prompt, resp, example["subset"], k, model_name, semaphore, criteria,
            )))

    # Gather all results
    gathered = await asyncio.gather(*[t[1] for t in tasks])

    # Organize by model key
    result: dict = {
        "id": example["id"],
        "subset": example["subset"],
        "prompt_variant": prompt_variant,
        "k": k,
    }
    cost: dict[str, int] = {}
    idx = 0
    for model_key in MODEL_MAP:
        scores, errors = [], []
        inp_tok, out_tok = 0, 0
        for _ in all_responses:
            r = gathered[idx]
            scores.append(r["scores"])
            errors.append(r["errors"])
            inp_tok += r["usage"]["input_tokens"]
            out_tok += r["usage"]["output_tokens"]
            idx += 1
        result[f"{model_key}_scores"] = scores
        result[f"{model_key}_errors"] = errors
        cost[f"{model_key}_input_tokens"] = inp_tok
        cost[f"{model_key}_output_tokens"] = out_tok

    result["cost"] = cost

    # Refused if ALL scores None for any response across ALL models
    refused = False
    for ri in range(len(all_responses)):
        if all(
            all(s is None for s in result[f"{mk}_scores"][ri])
            for mk in MODEL_MAP
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
        description="Collect RB2 judge scores using Claude CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prompt", required=True, choices=["base", "criteria"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=999,
                        help="Max examples per subset (999 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--subset", default=None)
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

    use_criteria = args.prompt == "criteria"

    print(f"\n  prompt={args.prompt}  k={args.k}  concurrency={args.concurrency}")
    print(f"  models: full={MODEL_MAP['full']}, mini={MODEL_MAP['mini']}")
    for s in subsets:
        print(f"  {s}: {targets[s]} / {len(subset_queues[s])}")

    # --- Output ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / output_filename(args.prompt, args.k)
    completed = load_completed_ids(out_file)
    if completed:
        print(f"  Resuming: {len(completed)} done")

    sem = asyncio.Semaphore(args.concurrency)

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
            # Build batch of uncompleted examples. Keep the batch small so
            # each example's calls don't get scattered across thousands of
            # queued tasks waiting on the concurrency semaphore — that makes
            # the first example land MUCH later than necessary.
            batch = []
            batch_size = min(3, target - valid_per_subset[subset])
            while len(batch) < batch_size and qi < len(queue):
                ex = queue[qi]; qi += 1
                if ex["id"] not in completed:
                    batch.append(ex)
            if not batch:
                break

            # Score batch — write each example as it completes (not after the
            # whole batch) so a crash mid-batch doesn't lose finished work.
            tasks = [
                asyncio.create_task(
                    score_example(ex, args.k, sem, args.prompt, criteria)
                )
                for ex in batch
            ]

            print(f"  {subset}: scoring batch of {len(batch)} "
                  f"({valid_per_subset[subset]}/{target} done)", flush=True)

            for fut in asyncio.as_completed(tasks):
                r = await fut
                if not r["refused"]:
                    outf.write(json.dumps(r) + "\n")
                    outf.flush()
                    n_valid += 1
                    valid_per_subset[subset] += 1
                    print(f"    [done] {r['id']} ({subset}) "
                          f"[{valid_per_subset[subset]}/{target}]", flush=True)
                else:
                    n_refused += 1
                    # Summarise why it was refused so we can track refusal
                    # reasons (e.g., policy_refusal vs other transient errors)
                    err_counts: dict[str, int] = defaultdict(int)
                    for mk in MODEL_MAP:
                        for resp_errs in r[f"{mk}_errors"]:
                            for e in resp_errs:
                                if e:
                                    err_counts[e] += 1
                    summary = ", ".join(f"{e}×{c}" for e, c in err_counts.items())
                    print(f"  Refused {r['id']} in {subset}: {summary}",
                          flush=True)

        if valid_per_subset[subset] < target:
            print(f"  WARNING: {valid_per_subset[subset]}/{target} for {subset}")

    outf.close()
    elapsed = time.time() - t0

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"DONE: {args.prompt} | claude (both) | k={args.k}")
    print(f"{'='*60}")
    print(f"  {out_file}")
    print(f"  {n_valid} valid, {n_refused} refused, {elapsed:.0f}s")
    for s in subsets:
        print(f"  {s}: {valid_per_subset[s]}")

    totals: dict[str, int] = defaultdict(int)
    with open(out_file) as f:
        for line in f:
            for key, val in json.loads(line).get("cost", {}).items():
                totals[key] += val
    for key in sorted(totals):
        print(f"  {key}: {totals[key]:,}")


if __name__ == "__main__":
    asyncio.run(main())

"""Compute metrics for RB2 experiment results.

Importable as a module or runnable as a script:
    python analysis/compute_metrics.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# --- Pricing ---
GPT54_INPUT_PER_M = 2.50
GPT54_OUTPUT_PER_M = 15.00
GPT54_MINI_INPUT_PER_M = 0.25
GPT54_MINI_OUTPUT_PER_M = 1.50


def load_results(filepath: str | Path) -> list[dict]:
    """Load a JSONL file, filtering out refused examples."""
    results = []
    with open(filepath) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("refused", False):
                results.append(r)
    return results


def _mean_ignoring_none(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def compute_accuracy(results: list[dict], k_subset: int | None = None) -> dict:
    """Compute accuracy: fraction of examples where response 0 has the strictly highest mean score.

    Args:
        results: List of result dicts with 'all_scores' field.
        k_subset: If set, use only the first k_subset calls per response.
                  Enables diminishing returns analysis from a single k=8 run.
    """
    by_subset = defaultdict(lambda: {"n": 0, "n_correct": 0, "n_tied": 0})

    for r in results:
        all_scores = r["all_scores"]
        subset = r["subset"]

        # Compute mean score per response
        means = []
        for scores in all_scores:
            if k_subset is not None:
                scores = scores[:k_subset]
            means.append(_mean_ignoring_none(scores))

        # Skip if any response has no valid scores
        if any(m is None for m in means):
            continue

        max_score = max(means)
        winners = [i for i, m in enumerate(means) if m == max_score]

        by_subset[subset]["n"] += 1
        if len(winners) > 1:
            by_subset[subset]["n_tied"] += 1
            # Ties = incorrect
        elif winners[0] == 0:
            by_subset[subset]["n_correct"] += 1

    # Aggregate
    total_n = sum(s["n"] for s in by_subset.values())
    total_correct = sum(s["n_correct"] for s in by_subset.values())
    total_tied = sum(s["n_tied"] for s in by_subset.values())

    subset_metrics = {}
    for subset, counts in sorted(by_subset.items()):
        n = counts["n"]
        subset_metrics[subset] = {
            "accuracy": counts["n_correct"] / n if n > 0 else 0.0,
            "n": n,
            "n_correct": counts["n_correct"],
            "n_tied": counts["n_tied"],
        }

    return {
        "overall": total_correct / total_n if total_n > 0 else 0.0,
        "n": total_n,
        "n_correct": total_correct,
        "n_tied": total_tied,
        "by_subset": subset_metrics,
    }


def compute_variance_metrics(results: list[dict]) -> dict:
    """Compute variance metrics for ensemble results (k > 1).

    Returns mean score std, per-subset breakdown, and correlation between
    variance and correctness.
    """
    all_stds = []
    by_subset = defaultdict(
        lambda: {"stds_correct": [], "stds_incorrect": [], "all_stds": []}
    )

    for r in results:
        all_scores = r["all_scores"]
        subset = r["subset"]

        # Compute per-response std
        response_stds = []
        means = []
        for scores in all_scores:
            valid = [s for s in scores if s is not None]
            if len(valid) > 1:
                response_stds.append(float(np.std(valid)))
            else:
                response_stds.append(0.0)
            means.append(_mean_ignoring_none(scores))

        if any(m is None for m in means):
            continue

        example_mean_std = float(np.mean(response_stds))
        all_stds.append(example_mean_std)

        # Determine correctness
        max_score = max(means)
        winners = [i for i, m in enumerate(means) if m == max_score]
        correct = len(winners) == 1 and winners[0] == 0

        by_subset[subset]["all_stds"].append(example_mean_std)
        if correct:
            by_subset[subset]["stds_correct"].append(example_mean_std)
        else:
            by_subset[subset]["stds_incorrect"].append(example_mean_std)

    # Compute correlation between variance and correctness
    correctness_labels = []
    variance_values = []
    for r in results:
        all_scores = r["all_scores"]
        response_stds = []
        means = []
        for scores in all_scores:
            valid = [s for s in scores if s is not None]
            if len(valid) > 1:
                response_stds.append(float(np.std(valid)))
            else:
                response_stds.append(0.0)
            means.append(_mean_ignoring_none(scores))

        if any(m is None for m in means):
            continue

        max_score = max(means)
        winners = [i for i, m in enumerate(means) if m == max_score]
        correct = 1 if (len(winners) == 1 and winners[0] == 0) else 0

        correctness_labels.append(correct)
        variance_values.append(float(np.mean(response_stds)))

    corr = None
    if len(correctness_labels) >= 3:
        r_val, _ = stats.pearsonr(variance_values, correctness_labels)
        corr = float(r_val)

    subset_metrics = {}
    for subset, data in sorted(by_subset.items()):
        subset_metrics[subset] = {
            "mean_std": float(np.mean(data["all_stds"])) if data["all_stds"] else 0.0,
            "std_when_correct": (
                float(np.mean(data["stds_correct"])) if data["stds_correct"] else 0.0
            ),
            "std_when_incorrect": (
                float(np.mean(data["stds_incorrect"]))
                if data["stds_incorrect"]
                else 0.0
            ),
        }

    return {
        "mean_score_std": float(np.mean(all_stds)) if all_stds else 0.0,
        "by_subset": subset_metrics,
        "variance_correctness_correlation": corr,
    }


def compute_escalation_metrics(
    results: list[dict], thresholds: list[float] | None = None
) -> dict:
    """Compute escalation metrics across multiple variance thresholds.

    For each threshold, applies the policy: if mini std < threshold, use mini
    mean score; else use full mean score. Returns accuracy and cost at each threshold.
    """
    if not results:
        return {"thresholds": [], "reference": {}}

    # Collect all mini stds for percentile-based thresholds
    all_mini_stds = []
    for r in results:
        for std_val in r.get("mini_stds", []):
            if std_val is not None:
                all_mini_stds.append(std_val)

    if thresholds is None:
        # Combine percentile-based and fixed thresholds
        if all_mini_stds:
            pct_thresholds = [
                float(np.percentile(all_mini_stds, p)) for p in range(10, 100, 10)
            ]
        else:
            pct_thresholds = []
        fixed_thresholds = [i * 0.1 for i in range(21)]  # 0.0 to 2.0
        thresholds = sorted(set(pct_thresholds + fixed_thresholds))

    # Reference points
    def _accuracy_from_scores(results, score_key, k_sub=None):
        """Compute accuracy using a specific score field."""
        adapted = []
        for r in results:
            adapted.append(
                {
                    "id": r["id"],
                    "subset": r["subset"],
                    "all_scores": r[score_key],
                }
            )
        return compute_accuracy(adapted, k_subset=k_sub)

    reference = {
        "always_mini": _accuracy_from_scores(results, "mini_scores"),
        "always_full": _accuracy_from_scores(results, "full_scores"),
        "always_mini_k1": _accuracy_from_scores(results, "mini_scores", k_sub=1),
        "always_full_k1": _accuracy_from_scores(results, "full_scores", k_sub=1),
    }

    # Compute total full-model cost as reference for cost ratio
    total_full_cost = sum(
        r["cost"]["full_input_tokens"] / 1e6 * GPT54_INPUT_PER_M
        + r["cost"]["full_output_tokens"] / 1e6 * GPT54_OUTPUT_PER_M
        for r in results
    )
    total_mini_cost = sum(
        r["cost"]["mini_input_tokens"] / 1e6 * GPT54_MINI_INPUT_PER_M
        + r["cost"]["mini_output_tokens"] / 1e6 * GPT54_MINI_OUTPUT_PER_M
        for r in results
    )
    always_full_cost = (
        total_mini_cost + total_full_cost
    )  # We always pay for both in our data

    # Threshold sweep
    threshold_results = []
    for threshold in thresholds:
        by_subset = defaultdict(lambda: {"n": 0, "n_correct": 0})
        total_escalated = 0
        total_responses = 0

        for r in results:
            mini_scores = r["mini_scores"]
            full_scores = r["full_scores"]
            mini_stds = r.get("mini_stds", [])
            subset = r["subset"]

            final_means = []
            for i in range(len(mini_scores)):
                mini_valid = [s for s in mini_scores[i] if s is not None]
                full_valid = [s for s in full_scores[i] if s is not None]

                mini_mean = float(np.mean(mini_valid)) if mini_valid else None
                full_mean = float(np.mean(full_valid)) if full_valid else None

                std = (
                    mini_stds[i]
                    if i < len(mini_stds) and mini_stds[i] is not None
                    else 0.0
                )

                total_responses += 1
                if std >= threshold and full_mean is not None:
                    final_means.append(full_mean)
                    total_escalated += 1
                elif mini_mean is not None:
                    final_means.append(mini_mean)
                else:
                    final_means.append(None)

            if any(m is None for m in final_means):
                continue

            max_score = max(final_means)
            winners = [i for i, m in enumerate(final_means) if m == max_score]

            by_subset[subset]["n"] += 1
            if len(winners) == 1 and winners[0] == 0:
                by_subset[subset]["n_correct"] += 1

        total_n = sum(s["n"] for s in by_subset.values())
        total_correct = sum(s["n_correct"] for s in by_subset.values())
        pct_escalated = (
            total_escalated / total_responses if total_responses > 0 else 0.0
        )

        # Cost ratio: mini is always paid, full only for escalated responses
        # Approximate: cost = mini_cost + (pct_escalated * full_cost)
        effective_cost = total_mini_cost + pct_escalated * total_full_cost
        cost_ratio = effective_cost / always_full_cost if always_full_cost > 0 else 1.0

        subset_metrics = {}
        for subset, counts in sorted(by_subset.items()):
            n = counts["n"]
            subset_metrics[subset] = {
                "accuracy": counts["n_correct"] / n if n > 0 else 0.0,
                "n": n,
                "n_correct": counts["n_correct"],
            }

        threshold_results.append(
            {
                "threshold": float(threshold),
                "accuracy": total_correct / total_n if total_n > 0 else 0.0,
                "pct_escalated": pct_escalated,
                "effective_cost_ratio": cost_ratio,
                "n": total_n,
                "by_subset": subset_metrics,
            }
        )

    return {
        "thresholds": threshold_results,
        "reference": {k: v for k, v in reference.items()},
    }


def compute_escalation_metrics_blended(
    results: list[dict], thresholds: list[float] | None = None, steepness: float = 10.0
) -> dict:
    """Soft blending escalation: blend mini and full scores based on per-response variance.

    For each response i:
        w_i = sigmoid(steepness * (sigma_i - midpoint))
        final_score_i = (1 - w_i) * mini_mean_i + w_i * full_mean_i

    The threshold parameter sweeps the sigmoid midpoint.
    """
    if not results:
        return {"thresholds": [], "reference": {}}

    all_response_stds = []
    for r in results:
        for s in r.get("mini_stds", []):
            if s is not None:
                all_response_stds.append(float(s))

    if thresholds is None:
        if all_response_stds:
            pct_thresholds = [
                float(np.percentile(all_response_stds, p)) for p in range(10, 100, 10)
            ]
        else:
            pct_thresholds = []
        fixed_thresholds = [i * 0.1 for i in range(21)]
        thresholds = sorted(set(pct_thresholds + fixed_thresholds))

    # Reference points
    def _accuracy_from_scores(results, score_key, k_sub=None):
        adapted = [
            {"id": r["id"], "subset": r["subset"], "all_scores": r[score_key]}
            for r in results
        ]
        return compute_accuracy(adapted, k_subset=k_sub)

    reference = {
        "always_mini": _accuracy_from_scores(results, "mini_scores"),
        "always_full": _accuracy_from_scores(results, "full_scores"),
        "always_mini_k1": _accuracy_from_scores(results, "mini_scores", k_sub=1),
        "always_full_k1": _accuracy_from_scores(results, "full_scores", k_sub=1),
    }

    total_full_cost = sum(
        r["cost"]["full_input_tokens"] / 1e6 * GPT54_INPUT_PER_M
        + r["cost"]["full_output_tokens"] / 1e6 * GPT54_OUTPUT_PER_M
        for r in results
    )
    total_mini_cost = sum(
        r["cost"]["mini_input_tokens"] / 1e6 * GPT54_MINI_INPUT_PER_M
        + r["cost"]["mini_output_tokens"] / 1e6 * GPT54_MINI_OUTPUT_PER_M
        for r in results
    )
    always_full_cost = total_mini_cost + total_full_cost

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    threshold_results = []
    for midpoint in thresholds:
        by_subset = defaultdict(lambda: {"n": 0, "n_correct": 0})
        all_weights = []

        for r in results:
            mini_scores = r["mini_scores"]
            full_scores = r["full_scores"]
            mini_stds = r.get("mini_stds", [])
            subset = r["subset"]

            final_means = []
            response_weights = []
            for i in range(len(mini_scores)):
                std_i = float(mini_stds[i]) if i < len(mini_stds) and mini_stds[i] is not None else 0.0
                w_i = float(_sigmoid(steepness * (std_i - midpoint)))
                response_weights.append(w_i)

                mini_valid = [s for s in mini_scores[i] if s is not None]
                full_valid = [s for s in full_scores[i] if s is not None]
                mini_mean = float(np.mean(mini_valid)) if mini_valid else None
                full_mean = float(np.mean(full_valid)) if full_valid else None

                if mini_mean is not None and full_mean is not None:
                    blended = (1 - w_i) * mini_mean + w_i * full_mean
                    final_means.append(blended)
                elif full_mean is not None:
                    final_means.append(full_mean)
                elif mini_mean is not None:
                    final_means.append(mini_mean)
                else:
                    final_means.append(None)

            all_weights.extend(response_weights)

            if any(m is None for m in final_means):
                continue

            max_score = max(final_means)
            winners = [i for i, m in enumerate(final_means) if m == max_score]

            by_subset[subset]["n"] += 1
            if len(winners) == 1 and winners[0] == 0:
                by_subset[subset]["n_correct"] += 1

        total_n = sum(s["n"] for s in by_subset.values())
        total_correct = sum(s["n_correct"] for s in by_subset.values())
        mean_weight = float(np.mean(all_weights)) if all_weights else 0.0

        effective_cost = total_mini_cost + total_full_cost
        cost_ratio = effective_cost / always_full_cost if always_full_cost > 0 else 1.0

        subset_metrics = {}
        for subset, counts in sorted(by_subset.items()):
            n = counts["n"]
            subset_metrics[subset] = {
                "accuracy": counts["n_correct"] / n if n > 0 else 0.0,
                "n": n,
                "n_correct": counts["n_correct"],
            }

        threshold_results.append(
            {
                "threshold": float(midpoint),
                "accuracy": total_correct / total_n if total_n > 0 else 0.0,
                "mean_weight": mean_weight,
                "effective_cost_ratio": cost_ratio,
                "steepness": steepness,
                "n": total_n,
                "by_subset": subset_metrics,
            }
        )

    return {
        "thresholds": threshold_results,
        "reference": {k: v for k, v in reference.items()},
    }


def _piecewise_linear_n2(
    sigma: float, sigma1: float, sigma2: float, n_max: int = 8
) -> int:
    """Piecewise linear function mapping variance to ensemble size."""
    if sigma <= sigma1:
        return 1
    elif sigma >= sigma2:
        return n_max
    else:
        n2 = 1 + (sigma - sigma1) * (n_max - 1) / (sigma2 - sigma1)
        return max(1, min(n_max, round(n2)))


def compute_variance_informed_ensembling(
    results: list[dict], n_max: int = 8, grid_steps: int = 15
) -> dict:
    """Variance-informed ensembling: use mini variance to decide how many full model calls.

    For each (sigma1, sigma2) parameter pair, applies a piecewise linear function
    to map per-response mini variance to n2 (number of full model calls per response).
    Evaluates accuracy using the first n2 full scores for each response independently.
    """
    if not results:
        return {
            "grid_results": [],
            "best": None,
            "best_cost_constrained": None,
            "reference": {},
            "by_n2": [],
        }

    # Compute per-response stds
    per_response_stds = []
    all_response_stds = []
    for r in results:
        stds = []
        for s in r.get("mini_stds", []):
            val = float(s) if s is not None else 0.0
            stds.append(val)
            all_response_stds.append(val)
        per_response_stds.append(stds)

    # Reference: accuracy at each fixed n2
    def _accuracy_at_fixed_k(results, k):
        adapted = [
            {"id": r["id"], "subset": r["subset"], "all_scores": r["full_scores"]}
            for r in results
        ]
        return compute_accuracy(adapted, k_subset=k)

    by_n2 = []
    for k in range(1, n_max + 1):
        acc = _accuracy_at_fixed_k(results, k)
        by_n2.append(
            {
                "n2": k,
                "accuracy": acc["overall"],
                "n": acc["n"],
                "by_subset": {s: d["accuracy"] for s, d in acc["by_subset"].items()},
            }
        )

    # Reference points
    def _accuracy_from_scores(results, score_key, k_sub=None):
        adapted = [
            {"id": r["id"], "subset": r["subset"], "all_scores": r[score_key]}
            for r in results
        ]
        return compute_accuracy(adapted, k_subset=k_sub)

    reference = {
        "always_mini": _accuracy_from_scores(results, "mini_scores"),
        "always_full": _accuracy_from_scores(results, "full_scores"),
        "full_k1": _accuracy_from_scores(results, "full_scores", k_sub=1),
    }

    # Cost baselines
    total_full_cost = sum(
        r["cost"]["full_input_tokens"] / 1e6 * GPT54_INPUT_PER_M
        + r["cost"]["full_output_tokens"] / 1e6 * GPT54_OUTPUT_PER_M
        for r in results
    )
    total_mini_cost = sum(
        r["cost"]["mini_input_tokens"] / 1e6 * GPT54_MINI_INPUT_PER_M
        + r["cost"]["mini_output_tokens"] / 1e6 * GPT54_MINI_OUTPUT_PER_M
        for r in results
    )
    always_full_cost = total_mini_cost + total_full_cost

    # Build grid of sigma1, sigma2 from percentiles of per-response stds
    percentiles = [
        float(np.percentile(all_response_stds, p)) for p in np.linspace(0, 95, grid_steps)
    ]
    # Add some fixed values
    fixed_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    sigma_vals = sorted(set(percentiles + fixed_vals))

    grid_results = []
    for sigma1 in sigma_vals:
        for sigma2 in sigma_vals:
            if sigma2 <= sigma1:
                continue

            by_subset = defaultdict(lambda: {"n": 0, "n_correct": 0})
            total_n2 = 0
            total_examples = 0

            for i, r in enumerate(results):
                total_examples += 1

                # Take first n2_j full scores per response, where n2_j depends on that response's variance
                final_means = []
                resp_stds = per_response_stds[i]
                for j, scores in enumerate(r["full_scores"]):
                    std_j = resp_stds[j] if j < len(resp_stds) else 0.0
                    n2_j = _piecewise_linear_n2(std_j, sigma1, sigma2, n_max)
                    total_n2 += n2_j
                    valid = [s for s in scores[:n2_j] if s is not None]
                    final_means.append(float(np.mean(valid)) if valid else None)

                if any(m is None for m in final_means):
                    continue

                max_score = max(final_means)
                winners = [idx for idx, m in enumerate(final_means) if m == max_score]
                subset = r["subset"]
                by_subset[subset]["n"] += 1
                if len(winners) == 1 and winners[0] == 0:
                    by_subset[subset]["n_correct"] += 1

            total_n = sum(s["n"] for s in by_subset.values())
            total_correct = sum(s["n_correct"] for s in by_subset.values())
            num_responses = 4
            mean_n2 = total_n2 / (total_examples * num_responses) if total_examples > 0 else 1.0

            # Cost: mini always + (mean_n2 / n_max) * full
            effective_cost = total_mini_cost + (mean_n2 / n_max) * total_full_cost
            cost_ratio = (
                effective_cost / always_full_cost if always_full_cost > 0 else 1.0
            )

            subset_metrics = {}
            for subset, counts in sorted(by_subset.items()):
                n = counts["n"]
                subset_metrics[subset] = {
                    "accuracy": counts["n_correct"] / n if n > 0 else 0.0,
                    "n": n,
                    "n_correct": counts["n_correct"],
                }

            grid_results.append(
                {
                    "sigma1": float(sigma1),
                    "sigma2": float(sigma2),
                    "accuracy": total_correct / total_n if total_n > 0 else 0.0,
                    "mean_n2": mean_n2,
                    "effective_cost_ratio": cost_ratio,
                    "n": total_n,
                    "by_subset": subset_metrics,
                }
            )

    # Find best overall and best cost-constrained
    best = max(grid_results, key=lambda g: g["accuracy"]) if grid_results else None
    cost_constrained = [g for g in grid_results if g["mean_n2"] <= 2.0]
    best_cost_constrained = (
        max(cost_constrained, key=lambda g: g["accuracy"]) if cost_constrained else None
    )

    return {
        "grid_results": grid_results,
        "best": best,
        "best_cost_constrained": best_cost_constrained,
        "reference": {k: v for k, v in reference.items()},
        "by_n2": by_n2,
    }


def compute_ensemble_convergence(results: list[dict]) -> dict:
    """Compute how mini-full agreement changes with ensemble size.

    For k_subset in 1..max_k: compute mini's winner using first k_subset calls,
    full's winner using all calls. Report agreement rate and rank correlation.
    """
    if not results:
        return {"by_k": [], "by_k_and_subset": {}}

    max_mini_k = min(len(r["mini_scores"][0]) for r in results)
    full_winners = {}

    # Compute full model winners (ground truth)
    for r in results:
        means = [_mean_ignoring_none(scores) for scores in r["full_scores"]]
        if any(m is None for m in means):
            continue
        max_score = max(means)
        winners = [i for i, m in enumerate(means) if m == max_score]
        full_winners[r["id"]] = {
            "winner": winners[0] if len(winners) == 1 else -1,
            "means": means,
            "subset": r["subset"],
        }

    by_k = []
    by_k_and_subset = defaultdict(list)

    for k in range(1, max_mini_k + 1):
        agreements = []
        rank_corrs = []
        subset_data = defaultdict(lambda: {"agreements": [], "rank_corrs": []})

        for r in results:
            if r["id"] not in full_winners:
                continue

            mini_means = [
                _mean_ignoring_none(scores[:k]) for scores in r["mini_scores"]
            ]
            if any(m is None for m in mini_means):
                continue

            max_mini = max(mini_means)
            mini_winners = [i for i, m in enumerate(mini_means) if m == max_mini]
            mini_winner = mini_winners[0] if len(mini_winners) == 1 else -1

            full_info = full_winners[r["id"]]
            agreed = (
                1 if mini_winner == full_info["winner"] and mini_winner != -1 else 0
            )
            agreements.append(agreed)

            # Spearman rank correlation between mini and full mean scores
            if len(mini_means) >= 3:
                rho, _ = stats.spearmanr(mini_means, full_info["means"])
                if not np.isnan(rho):
                    rank_corrs.append(float(rho))

            subset = full_info["subset"]
            subset_data[subset]["agreements"].append(agreed)
            if len(mini_means) >= 3:
                rho, _ = stats.spearmanr(mini_means, full_info["means"])
                if not np.isnan(rho):
                    subset_data[subset]["rank_corrs"].append(float(rho))

        entry = {
            "k": k,
            "agreement_with_full": float(np.mean(agreements)) if agreements else 0.0,
            "rank_correlation": float(np.mean(rank_corrs)) if rank_corrs else None,
        }
        by_k.append(entry)

        for subset, data in subset_data.items():
            by_k_and_subset[subset].append(
                {
                    "k": k,
                    "agreement_with_full": (
                        float(np.mean(data["agreements"]))
                        if data["agreements"]
                        else 0.0
                    ),
                    "rank_correlation": (
                        float(np.mean(data["rank_corrs"]))
                        if data["rank_corrs"]
                        else None
                    ),
                }
            )

    return {
        "by_k": by_k,
        "by_k_and_subset": dict(by_k_and_subset),
    }


def compute_cost(
    results: list[dict],
    price_per_1m_input: float = GPT54_INPUT_PER_M,
    price_per_1m_output: float = GPT54_OUTPUT_PER_M,
) -> dict:
    """Compute dollar cost from token counts."""
    # Detect format from first result
    is_escalation_format = "mini_input_tokens" in results[0].get("cost", {})

    if is_escalation_format:
        total_mini_cost = 0.0
        total_full_cost = 0.0
        for r in results:
            cost = r["cost"]
            total_mini_cost += (
                cost["mini_input_tokens"] / 1e6 * GPT54_MINI_INPUT_PER_M
                + cost["mini_output_tokens"] / 1e6 * GPT54_MINI_OUTPUT_PER_M
            )
            total_full_cost += (
                cost["full_input_tokens"] / 1e6 * price_per_1m_input
                + cost["full_output_tokens"] / 1e6 * price_per_1m_output
            )
        total = total_mini_cost + total_full_cost
        return {
            "total_cost": total,
            "mini_cost": total_mini_cost,
            "full_cost": total_full_cost,
            "n": len(results),
            "cost_per_example": total / len(results) if results else 0,
        }

    total_input = 0
    total_output = 0
    for r in results:
        cost = r.get("cost", {})
        total_input += cost.get("input_tokens", 0)
        total_output += cost.get("output_tokens", 0)

    input_cost = total_input / 1e6 * price_per_1m_input
    output_cost = total_output / 1e6 * price_per_1m_output
    total = input_cost + output_cost

    return {
        "total_cost": total,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "n": len(results),
        "cost_per_example": total / len(results) if results else 0,
    }


def detect_file_type(filename: str) -> str:
    """Detect experiment type from filename."""
    name = Path(filename).name
    for prefix in [
        "combined",
        "baseline",
        "criteria",
        "calibration",
        "ensemble",
        "escalation",
    ]:
        if name.startswith(prefix):
            return prefix
    return "unknown"


def main():
    results_dir = Path("results/raw")
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(
            "ERROR: results/raw/ directory not found. Run from experiments/llm-judge-ablations/"
        )
        sys.exit(1)

    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found in results/raw/")
        sys.exit(1)

    all_metrics = {}

    for filepath in jsonl_files:
        file_type = detect_file_type(filepath.name)
        key = filepath.stem
        print(f"\n{'='*60}")
        print(f"Processing: {filepath.name} (type: {file_type})")
        print(f"{'='*60}")

        results = load_results(filepath)
        print(f"  Loaded {len(results)} examples (refused filtered out)")

        if not results:
            print("  No valid results, skipping.")
            continue

        metrics = {"file": filepath.name, "type": file_type, "n": len(results)}

        # Accuracy — escalation/combined use full_scores instead of all_scores
        if file_type in ("escalation", "combined"):
            acc_results = [
                {"id": r["id"], "subset": r["subset"], "all_scores": r["full_scores"]}
                for r in results
            ]
            acc = compute_accuracy(acc_results)
        else:
            acc = compute_accuracy(results)
        metrics["accuracy"] = acc
        print(
            f"\n  Overall accuracy: {acc['overall']:.3f} ({acc['n_correct']}/{acc['n']})"
        )
        print(f"  Tied: {acc['n_tied']}")
        for subset, sub_acc in sorted(acc["by_subset"].items()):
            print(
                f"    {subset}: {sub_acc['accuracy']:.3f} ({sub_acc['n_correct']}/{sub_acc['n']}, tied={sub_acc['n_tied']})"
            )

        # Cost
        cost = compute_cost(results)
        metrics["cost"] = cost
        print(
            f"\n  Cost: ${cost['total_cost']:.4f} (${cost['cost_per_example']:.6f}/example)"
        )

        # Type-specific metrics
        if file_type == "ensemble":
            # Variance metrics
            var_metrics = compute_variance_metrics(results)
            metrics["variance"] = var_metrics
            print(f"\n  Mean score std: {var_metrics['mean_score_std']:.3f}")
            print(
                f"  Variance-correctness correlation: {var_metrics['variance_correctness_correlation']}"
            )
            for subset, sub_var in sorted(var_metrics["by_subset"].items()):
                print(
                    f"    {subset}: std={sub_var['mean_std']:.3f} (correct={sub_var['std_when_correct']:.3f}, incorrect={sub_var['std_when_incorrect']:.3f})"
                )

            # Diminishing returns
            k_max = results[0]["k"] if results else 1
            diminishing = {}
            for k_sub in range(1, k_max + 1):
                sub_acc = compute_accuracy(results, k_subset=k_sub)
                diminishing[k_sub] = sub_acc
            metrics["diminishing_returns"] = {
                str(k): {
                    "overall": v["overall"],
                    "by_subset": {s: d["accuracy"] for s, d in v["by_subset"].items()},
                }
                for k, v in diminishing.items()
            }
            print(f"\n  Diminishing returns (accuracy by k):")
            for k_sub in range(1, k_max + 1):
                print(f"    k={k_sub}: {diminishing[k_sub]['overall']:.3f}")

        elif file_type == "escalation":
            # Per-response escalation (existing)
            esc_metrics = compute_escalation_metrics(results)
            metrics["escalation"] = esc_metrics
            print(f"\n  Reference points:")
            for ref_name, ref_acc in esc_metrics["reference"].items():
                print(f"    {ref_name}: {ref_acc['overall']:.3f}")

            print(f"\n  Per-response escalation (selected thresholds):")
            for tr in esc_metrics["thresholds"][::5]:
                print(
                    f"    threshold={tr['threshold']:.2f}: acc={tr['accuracy']:.3f}, escalated={tr['pct_escalated']:.1%}, cost_ratio={tr['effective_cost_ratio']:.2f}"
                )

            # Soft blending
            esc_blended = compute_escalation_metrics_blended(results)
            metrics["escalation_blended"] = esc_blended
            print(f"\n  Soft blending escalation (selected midpoints):")
            for tr in esc_blended["thresholds"][::5]:
                print(
                    f"    midpoint={tr['threshold']:.2f}: acc={tr['accuracy']:.3f}, mean_weight={tr['mean_weight']:.3f}, cost_ratio={tr['effective_cost_ratio']:.2f}"
                )

            # Variance-informed ensembling
            vie = compute_variance_informed_ensembling(results)
            metrics["variance_informed_ensembling"] = vie
            if vie["best"]:
                b = vie["best"]
                print(f"\n  Variance-informed ensembling (best):")
                print(
                    f"    σ1={b['sigma1']:.4f}, σ2={b['sigma2']:.4f}: acc={b['accuracy']:.3f}, mean_n2={b['mean_n2']:.1f}, cost_ratio={b['effective_cost_ratio']:.2f}"
                )
            if vie["best_cost_constrained"]:
                b = vie["best_cost_constrained"]
                print(f"  Best cost-constrained (mean_n2 ≤ 2.0):")
                print(
                    f"    σ1={b['sigma1']:.4f}, σ2={b['sigma2']:.4f}: acc={b['accuracy']:.3f}, mean_n2={b['mean_n2']:.1f}, cost_ratio={b['effective_cost_ratio']:.2f}"
                )
            print(f"\n  Fixed ensemble size reference:")
            for entry in vie["by_n2"]:
                print(f"    n2={entry['n2']}: {entry['accuracy']:.3f}")

            conv = compute_ensemble_convergence(results)
            metrics["convergence"] = conv
            if conv["by_k"]:
                print(f"\n  Mini-full convergence:")
                for entry in conv["by_k"]:
                    rc = (
                        f"{entry['rank_correlation']:.3f}"
                        if entry["rank_correlation"] is not None
                        else "N/A"
                    )
                    print(
                        f"    k={entry['k']}: agreement={entry['agreement_with_full']:.3f}, rank_corr={rc}"
                    )

        elif file_type == "combined":
            # Combined = escalation format + criteria + calibration
            # Run all the same analyses as escalation
            esc_metrics = compute_escalation_metrics(results)
            metrics["escalation"] = esc_metrics
            print(f"\n  Reference points:")
            for ref_name, ref_acc in esc_metrics["reference"].items():
                print(f"    {ref_name}: {ref_acc['overall']:.3f}")

            esc_blended = compute_escalation_metrics_blended(results)
            metrics["escalation_blended"] = esc_blended
            if esc_blended.get("thresholds"):
                best_blend = max(esc_blended["thresholds"], key=lambda t: t["accuracy"])
                print(
                    f"\n  Best soft blend: acc={best_blend['accuracy']:.3f}, cost_ratio={best_blend['effective_cost_ratio']:.2f}"
                )

            vie = compute_variance_informed_ensembling(results)
            metrics["variance_informed_ensembling"] = vie
            if vie["best"]:
                b = vie["best"]
                print(
                    f"\n  Best var-informed: acc={b['accuracy']:.3f}, mean_n2={b['mean_n2']:.1f}"
                )

            # Diminishing returns for both models
            k_max = results[0].get("full_n", 8) if results else 8
            diminishing = {}
            for k_sub in range(1, k_max + 1):
                adapted = [
                    {
                        "id": r["id"],
                        "subset": r["subset"],
                        "all_scores": r["full_scores"],
                    }
                    for r in results
                ]
                sub_acc = compute_accuracy(adapted, k_subset=k_sub)
                diminishing[k_sub] = sub_acc
            metrics["diminishing_returns"] = {
                str(k): {
                    "overall": v["overall"],
                    "by_subset": {s: d["accuracy"] for s, d in v["by_subset"].items()},
                }
                for k, v in diminishing.items()
            }
            print(f"\n  Diminishing returns (full model, accuracy by k):")
            for k_sub in range(1, k_max + 1):
                print(f"    k={k_sub}: {diminishing[k_sub]['overall']:.3f}")

            conv = compute_ensemble_convergence(results)
            metrics["convergence"] = conv

        all_metrics[key] = metrics

    # Save all metrics
    output_path = tables_dir / "all_metrics.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n\nAll metrics saved to {output_path}")


if __name__ == "__main__":
    main()

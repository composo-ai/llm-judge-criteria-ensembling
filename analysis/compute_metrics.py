"""Compute metrics for RB2 experiment results.

Reads the unified collection format produced by collect.py and derives
all experimental conditions offline. Supports train/test split for
parameter-optimised conditions and bootstrap confidence intervals.

Usage:
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
GPT54_MINI_INPUT_PER_M = 0.75
GPT54_MINI_OUTPUT_PER_M = 4.50
GPT54_NANO_INPUT_PER_M = 0.20
GPT54_NANO_OUTPUT_PER_M = 1.25

RAW_DIR = Path("results/raw")
TABLES_DIR = Path("results/tables")


# ===================================================================
# Data loading
# ===================================================================

def load_collection(filepath: str | Path) -> list[dict]:
    """Load a JSONL collection, filtering refused examples."""
    results = []
    with open(filepath) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("refused", False):
                results.append(r)
    return results


def _mean_ignoring_none(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


# ===================================================================
# Train/test split
# ===================================================================

def train_test_split(
    data: list[dict], test_frac: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Stratified split by subset. Returns (train, test)."""
    rng = np.random.RandomState(seed)
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in data:
        by_subset[r["subset"]].append(r)

    train, test = [], []
    for subset in sorted(by_subset):
        items = by_subset[subset]
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_frac))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


def intersect_collections(collections: dict[str, list[dict]]) -> set[str]:
    """Return IDs present in all collections."""
    id_sets = [set(r["id"] for r in c) for c in collections.values()]
    return set.intersection(*id_sets) if id_sets else set()


# ===================================================================
# Accuracy (core metric)
# ===================================================================

def compute_accuracy(
    data: list[dict],
    model: str = "full",
    k_subset: int | None = None,
) -> dict:
    """Fraction of examples where response 0 has strictly highest mean score.

    Args:
        data: Collection results with {model}_scores fields.
        model: Which model's scores to use ("mini" or "full").
        k_subset: Use only first k scores per response (for diminishing returns).
    """
    score_key = f"{model}_scores"
    by_subset = defaultdict(lambda: {"n": 0, "n_correct": 0, "n_tied": 0})

    for r in data:
        scores_per_resp = r.get(score_key)
        if scores_per_resp is None:
            continue
        subset = r["subset"]
        means = []
        for scores in scores_per_resp:
            s = scores[:k_subset] if k_subset else scores
            means.append(_mean_ignoring_none(s))
        if any(m is None for m in means):
            continue

        max_score = max(means)
        winners = [i for i, m in enumerate(means) if m == max_score]
        by_subset[subset]["n"] += 1
        if len(winners) > 1:
            by_subset[subset]["n_tied"] += 1
        elif winners[0] == 0:
            by_subset[subset]["n_correct"] += 1

    total_n = sum(s["n"] for s in by_subset.values())
    total_correct = sum(s["n_correct"] for s in by_subset.values())
    total_tied = sum(s["n_tied"] for s in by_subset.values())

    subset_metrics = {}
    for sub, counts in sorted(by_subset.items()):
        n = counts["n"]
        subset_metrics[sub] = {
            "accuracy": counts["n_correct"] / n if n > 0 else 0.0,
            "n": n, "n_correct": counts["n_correct"], "n_tied": counts["n_tied"],
        }

    return {
        "overall": total_correct / total_n if total_n > 0 else 0.0,
        "n": total_n, "n_correct": total_correct, "n_tied": total_tied,
        "by_subset": subset_metrics,
    }


# ===================================================================
# Bootstrap confidence intervals
# ===================================================================

def bootstrap_accuracy_ci(
    data: list[dict],
    model: str = "full",
    k_subset: int | None = None,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for overall and per-subset accuracy."""
    rng = np.random.RandomState(seed)
    n = len(data)

    overall_accs = []
    subset_accs: dict[str, list[float]] = defaultdict(list)

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = [data[i] for i in indices]
        acc = compute_accuracy(sample, model=model, k_subset=k_subset)
        overall_accs.append(acc["overall"])
        for sub, sub_data in acc["by_subset"].items():
            subset_accs[sub].append(sub_data["accuracy"])

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    result = {
        "overall": {
            "mean": float(np.mean(overall_accs)),
            "ci_low": float(np.percentile(overall_accs, lo)),
            "ci_high": float(np.percentile(overall_accs, hi)),
        },
        "by_subset": {},
    }
    for sub in sorted(subset_accs):
        vals = subset_accs[sub]
        result["by_subset"][sub] = {
            "mean": float(np.mean(vals)),
            "ci_low": float(np.percentile(vals, lo)),
            "ci_high": float(np.percentile(vals, hi)),
        }
    return result


# ===================================================================
# Cost
# ===================================================================

_PRICING = {
    "full": (GPT54_INPUT_PER_M, GPT54_OUTPUT_PER_M),
    "mini": (GPT54_MINI_INPUT_PER_M, GPT54_MINI_OUTPUT_PER_M),
    "nano": (GPT54_NANO_INPUT_PER_M, GPT54_NANO_OUTPUT_PER_M),
}


def compute_cost(data: list[dict]) -> dict:
    """Compute actual dollar cost from token counts (total collection cost)."""
    totals = {mk: {"in": 0, "out": 0} for mk in _PRICING}
    for r in data:
        c = r.get("cost", {})
        for mk in _PRICING:
            if f"{mk}_input_tokens" in c:
                totals[mk]["in"] += c[f"{mk}_input_tokens"]
                totals[mk]["out"] += c[f"{mk}_output_tokens"]

    costs = {}
    total = 0.0
    for mk, (inp_rate, out_rate) in _PRICING.items():
        mk_cost = totals[mk]["in"] / 1e6 * inp_rate + totals[mk]["out"] / 1e6 * out_rate
        costs[f"{mk}_cost"] = mk_cost
        total += mk_cost
    n = len(data) or 1
    return {
        "total_cost": total, **costs,
        "cost_per_example": total / n, "n": len(data),
    }


def compute_deployment_cost(
    data: list[dict],
    models: str = "full",
    k: int = 1,
    collection_k: int = 8,
) -> float:
    """Estimate per-example deployment cost for a specific condition.

    Collections gather both models at k=8, but each derived condition
    uses only a subset of those calls. This function estimates what a
    condition would cost if deployed independently.

    Input tokens are charged once per API call (independent of n).
    Output tokens scale linearly with k (the n parameter).

    Args:
        models: "full", "mini", "nano", "both", or "all"
        k: number of completions per response in the deployed condition
        collection_k: k used during data collection (to scale output tokens)
    """
    model_keys = {
        "both": ["mini", "full"],
        "all": ["nano", "mini", "full"],
    }.get(models, [models])

    n = len(data) or 1
    cost = 0.0
    for r in data:
        c = r.get("cost", {})
        for mk in model_keys:
            inp_rate, out_rate = _PRICING[mk]
            inp = c.get(f"{mk}_input_tokens", 0)
            out = c.get(f"{mk}_output_tokens", 0)
            # Output tokens scale with k; input charged once
            scaled_out = out * k / collection_k
            cost += inp / 1e6 * inp_rate + scaled_out / 1e6 * out_rate
    return cost / n


# ===================================================================
# Variance metrics
# ===================================================================

def compute_variance_metrics(data: list[dict], model: str = "full") -> dict:
    """Variance as error signal analysis."""
    score_key = f"{model}_scores"
    all_stds = []
    by_subset = defaultdict(lambda: {"stds_correct": [], "stds_incorrect": [], "all_stds": []})

    for r in data:
        scores_per_resp = r.get(score_key)
        if scores_per_resp is None:
            continue

        response_stds, means = [], []
        for scores in scores_per_resp:
            valid = [s for s in scores if s is not None]
            response_stds.append(float(np.std(valid)) if len(valid) > 1 else 0.0)
            means.append(_mean_ignoring_none(scores))
        if any(m is None for m in means):
            continue

        example_std = float(np.mean(response_stds))
        all_stds.append(example_std)
        max_score = max(means)
        winners = [i for i, m in enumerate(means) if m == max_score]
        correct = len(winners) == 1 and winners[0] == 0
        subset = r["subset"]
        by_subset[subset]["all_stds"].append(example_std)
        (by_subset[subset]["stds_correct"] if correct else by_subset[subset]["stds_incorrect"]).append(example_std)

    # Pearson correlation
    corr = None
    labels, variances = [], []
    for r in data:
        scores_per_resp = r.get(score_key)
        if scores_per_resp is None:
            continue
        stds, means = [], []
        for scores in scores_per_resp:
            valid = [s for s in scores if s is not None]
            stds.append(float(np.std(valid)) if len(valid) > 1 else 0.0)
            means.append(_mean_ignoring_none(scores))
        if any(m is None for m in means):
            continue
        max_s = max(means)
        w = [i for i, m in enumerate(means) if m == max_s]
        labels.append(1 if len(w) == 1 and w[0] == 0 else 0)
        variances.append(float(np.mean(stds)))
    if len(labels) >= 3:
        corr = float(stats.pearsonr(variances, labels)[0])

    subset_metrics = {}
    for sub, d in sorted(by_subset.items()):
        subset_metrics[sub] = {
            "mean_std": float(np.mean(d["all_stds"])) if d["all_stds"] else 0.0,
            "std_when_correct": float(np.mean(d["stds_correct"])) if d["stds_correct"] else 0.0,
            "std_when_incorrect": float(np.mean(d["stds_incorrect"])) if d["stds_incorrect"] else 0.0,
        }

    return {
        "mean_score_std": float(np.mean(all_stds)) if all_stds else 0.0,
        "by_subset": subset_metrics,
        "variance_correctness_correlation": corr,
    }


# ===================================================================
# Escalation: hard routing
# ===================================================================

def _compute_mini_stds(data: list[dict]) -> list[list[float]]:
    """Compute per-response mini stds for each example."""
    result = []
    for r in data:
        stds = []
        for scores in r.get("mini_scores", []):
            valid = [s for s in scores if s is not None]
            stds.append(float(np.std(valid)) if len(valid) > 1 else 0.0)
        result.append(stds)
    return result


def compute_escalation_metrics(
    data: list[dict], thresholds: list[float] | None = None
) -> dict:
    if not data:
        return {"thresholds": [], "reference": {}}

    all_stds = _compute_mini_stds(data)
    flat_stds = [s for stds in all_stds for s in stds]

    if thresholds is None:
        pct = [float(np.percentile(flat_stds, p)) for p in range(10, 100, 10)] if flat_stds else []
        fixed = [i * 0.1 for i in range(21)]
        thresholds = sorted(set(pct + fixed))

    # Reference
    ref = {
        "mini_k8": compute_accuracy(data, model="mini"),
        "full_k8": compute_accuracy(data, model="full"),
        "mini_k1": compute_accuracy(data, model="mini", k_subset=1),
        "full_k1": compute_accuracy(data, model="full", k_subset=1),
    }

    total_full_cost = sum(
        r["cost"].get("full_input_tokens", 0) / 1e6 * GPT54_INPUT_PER_M
        + r["cost"].get("full_output_tokens", 0) / 1e6 * GPT54_OUTPUT_PER_M
        for r in data
    )
    total_mini_cost = sum(
        r["cost"].get("mini_input_tokens", 0) / 1e6 * GPT54_MINI_INPUT_PER_M
        + r["cost"].get("mini_output_tokens", 0) / 1e6 * GPT54_MINI_OUTPUT_PER_M
        for r in data
    )
    total_cost = total_mini_cost + total_full_cost

    results_list = []
    for threshold in thresholds:
        by_sub = defaultdict(lambda: {"n": 0, "n_correct": 0})
        n_escalated = n_total = 0

        for ri, r in enumerate(data):
            mini_scores = r["mini_scores"]
            full_scores = r["full_scores"]
            stds = all_stds[ri]
            final_means = []
            for j in range(len(mini_scores)):
                mini_valid = [s for s in mini_scores[j] if s is not None]
                full_valid = [s for s in full_scores[j] if s is not None]
                mini_mean = float(np.mean(mini_valid)) if mini_valid else None
                full_mean = float(np.mean(full_valid)) if full_valid else None
                std = stds[j] if j < len(stds) else 0.0
                n_total += 1
                if std >= threshold and full_mean is not None:
                    final_means.append(full_mean)
                    n_escalated += 1
                elif mini_mean is not None:
                    final_means.append(mini_mean)
                else:
                    final_means.append(None)

            if any(m is None for m in final_means):
                continue
            max_s = max(final_means)
            winners = [i for i, m in enumerate(final_means) if m == max_s]
            by_sub[r["subset"]]["n"] += 1
            if len(winners) == 1 and winners[0] == 0:
                by_sub[r["subset"]]["n_correct"] += 1

        tot_n = sum(s["n"] for s in by_sub.values())
        tot_c = sum(s["n_correct"] for s in by_sub.values())
        pct_esc = n_escalated / n_total if n_total else 0.0
        eff_cost = total_mini_cost + pct_esc * total_full_cost
        cost_ratio = eff_cost / total_cost if total_cost else 1.0

        results_list.append({
            "threshold": float(threshold),
            "accuracy": tot_c / tot_n if tot_n else 0.0,
            "pct_escalated": pct_esc,
            "effective_cost_ratio": cost_ratio,
            "n": tot_n,
        })

    return {"thresholds": results_list, "reference": ref}


# ===================================================================
# Escalation: soft blending
# ===================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_blend_accuracy(
    data: list[dict], midpoint: float, steepness: float = 10.0
) -> dict:
    """Compute accuracy using soft blend at a single midpoint."""
    all_stds = _compute_mini_stds(data)
    by_sub = defaultdict(lambda: {"n": 0, "n_correct": 0})
    weights = []

    for ri, r in enumerate(data):
        stds = all_stds[ri]
        final_means = []
        for j in range(len(r["mini_scores"])):
            std_j = stds[j] if j < len(stds) else 0.0
            w = float(_sigmoid(steepness * (std_j - midpoint)))
            weights.append(w)
            mini_valid = [s for s in r["mini_scores"][j] if s is not None]
            full_valid = [s for s in r["full_scores"][j] if s is not None]
            mini_m = float(np.mean(mini_valid)) if mini_valid else None
            full_m = float(np.mean(full_valid)) if full_valid else None
            if mini_m is not None and full_m is not None:
                final_means.append((1 - w) * mini_m + w * full_m)
            elif full_m is not None:
                final_means.append(full_m)
            elif mini_m is not None:
                final_means.append(mini_m)
            else:
                final_means.append(None)

        if any(m is None for m in final_means):
            continue
        max_s = max(final_means)
        winners = [i for i, m in enumerate(final_means) if m == max_s]
        by_sub[r["subset"]]["n"] += 1
        if len(winners) == 1 and winners[0] == 0:
            by_sub[r["subset"]]["n_correct"] += 1

    tot_n = sum(s["n"] for s in by_sub.values())
    tot_c = sum(s["n_correct"] for s in by_sub.values())

    subset_metrics = {
        sub: {"accuracy": c["n_correct"] / c["n"] if c["n"] else 0.0, "n": c["n"]}
        for sub, c in sorted(by_sub.items())
    }
    return {
        "accuracy": tot_c / tot_n if tot_n else 0.0,
        "mean_weight": float(np.mean(weights)) if weights else 0.0,
        "n": tot_n,
        "by_subset": subset_metrics,
    }


def optimise_blend(
    train: list[dict], test: list[dict], steepness: float = 10.0
) -> dict:
    """Find best blend midpoint on train, evaluate on test."""
    all_stds = _compute_mini_stds(train)
    unique_stds = sorted(set(s for stds in all_stds for s in stds))

    best_m, best_acc = 0.0, 0.0
    for m in unique_stds:
        acc = compute_blend_accuracy(train, m, steepness)["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_m = m

    train_result = compute_blend_accuracy(train, best_m, steepness)
    test_result = compute_blend_accuracy(test, best_m, steepness)

    return {
        "best_midpoint": best_m,
        "steepness": steepness,
        "train_accuracy": train_result["accuracy"],
        "test_accuracy": test_result["accuracy"],
        "test_by_subset": test_result["by_subset"],
        "test_n": test_result["n"],
        "test_mean_weight": test_result["mean_weight"],
    }


# ===================================================================
# Escalation: variance-informed ensembling
# ===================================================================

def _piecewise_linear_n2(sigma, sigma1, sigma2, n_max=8):
    if sigma <= sigma1:
        return 1
    elif sigma >= sigma2:
        return n_max
    else:
        return max(1, min(n_max, round(1 + (sigma - sigma1) * (n_max - 1) / (sigma2 - sigma1))))


def compute_var_informed_accuracy(
    data: list[dict], sigma1: float, sigma2: float, n_max: int = 8
) -> dict:
    all_stds = _compute_mini_stds(data)
    by_sub = defaultdict(lambda: {"n": 0, "n_correct": 0})
    total_n2 = 0

    for ri, r in enumerate(data):
        stds = all_stds[ri]
        final_means = []
        for j, scores in enumerate(r["full_scores"]):
            std_j = stds[j] if j < len(stds) else 0.0
            n2 = _piecewise_linear_n2(std_j, sigma1, sigma2, n_max)
            total_n2 += n2
            valid = [s for s in scores[:n2] if s is not None]
            final_means.append(float(np.mean(valid)) if valid else None)

        if any(m is None for m in final_means):
            continue
        max_s = max(final_means)
        winners = [i for i, m in enumerate(final_means) if m == max_s]
        by_sub[r["subset"]]["n"] += 1
        if len(winners) == 1 and winners[0] == 0:
            by_sub[r["subset"]]["n_correct"] += 1

    tot_n = sum(s["n"] for s in by_sub.values())
    tot_c = sum(s["n_correct"] for s in by_sub.values())
    n_responses = len(data) * 4 or 1
    mean_n2 = total_n2 / n_responses

    return {
        "accuracy": tot_c / tot_n if tot_n else 0.0,
        "mean_n2": mean_n2,
        "n": tot_n,
    }


def optimise_var_informed(
    train: list[dict], test: list[dict], n_max: int = 8, grid_steps: int = 15,
) -> dict:
    all_stds = _compute_mini_stds(train)
    flat = [s for stds in all_stds for s in stds]
    pct = [float(np.percentile(flat, p)) for p in np.linspace(0, 95, grid_steps)]
    fixed = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    sigma_vals = sorted(set(pct + fixed))

    best, best_constrained = None, None
    grid = []
    for s1 in sigma_vals:
        for s2 in sigma_vals:
            if s2 <= s1:
                continue
            r = compute_var_informed_accuracy(train, s1, s2, n_max)
            grid.append({"sigma1": s1, "sigma2": s2, **r})
            if best is None or r["accuracy"] > best["accuracy"]:
                best = {"sigma1": s1, "sigma2": s2, **r}
            if r["mean_n2"] <= 2.0 and (best_constrained is None or r["accuracy"] > best_constrained["accuracy"]):
                best_constrained = {"sigma1": s1, "sigma2": s2, **r}

    # Evaluate on test
    test_best = test_constrained = None
    if best:
        test_best = compute_var_informed_accuracy(test, best["sigma1"], best["sigma2"], n_max)
    if best_constrained:
        test_constrained = compute_var_informed_accuracy(
            test, best_constrained["sigma1"], best_constrained["sigma2"], n_max)

    return {
        "train_best": best,
        "train_best_constrained": best_constrained,
        "test_best": test_best,
        "test_best_constrained": test_constrained,
        "grid_results": grid,
    }


# ===================================================================
# Mini-full convergence
# ===================================================================

def compute_ensemble_convergence(
    data: list[dict], cheap_model: str = "mini", reference_model: str = "full",
) -> dict:
    score_key = f"{cheap_model}_scores"
    ref_key = f"{reference_model}_scores"
    # Filter to examples that have both model scores
    data = [r for r in data if score_key in r and ref_key in r]
    if not data:
        return {"by_k": []}

    max_k = min(len(r[score_key][0]) for r in data)
    ref_winners = {}
    for r in data:
        means = [_mean_ignoring_none(s) for s in r[ref_key]]
        if any(m is None for m in means):
            continue
        max_s = max(means)
        w = [i for i, m in enumerate(means) if m == max_s]
        ref_winners[r["id"]] = {"winner": w[0] if len(w) == 1 else -1, "means": means}

    by_k = []
    for k in range(1, max_k + 1):
        agreements, rank_corrs = [], []
        for r in data:
            if r["id"] not in ref_winners:
                continue
            cheap_means = [_mean_ignoring_none(s[:k]) for s in r[score_key]]
            if any(m is None for m in cheap_means):
                continue
            max_m = max(cheap_means)
            mw = [i for i, m in enumerate(cheap_means) if m == max_m]
            cheap_winner = mw[0] if len(mw) == 1 else -1
            fw = ref_winners[r["id"]]
            agreements.append(1 if cheap_winner == fw["winner"] and cheap_winner != -1 else 0)
            if len(cheap_means) >= 3:
                rho, _ = stats.spearmanr(cheap_means, fw["means"])
                if not np.isnan(rho):
                    rank_corrs.append(float(rho))

        by_k.append({
            "k": k,
            "agreement": float(np.mean(agreements)) if agreements else 0.0,
            "rank_correlation": float(np.mean(rank_corrs)) if rank_corrs else None,
        })
    return {"by_k": by_k}


# ===================================================================
# Paired bootstrap (per-example correctness vectors)
# ===================================================================

def _per_example_correctness(data, model: str, k_subset: int | None = None) -> dict[str, int]:
    """Return {id: 1/0} for correctness on each example. Ties count as 0."""
    key = f"{model}_scores"
    out: dict[str, int] = {}
    for r in data:
        scores_per_resp = r.get(key)
        if scores_per_resp is None:
            continue
        means = []
        for scores in scores_per_resp:
            s = scores[:k_subset] if k_subset else scores
            means.append(_mean_ignoring_none(s))
        if any(m is None for m in means):
            continue
        max_s = max(means)
        winners = [i for i, m in enumerate(means) if m == max_s]
        out[r["id"]] = 1 if (len(winners) == 1 and winners[0] == 0) else 0
    return out


def paired_bootstrap(
    data_a, data_b, model_a: str, model_b: str,
    k_a: int | None = None, k_b: int | None = None,
    n_bootstrap: int = 2000, alpha: float = 0.05, seed: int = 42,
) -> dict:
    """Paired bootstrap on the intersection of examples.

    Returns:
        {n, mean_delta, ci_low, ci_high, p_a_gt_b, acc_a, acc_b}
    """
    corr_a = _per_example_correctness(data_a, model_a, k_a)
    corr_b = _per_example_correctness(data_b, model_b, k_b)
    shared = sorted(set(corr_a) & set(corr_b))
    if not shared:
        return {"n": 0}
    a = np.array([corr_a[i] for i in shared], dtype=float)
    b = np.array([corr_b[i] for i in shared], dtype=float)
    n = len(shared)
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    return {
        "n": n,
        "acc_a": float(a.mean()),
        "acc_b": float(b.mean()),
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, alpha / 2 * 100)),
        "ci_high": float(np.percentile(deltas, (1 - alpha / 2) * 100)),
        "p_a_gt_b": float((deltas > 0).mean()),
    }


# ===================================================================
# Per-response variance distribution + KS test
# ===================================================================

def _per_example_variance(data, model: str, k_subset: int | None = None) -> list[float]:
    """Mean across 4 responses of per-response score std."""
    key = f"{model}_scores"
    out = []
    for r in data:
        scores_per_resp = r.get(key)
        if scores_per_resp is None:
            continue
        stds = []
        for scores in scores_per_resp:
            s = scores[:k_subset] if k_subset else scores
            valid = [v for v in s if v is not None]
            stds.append(float(np.std(valid)) if len(valid) > 1 else 0.0)
        if stds:
            out.append(float(np.mean(stds)))
    return out


def per_response_variance_by_condition(
    conditions: list[tuple[str, list[dict], str, int | None]],
) -> dict:
    """Compute σ_i distributions per condition and pairwise KS tests.

    Args:
        conditions: list of (name, data, model, k_subset).

    Returns:
        {per_condition: {name: {mean, median, std, n}}, ks_tests: {a_vs_b: {...}}}
    """
    per_cond = {}
    arrs: dict[str, list[float]] = {}
    for name, data, model, k in conditions:
        arr = _per_example_variance(data, model, k)
        arrs[name] = arr
        per_cond[name] = {
            "mean": float(np.mean(arr)) if arr else 0.0,
            "median": float(np.median(arr)) if arr else 0.0,
            "std": float(np.std(arr)) if arr else 0.0,
            "n": len(arr),
        }

    ks = {}
    names = list(arrs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if arrs[a] and arrs[b]:
                stat, p = stats.ks_2samp(arrs[a], arrs[b])
                ks[f"{a}_vs_{b}"] = {
                    "ks_statistic": float(stat),
                    "p_value": float(p),
                    "mean_delta": per_cond[b]["mean"] - per_cond[a]["mean"],
                }
    return {"per_condition": per_cond, "ks_tests": ks}


# ===================================================================
# Variance as a classifier for incorrectness (ROC-AUC)
# ===================================================================

def variance_as_classifier_auc(data, model: str = "full", k_subset: int | None = None) -> dict:
    """Treat per-example variance as a score for 'is incorrect'.

    Returns:
        {auc, n_incorrect, n_correct, precision_at_recall_20, precision_at_recall_50}
    """
    key = f"{model}_scores"
    variances = []
    incorrect = []
    for r in data:
        scores_per_resp = r.get(key)
        if scores_per_resp is None:
            continue
        stds = []
        means = []
        for scores in scores_per_resp:
            s = scores[:k_subset] if k_subset else scores
            valid = [v for v in s if v is not None]
            stds.append(float(np.std(valid)) if len(valid) > 1 else 0.0)
            means.append(_mean_ignoring_none(s))
        if any(m is None for m in means):
            continue
        max_s = max(means)
        winners = [i for i, m in enumerate(means) if m == max_s]
        is_correct = (len(winners) == 1 and winners[0] == 0)
        variances.append(float(np.mean(stds)))
        incorrect.append(0 if is_correct else 1)

    if not variances or sum(incorrect) == 0 or sum(incorrect) == len(incorrect):
        return {"auc": None, "n": len(variances)}

    var_arr = np.array(variances)
    lbl = np.array(incorrect)

    # ROC-AUC: fraction of (incorrect, correct) pairs ranked correctly.
    order = np.argsort(-var_arr)
    sorted_lbl = lbl[order]
    n_pos = lbl.sum()
    n_neg = len(lbl) - n_pos
    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    precisions = []
    recalls = []
    for y in sorted_lbl:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    auc = float(np.trapezoid(tpr, fpr))

    def _precision_at_recall(target: float) -> float | None:
        for p, r in zip(precisions, recalls):
            if r >= target:
                return float(p)
        return None

    return {
        "auc": auc,
        "n": len(variances),
        "n_incorrect": int(n_pos),
        "n_correct": int(n_neg),
        "precision_at_recall_20": _precision_at_recall(0.20),
        "precision_at_recall_50": _precision_at_recall(0.50),
        "base_rate_incorrect": float(n_pos / len(lbl)),
        "roc_fpr": [float(x) for x in fpr],
        "roc_tpr": [float(x) for x in tpr],
    }


# ===================================================================
# Tie rate by condition (surface existing counts cleanly)
# ===================================================================

def tie_rate_by_condition(
    conditions: list[tuple[str, list[dict], str, int | None]],
) -> dict:
    """Surface tie counts for each condition using compute_accuracy."""
    out = {}
    for name, data, model, k in conditions:
        acc = compute_accuracy(data, model=model, k_subset=k)
        n = acc["n"]
        out[name] = {
            "n": n,
            "n_tied": acc["n_tied"],
            "tie_rate": acc["n_tied"] / n if n else 0.0,
        }
    return out


# ===================================================================
# Main: derive all conditions from collections
# ===================================================================


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        print("ERROR: results/raw/ not found")
        sys.exit(1)

    all_metrics = {}

    # --- Load available collections ---
    collections: dict[str, list[dict]] = {}
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        key = f.stem
        data = load_collection(f)
        if data:
            collections[key] = data
            print(f"Loaded {key}: {len(data)} examples")

    if not collections:
        print("No collection files found")
        sys.exit(1)

    # --- Find collections ---
    def _find(prefix: str) -> tuple[str, list[dict]] | None:
        for key, data in collections.items():
            if key.startswith(prefix):
                return key, data
        return None

    base = _find("base_both") or _find("base_all")
    criteria = _find("criteria_both") or _find("criteria_all")
    cal_low = _find("cal-low_both") or _find("cal-low_all")
    cal_high = _find("cal-high_both") or _find("cal-high_all")
    cal_both = _find("cal-both_both") or _find("cal-both_all")
    cal_cross = _find("cal-cross_both") or _find("cal-cross_all")
    combined = _find("combined_both") or _find("combined_all")

    # Merge standalone nano collection into base if present
    base_nano = _find("base_nano")
    if base and base_nano and "nano_scores" not in base[1][0]:
        nano_by_id = {r["id"]: r for r in base_nano[1]}
        merged = []
        n_with_nano = 0
        for r in base[1]:
            nr = nano_by_id.get(r["id"])
            if nr:
                r = {**r}
                r["nano_scores"] = nr["nano_scores"]
                r["nano_errors"] = nr["nano_errors"]
                r["cost"] = {**r.get("cost", {})}
                for tk in ("nano_input_tokens", "nano_output_tokens"):
                    if tk in nr.get("cost", {}):
                        r["cost"][tk] = nr["cost"][tk]
                n_with_nano += 1
            merged.append(r)
        print(f"Merged nano scores into base: {n_with_nano} / {len(base[1])} examples")
        base = (base[0], merged)

    # Temperature sweep files (new format only)
    temp_files = {}
    for f in sorted(RAW_DIR.glob("base_full_k8_t*.jsonl")):
        temp_files[f.stem] = load_collection(f)

    # --- Intersection of IDs across main collections ---
    main_collections = {}
    for name, found in [("base", base), ("criteria", criteria), ("cal_low", cal_low),
                        ("combined", combined)]:
        if found:
            main_collections[name] = found[1]
    shared_ids = intersect_collections(main_collections) if main_collections else set()
    print(f"\nShared IDs across main collections: {len(shared_ids)}")

    # --- Helper: compute condition metrics ---
    def _condition_metrics(
        name: str, data: list[dict], model: str = "full", k: int | None = None,
        use_intersection: bool = False,
        deploy_models: str = "full", deploy_k: int = 1,
    ) -> dict:
        if use_intersection and shared_ids:
            data = [r for r in data if r["id"] in shared_ids]
        acc = compute_accuracy(data, model=model, k_subset=k)
        ci = bootstrap_accuracy_ci(data, model=model, k_subset=k)
        cost_per_ex = compute_deployment_cost(data, models=deploy_models, k=deploy_k)
        return {
            "name": name, "n": acc["n"],
            "accuracy": acc, "accuracy_ci": ci,
            "cost": {"cost_per_example": cost_per_ex},
        }

    # === Derive conditions ===

    print(f"\n{'='*60}")
    print("DERIVED CONDITIONS")
    print(f"{'='*60}")

    # 1. From base collection
    if base:
        key, data = base
        m = _condition_metrics("Baseline (full k=1)", data, "full", k=1,
                              deploy_models="full", deploy_k=1)
        all_metrics["baseline"] = m
        print(f"\n  Baseline: {m['accuracy']['overall']:.3f} "
              f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, {m['accuracy_ci']['overall']['ci_high']:.3f}]"
              f"  n={m['n']}  ${m['cost']['cost_per_example']:.4f}/ex")

        m = _condition_metrics("Ensemble full k=8", data, "full", k=8,
                              deploy_models="full", deploy_k=8)
        all_metrics["ensemble_k8"] = m
        print(f"  Ensemble k=8: {m['accuracy']['overall']:.3f} "
              f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, {m['accuracy_ci']['overall']['ci_high']:.3f}]"
              f"  ${m['cost']['cost_per_example']:.4f}/ex")

        m = _condition_metrics("Mini k=8", data, "mini", k=8,
                              deploy_models="mini", deploy_k=8)
        all_metrics["mini_k8"] = m
        print(f"  Mini k=8: {m['accuracy']['overall']:.3f} "
              f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, {m['accuracy_ci']['overall']['ci_high']:.3f}]"
              f"  ${m['cost']['cost_per_example']:.4f}/ex")

        m = _condition_metrics("Mini k=1", data, "mini", k=1,
                              deploy_models="mini", deploy_k=1)
        all_metrics["mini_k1"] = m
        print(f"  Mini k=1: {m['accuracy']['overall']:.3f}")

        # Nano conditions (if nano scores are present)
        if any("nano_scores" in r for r in data):
            m = _condition_metrics("Nano k=8", data, "nano", k=8,
                                  deploy_models="nano", deploy_k=8)
            all_metrics["nano_k8"] = m
            print(f"  Nano k=8: {m['accuracy']['overall']:.3f} "
                  f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, {m['accuracy_ci']['overall']['ci_high']:.3f}]"
                  f"  ${m['cost']['cost_per_example']:.4f}/ex")

            m = _condition_metrics("Nano k=1", data, "nano", k=1,
                                  deploy_models="nano", deploy_k=1)
            all_metrics["nano_k1"] = m
            print(f"  Nano k=1: {m['accuracy']['overall']:.3f}")

        # Diminishing returns
        dim = {}
        for k_sub in range(1, 9):
            a = compute_accuracy(data, model="full", k_subset=k_sub)
            dim[k_sub] = a["overall"]
        all_metrics["diminishing_returns_full"] = dim
        dim_mini = {}
        for k_sub in range(1, 9):
            a = compute_accuracy(data, model="mini", k_subset=k_sub)
            dim_mini[k_sub] = a["overall"]
        all_metrics["diminishing_returns_mini"] = dim_mini
        if any("nano_scores" in r for r in data):
            dim_nano = {}
            for k_sub in range(1, 9):
                a = compute_accuracy(data, model="nano", k_subset=k_sub)
                dim_nano[k_sub] = a["overall"]
            all_metrics["diminishing_returns_nano"] = dim_nano
        print(f"  Diminishing returns (full): {' '.join(f'k={k}:{v:.3f}' for k, v in dim.items())}")
        if "diminishing_returns_nano" in all_metrics:
            print(f"  Diminishing returns (nano): {' '.join(f'k={k}:{v:.3f}' for k, v in all_metrics['diminishing_returns_nano'].items())}")

        # Variance metrics
        var = compute_variance_metrics(data, model="full")
        all_metrics["variance_full"] = var
        print(f"  Variance-correctness corr: {var['variance_correctness_correlation']}")

        # Escalation (train/test split for optimised params)
        if "mini_scores" in data[0]:
            train, test = train_test_split(data)
            print(f"\n  Escalation (train={len(train)}, test={len(test)}):")

            esc = compute_escalation_metrics(data)
            all_metrics["escalation_hard"] = esc

            blend = optimise_blend(train, test)
            all_metrics["blend_optimised"] = blend
            print(f"    Soft blend: train={blend['train_accuracy']:.3f} test={blend['test_accuracy']:.3f} "
                  f"midpoint={blend['best_midpoint']:.3f}")

            vie = optimise_var_informed(train, test)
            all_metrics["var_informed_optimised"] = vie
            if vie["test_best"]:
                print(f"    Var-informed best: test={vie['test_best']['accuracy']:.3f}")
            if vie["test_best_constrained"]:
                print(f"    Var-informed (<=2 calls): test={vie['test_best_constrained']['accuracy']:.3f}")

            conv = compute_ensemble_convergence(data, cheap_model="mini")
            all_metrics["convergence"] = conv

            if any("nano_scores" in r for r in data):
                conv_nano = compute_ensemble_convergence(data, cheap_model="nano")
                all_metrics["convergence_nano"] = conv_nano

    # 2. From criteria collection
    if criteria:
        key, data = criteria
        m = _condition_metrics("Criteria (full k=1)", data, "full", k=1,
                              deploy_models="full", deploy_k=1)
        all_metrics["criteria"] = m
        print(f"\n  Criteria k=1: {m['accuracy']['overall']:.3f} "
              f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, {m['accuracy_ci']['overall']['ci_high']:.3f}]"
              f"  ${m['cost']['cost_per_example']:.4f}/ex")

        m = _condition_metrics("Criteria (full k=8)", data, "full", k=8,
                              deploy_models="full", deploy_k=8)
        all_metrics["criteria_k8"] = m
        print(f"  Criteria k=8 (full): {m['accuracy']['overall']:.3f}"
              f"  ${m['cost']['cost_per_example']:.4f}/ex")

        m = _condition_metrics("Criteria (mini k=8)", data, "mini", k=8,
                              deploy_models="mini", deploy_k=8)
        all_metrics["criteria_mini_k8"] = m
        print(f"  Criteria k=8 (mini): {m['accuracy']['overall']:.3f}"
              f"  ${m['cost']['cost_per_example']:.4f}/ex")

    # 3. From calibration collections
    for cal_name, cal_found in [("cal_high", cal_high), ("cal_low", cal_low),
                                 ("cal_both", cal_both), ("cal_cross", cal_cross)]:
        if cal_found:
            key, data = cal_found
            label = cal_name.replace("cal_", "Calibration (") + ")"
            # Cal cost includes the calibration scoring call (~1.5x baseline)
            m = _condition_metrics(label, data, "full", k=1,
                                  deploy_models="full", deploy_k=1)
            all_metrics[cal_name] = m
            print(f"\n  {label} k=1: {m['accuracy']['overall']:.3f} "
                  f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, {m['accuracy_ci']['overall']['ci_high']:.3f}]"
                  f"  ${m['cost']['cost_per_example']:.4f}/ex")

            m = _condition_metrics(label + " k=8", data, "full", k=8,
                                  deploy_models="full", deploy_k=8)
            all_metrics[cal_name + "_k8"] = m
            print(f"  {label} k=8: {m['accuracy']['overall']:.3f}"
                  f"  ${m['cost']['cost_per_example']:.4f}/ex")

    # 4. From combined collection
    if combined:
        key, data = combined
        m = _condition_metrics("Combined (full k=8)", data, "full", k=8,
                              deploy_models="both", deploy_k=8)
        all_metrics["combined"] = m
        print(f"\n  Combined (full k=8): {m['accuracy']['overall']:.3f} "
              f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, {m['accuracy_ci']['overall']['ci_high']:.3f}]"
              f"  ${m['cost']['cost_per_example']:.4f}/ex")

        m = _condition_metrics("Combined (full k=1)", data, "full", k=1,
                              deploy_models="both", deploy_k=1)
        all_metrics["combined_k1"] = m
        print(f"  Combined (full k=1): {m['accuracy']['overall']:.3f}")

        if "mini_scores" in data[0]:
            train, test = train_test_split(data)
            blend = optimise_blend(train, test)
            all_metrics["combined_blend"] = blend
            print(f"  Combined + blend: train={blend['train_accuracy']:.3f} "
                  f"test={blend['test_accuracy']:.3f}")

    # 5. Temperature sweep
    if temp_files:
        print(f"\n  Temperature sweep:")
        sweep = {}
        for key, data in sorted(temp_files.items()):
            for k_sub in [1, 8]:
                acc = compute_accuracy(data, model="full", k_subset=k_sub)
                ci = bootstrap_accuracy_ci(data, model="full", k_subset=k_sub)
                sweep[f"{key}_k{k_sub}"] = {
                    "accuracy": acc["overall"],
                    "ci_low": ci["overall"]["ci_low"],
                    "ci_high": ci["overall"]["ci_high"],
                    "n": acc["n"],
                }
                print(f"    {key} k={k_sub}: {acc['overall']:.3f} "
                      f"[{ci['overall']['ci_low']:.3f}, {ci['overall']['ci_high']:.3f}]")
        # Add the base temp=1.0 data for completeness
        if base:
            for k_sub in [1, 8]:
                acc = compute_accuracy(base[1], model="full", k_subset=k_sub)
                ci = bootstrap_accuracy_ci(base[1], model="full", k_subset=k_sub)
                sweep[f"base_full_k8_t1.0_k{k_sub}"] = {
                    "accuracy": acc["overall"],
                    "ci_low": ci["overall"]["ci_low"],
                    "ci_high": ci["overall"]["ci_high"],
                    "n": acc["n"],
                }
        all_metrics["temperature_sweep"] = sweep

    # 6. Intersection analysis
    if shared_ids and base and criteria:
        print(f"\n  Intersection analysis (n={len(shared_ids)}):")
        for name, found in [("baseline", base), ("criteria", criteria),
                            ("cal_low", cal_low), ("combined", combined)]:
            if found:
                _, data = found
                m = _condition_metrics(name, data, "full", k=1, use_intersection=True)
                print(f"    {name}: {m['accuracy']['overall']:.3f} (n={m['n']})")
                all_metrics[f"intersection_{name}"] = m

        # k=8 intersection variants for headline conditions
        extra_intersection = []
        if base:
            extra_intersection += [
                ("ensemble_k8", base[1], "full", 8),
                ("mini_k8", base[1], "mini", 8),
            ]
        if criteria:
            extra_intersection += [
                ("criteria_k8", criteria[1], "full", 8),
                ("criteria_mini_k8", criteria[1], "mini", 8),
            ]
        if cal_low:
            extra_intersection += [("cal_low_k8", cal_low[1], "full", 8)]
        if combined:
            extra_intersection += [("combined_k8", combined[1], "full", 8)]
        for name, data, model, k in extra_intersection:
            m = _condition_metrics(name, data, model=model, k=k, use_intersection=True)
            print(f"    {name}: {m['accuracy']['overall']:.3f} (n={m['n']})")
            all_metrics[f"intersection_{name}"] = m

    # === 7. Reviewer-mandated analyses ===
    print(f"\n{'='*60}")
    print("REVIEWER-MANDATED ANALYSES")
    print(f"{'='*60}")

    # Paired bootstrap — the three claims from the response plan
    pb_results = {}
    if base and criteria:
        pb_results["criteria_k1_vs_base_k1"] = paired_bootstrap(
            criteria[1], base[1], "full", "full", k_a=1, k_b=1)
        pb_results["criteria_k8_vs_ensemble_k8"] = paired_bootstrap(
            criteria[1], base[1], "full", "full", k_a=8, k_b=8)
    if combined and base:
        pb_results["combined_k8_vs_ensemble_k8"] = paired_bootstrap(
            combined[1], base[1], "full", "full", k_a=8, k_b=8)
    if combined and criteria:
        pb_results["criteria_k8_vs_combined_k8"] = paired_bootstrap(
            criteria[1], combined[1], "full", "full", k_a=8, k_b=8)
    all_metrics["paired_bootstrap"] = pb_results
    print("\n  Paired bootstrap (intersection, n_resamples=2000):")
    for name, r in pb_results.items():
        if r.get("n"):
            print(f"    {name}: Δ={r['mean_delta']*100:+.2f}pp "
                  f"[{r['ci_low']*100:+.2f}, {r['ci_high']*100:+.2f}] "
                  f"p(a>b)={r['p_a_gt_b']:.3f} n={r['n']}")

    # Per-response variance by condition
    var_conditions = []
    if base:
        var_conditions.append(("baseline_k8", base[1], "full", 8))
    if criteria:
        var_conditions.append(("criteria_k8", criteria[1], "full", 8))
    if combined:
        var_conditions.append(("combined_k8", combined[1], "full", 8))
    if var_conditions:
        vr = per_response_variance_by_condition(var_conditions)
        all_metrics["per_response_variance"] = vr
        print("\n  Per-response variance by condition (mean σ_i):")
        for name, stats_d in vr["per_condition"].items():
            print(f"    {name}: mean={stats_d['mean']:.4f} "
                  f"median={stats_d['median']:.4f} n={stats_d['n']}")
        for name, ks in vr["ks_tests"].items():
            print(f"    KS {name}: D={ks['ks_statistic']:.3f} p={ks['p_value']:.4f}")

    # Variance as classifier AUC
    auc_results = {}
    if base:
        auc_results["baseline_k8"] = variance_as_classifier_auc(base[1], "full", 8)
    if criteria:
        auc_results["criteria_k8"] = variance_as_classifier_auc(criteria[1], "full", 8)
    all_metrics["variance_auc"] = auc_results
    print("\n  Variance as incorrectness classifier (ROC-AUC):")
    for name, r in auc_results.items():
        if r.get("auc") is not None:
            print(f"    {name}: AUC={r['auc']:.3f} "
                  f"P@R20={r['precision_at_recall_20']:.3f} "
                  f"P@R50={r['precision_at_recall_50']:.3f} "
                  f"base_rate={r['base_rate_incorrect']:.3f} n={r['n']}")

    # Tie rate by condition
    tie_conditions = []
    if base:
        tie_conditions += [
            ("baseline_k1", base[1], "full", 1),
            ("baseline_k8", base[1], "full", 8),
        ]
    if criteria:
        tie_conditions += [
            ("criteria_k1", criteria[1], "full", 1),
            ("criteria_k8", criteria[1], "full", 8),
        ]
    if combined:
        tie_conditions.append(("combined_k8", combined[1], "full", 8))
    if tie_conditions:
        tr = tie_rate_by_condition(tie_conditions)
        all_metrics["tie_rate"] = tr
        print("\n  Tie rates:")
        for name, t in tr.items():
            print(f"    {name}: {t['n_tied']}/{t['n']} = {t['tie_rate']:.3f}")

    # === 8. Claude generalisability experiments ===
    claude_base = _find("base_claude_both")
    claude_criteria = _find("criteria_claude_both")

    if claude_base or claude_criteria:
        print(f"\n{'='*60}")
        print("CLAUDE CONDITIONS")
        print(f"{'='*60}")

    def _claude_condition_metrics(name, data, model="full", k=None):
        """Like _condition_metrics but with $0 cost (Pro Max subscription)."""
        acc = compute_accuracy(data, model=model, k_subset=k)
        ci = bootstrap_accuracy_ci(data, model=model, k_subset=k)
        return {
            "name": name, "n": acc["n"],
            "accuracy": acc, "accuracy_ci": ci,
            "cost": {"cost_per_example": 0.0},
        }

    if claude_base:
        key, data = claude_base
        for model_label, model_key in [("Sonnet 4.6", "full"), ("Haiku 4.5", "mini")]:
            m = _claude_condition_metrics(
                f"Claude {model_label} Baseline k=1", data, model_key, k=1)
            all_metrics[f"claude_{model_key}_baseline"] = m
            print(f"\n  Claude {model_label} Baseline k=1: {m['accuracy']['overall']:.3f} "
                  f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, "
                  f"{m['accuracy_ci']['overall']['ci_high']:.3f}]  n={m['n']}")

            m = _claude_condition_metrics(
                f"Claude {model_label} Ensemble k=8", data, model_key, k=8)
            all_metrics[f"claude_{model_key}_ensemble_k8"] = m
            print(f"  Claude {model_label} Ensemble k=8: {m['accuracy']['overall']:.3f} "
                  f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, "
                  f"{m['accuracy_ci']['overall']['ci_high']:.3f}]")

            # Diminishing returns
            dim = {}
            for k_sub in range(1, 9):
                a = compute_accuracy(data, model=model_key, k_subset=k_sub)
                dim[k_sub] = a["overall"]
            all_metrics[f"claude_diminishing_returns_{model_key}"] = dim
            print(f"  Claude {model_label} diminishing returns: "
                  f"{' '.join(f'k={k}:{v:.3f}' for k, v in dim.items())}")

    if claude_criteria:
        key, data = claude_criteria
        for model_label, model_key in [("Sonnet 4.6", "full"), ("Haiku 4.5", "mini")]:
            m = _claude_condition_metrics(
                f"Claude {model_label} Criteria k=1", data, model_key, k=1)
            all_metrics[f"claude_{model_key}_criteria"] = m
            print(f"\n  Claude {model_label} Criteria k=1: {m['accuracy']['overall']:.3f} "
                  f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, "
                  f"{m['accuracy_ci']['overall']['ci_high']:.3f}]  n={m['n']}")

            m = _claude_condition_metrics(
                f"Claude {model_label} Criteria k=8", data, model_key, k=8)
            all_metrics[f"claude_{model_key}_criteria_k8"] = m
            print(f"  Claude {model_label} Criteria k=8: {m['accuracy']['overall']:.3f} "
                  f"[{m['accuracy_ci']['overall']['ci_low']:.3f}, "
                  f"{m['accuracy_ci']['overall']['ci_high']:.3f}]")

    # --- Save ---
    output = TABLES_DIR / "all_metrics.json"
    with open(output, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n\nSaved to {output}")


if __name__ == "__main__":
    main()

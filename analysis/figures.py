"""Generate figures for RB2 experiment results.

Reads from results/tables/all_metrics.json and results/raw/*.jsonl.
Saves figures to figures/ as PNG (300 DPI).

Run as: python analysis/figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.compute_metrics import (
    load_collection,
    compute_accuracy,
    compute_blend_accuracy,
    _compute_mini_stds,
    _mean_ignoring_none,
)

SUBSET_ORDER = ["Factuality", "Focus", "Math", "Precise IF", "Safety"]
FIGURES_DIR = Path("figures")
RAW_DIR = Path("results/raw")
TABLES_DIR = Path("results/tables")

COLORS = {
    "Baseline": "#0173b2",
    "Criteria": "#029e73",
    "Criteria (k=1)": "#029e73",
    "Calibration (low)": "#cc78bc",
    "Calibration (high)": "#fbafe4",
    "Calibration (both)": "#ca9161",
    "Calibration (cross)": "#ece133",
    "Ensemble k=8": "#de8f05",
    "Mini k=8": "#ca9161",
    "Nano k=8": "#56b4e9",
    "Soft blend": "#fbafe4",
    "Combined": "#949494",
    "Criteria k=8": "#d55e00",
    "Combined + blend": "#ece133",
}


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"  Saved figures/{name}.png")
    plt.close(fig)


def _load_metrics():
    p = TABLES_DIR / "all_metrics.json"
    if not p.exists():
        print("ERROR: all_metrics.json not found. Run compute_metrics.py first.")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def _find_collection(prefix: str) -> list[dict] | None:
    # Try _all first (includes nano), then fall back to any match
    for suffix in ["_all", "_both", ""]:
        for f in sorted(RAW_DIR.glob(f"{prefix}{suffix}*.jsonl")):
            data = load_collection(f)
            if data:
                return _merge_nano(data, prefix)
    return None


def _merge_nano(data: list[dict], prefix: str) -> list[dict]:
    """Merge standalone nano collection into data if present and needed."""
    if data and any("nano_scores" in r for r in data):
        return data
    # Strip model suffix to get the base prefix (e.g., "base_both" -> "base")
    base_prefix = prefix.split("_")[0] if "_" in prefix else prefix
    nano_files = sorted(RAW_DIR.glob(f"{base_prefix}_nano*.jsonl"))
    if not nano_files:
        return data
    nano_data = load_collection(nano_files[0])
    if not nano_data:
        return data
    nano_by_id = {r["id"]: r for r in nano_data}
    merged = []
    for r in data:
        nr = nano_by_id.get(r["id"])
        if nr:
            r = {**r}
            r["nano_scores"] = nr["nano_scores"]
            r["nano_errors"] = nr["nano_errors"]
            r["cost"] = {**r.get("cost", {})}
            for tk in ("nano_input_tokens", "nano_output_tokens"):
                if tk in nr.get("cost", {}):
                    r["cost"][tk] = nr["cost"][tk]
        merged.append(r)
    return merged


# ===================================================================
# Figure 1: Hero accuracy by condition and category
# ===================================================================

def plot_hero_accuracy(metrics):
    """Per-category accuracy by condition. Best-of-class per (condition, category).

    For each condition we have multiple candidate providers; we pick the
    provider with the highest *overall* accuracy and show its per-category
    breakdown. Mixed-provider lines are kept honest by noting the
    selected provider in the legend label.
    """
    from analysis.compute_metrics import load_collection, compute_accuracy

    # (display_label, candidates [(provider, path, model, k)])
    spec = [
        ("Baseline",          [("GPT-5.4",    "results/raw/base_both_k8.jsonl",        "full", 1),
                                ("Sonnet 4.6", "results/raw/base_claude_both_k8.jsonl", "full", 1)]),
        ("Criteria (k=1)",    [("GPT-5.4",    "results/raw/criteria_both_k8.jsonl",    "full", 1),
                                ("Sonnet 4.6", "results/raw/criteria_claude_both_k8.jsonl", "full", 1)]),
        ("Calibration (low)", [("GPT-5.4",    "results/raw/cal-low_both_k8.jsonl",     "full", 8)]),
        ("Ensemble k=8",      [("GPT-5.4",    "results/raw/base_both_k8.jsonl",        "full", 8),
                                ("Sonnet 4.6", "results/raw/base_claude_both_k8.jsonl", "full", 8)]),
        ("Mini k=8",          [("GPT mini",   "results/raw/base_both_k8.jsonl",        "mini", 8),
                                ("Haiku 4.5",  "results/raw/base_claude_both_k8.jsonl", "mini", 8)]),
        ("Nano k=8",          [("GPT nano",   "results/raw/base_nano_k8.jsonl",        "nano", 8)]),
        ("Criteria k=8",      [("GPT-5.4",    "results/raw/criteria_both_k8.jsonl",    "full", 8),
                                ("Sonnet 4.6", "results/raw/criteria_claude_both_k8.jsonl", "full", 8)]),
        ("Combined",          [("GPT-5.4",    "results/raw/combined_both_k8.jsonl",    "full", 8)]),
    ]

    conditions = {}
    for label, candidates in spec:
        best_acc, best_pkg, best_provider = -1, None, None
        for provider, path, model, k in candidates:
            try:
                a = compute_accuracy(load_collection(path), model=model, k_subset=k)
            except FileNotFoundError:
                continue
            if a["overall"] > best_acc:
                best_acc, best_pkg, best_provider = a["overall"], a, provider
        if best_pkg is None:
            continue
        full_label = f"{label} ({best_provider})"
        conditions[full_label] = {
            sub: d["accuracy"]
            for sub, d in best_pkg["by_subset"].items()
        }
        # Keep base-label color mapping
        if label not in COLORS and full_label not in COLORS:
            pass  # fallback handled below

    if not conditions:
        print("  Skipping hero figure: no data")
        return

    subsets = [s for s in SUBSET_ORDER if any(s in c for c in conditions.values())]
    cond_names = list(conditions.keys())
    n_cond = len(cond_names)

    fig, ax = plt.subplots(figsize=(max(10, len(subsets) * 2) + 3, 6))
    bar_w = 0.8 / n_cond
    x = np.arange(len(subsets))

    for i, name in enumerate(cond_names):
        vals = [conditions[name].get(s, 0) for s in subsets]
        # The provider tag is appended at the END (last " ("), so strip it for color lookup.
        last_open = name.rfind(" (")
        base_label = name[:last_open] if last_open != -1 else name
        color = COLORS.get(base_label, COLORS.get(name, f"C{i}"))
        ax.bar(x + i * bar_w, vals, bar_w, label=name, color=color)

    ax.axhline(0.25, color="gray", linestyle="--", lw=0.8, label="Random (25%)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Condition and Category")
    ax.set_xticks(x + bar_w * (n_cond - 1) / 2)
    ax.set_xticklabels(subsets)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper left", fontsize=13, bbox_to_anchor=(1.01, 1.0))
    _save(fig, "hero_accuracy")


# ===================================================================
# Figure 2: Pareto frontier
# ===================================================================

def plot_pareto_frontier(metrics):
    """Cost-accuracy Pareto frontier, best-of-class per condition.

    Each point is (class, condition); accuracy is max across providers
    that have data for that (class, condition); cost is the class cost
    derived from the GPT pricing (Anthropic costs unspecified).
    """
    from analysis.compute_metrics import load_collection, compute_accuracy

    def _acc(path, model, k):
        try:
            return compute_accuracy(load_collection(path), model=model, k_subset=k)["overall"]
        except FileNotFoundError:
            return None

    # Per-condition: try each candidate (path, model, k); keep the best accuracy
    # and remember which provider supplied it (provider used only for color/caption).
    # Placement: dx_pts and dy_pts are pixel offsets in the figure; ha is the
    # text horizontal alignment relative to the anchor point.
    spec = [
        # (label, gpt_cost_per_ex_key, candidates, dx_pts, dy_pts, ha)
        ("Baseline",            "baseline",         [("GPT-5.4",     "results/raw/base_both_k8.jsonl", "full", 1),
                                                     ("Sonnet 4.6",  "results/raw/base_claude_both_k8.jsonl", "full", 1)],
                                                    7, -8, "left"),
        ("Criteria (k=1)",      "criteria",         [("GPT-5.4",     "results/raw/criteria_both_k8.jsonl", "full", 1),
                                                     ("Sonnet 4.6",  "results/raw/criteria_claude_both_k8.jsonl", "full", 1)],
                                                    7, 0,  "left"),
        ("Ensemble k=8",        "ensemble_k8",      [("GPT-5.4",     "results/raw/base_both_k8.jsonl", "full", 8),
                                                     ("Sonnet 4.6",  "results/raw/base_claude_both_k8.jsonl", "full", 8)],
                                                    -7, -8, "right"),
        ("Criteria k=8",        "criteria_k8",      [("GPT-5.4",     "results/raw/criteria_both_k8.jsonl", "full", 8),
                                                     ("Sonnet 4.6",  "results/raw/criteria_claude_both_k8.jsonl", "full", 8)],
                                                    -7, 8,  "right"),
        ("Mini k=8",            "mini_k8",          [("GPT mini",    "results/raw/base_both_k8.jsonl", "mini", 8),
                                                     ("Haiku 4.5",   "results/raw/base_claude_both_k8.jsonl", "mini", 8)],
                                                    -7, 0,  "right"),
        ("Mini+Criteria k=8",   "criteria_mini_k8", [("GPT mini",    "results/raw/criteria_both_k8.jsonl", "mini", 8),
                                                     ("Haiku 4.5",   "results/raw/criteria_claude_both_k8.jsonl", "mini", 8)],
                                                    7, 0,   "left"),
        ("Nano k=8",            "nano_k8",          [("GPT nano",    "results/raw/base_nano_k8.jsonl", "nano", 8)],
                                                    7, 0,   "left"),
        ("Calibration (low)",   "cal_low",          [("GPT-5.4",     "results/raw/cal-low_both_k8.jsonl", "full", 8)],
                                                    7, -8,  "left"),
        ("Combined",            "combined",         [("GPT-5.4",     "results/raw/combined_both_k8.jsonl", "full", 8)],
                                                    7, 0,   "left"),
    ]

    points = []
    for label, cost_key, candidates, dx, dy, ha in spec:
        cost_m = metrics.get(cost_key, {}).get("cost", {})
        cost_per_ex = cost_m.get("cost_per_example")
        if cost_per_ex is None:
            print(f"  pareto: missing cost for {label}")
            continue
        best_acc = -1
        best_provider = None
        for provider, path, model, k in candidates:
            a = _acc(path, model, k)
            if a is not None and a > best_acc:
                best_acc, best_provider = a, provider
        if best_provider is None:
            continue
        # Drop the provider tag from the displayed label — only the class+condition
        # is shown on the chart. The provider that supplied each point is recorded
        # in the caption and Table 4.
        points.append((cost_per_ex, best_acc, label, dx, dy, ha))

    if not points:
        print("  Skipping Pareto: no data")
        return

    # Baseline cost is from the Baseline (GPT full k=1) entry's GPT cost.
    baseline_cost = next((c for c, a, l, *_ in points if l.startswith("Baseline")), None)
    if not baseline_cost:
        print("  Skipping Pareto: no baseline")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    rel = [(c / baseline_cost, a, l, dx, dy, h) for c, a, l, dx, dy, h in points]

    for rc, acc, label, dx, dy, ha in rel:
        color = COLORS.get(label, "gray")
        ax.scatter(rc, acc, s=60, color=color, zorder=5)
        ax.annotate(label, (rc, acc), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8,
                    va="center", ha=ha)

    # Pareto frontier (monotone increasing accuracy with cost)
    sorted_pts = sorted([(c, a) for c, a, _, _, _, _ in rel])
    frontier, best_acc = [], -1
    for c, a in sorted_pts:
        if a > best_acc:
            frontier.append((c, a))
            best_acc = a
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k--", alpha=0.3, lw=1)

    ax.set_xscale("log")
    ax.set_xlim(0.3, 12)
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Cost (x baseline)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost-Accuracy Pareto Frontier")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}x"))
    _save(fig, "pareto_frontier")


# ===================================================================
# Figure 3: Diminishing returns
# ===================================================================

def plot_diminishing_returns(metrics):
    dr_full = metrics.get("diminishing_returns_full")
    dr_mini = metrics.get("diminishing_returns_mini")
    if not dr_full:
        print("  Skipping diminishing returns: no data")
        return

    ks = sorted(int(k) for k in dr_full.keys())
    accs_full = [dr_full[str(k)] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, accs_full, "b-o", lw=2, ms=6, label="Full (GPT-5.4)")

    if dr_mini:
        accs_mini = [dr_mini[str(k)] for k in ks]
        ax.plot(ks, accs_mini, "r-s", lw=2, ms=6, label="Mini (GPT-5.4 mini)")

    dr_nano = metrics.get("diminishing_returns_nano")
    if dr_nano:
        accs_nano = [dr_nano[str(k)] for k in ks]
        ax.plot(ks, accs_nano, "g-^", lw=2, ms=6, label="Nano (GPT-5.4 nano)")

    ax.axhline(0.25, color="gray", linestyle="--", lw=0.8, label="Random")
    ax.set_xlabel("Ensemble size (k)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Diminishing Returns of Ensembling")
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    _save(fig, "diminishing_returns")


# ===================================================================
# Figure 4: Variance error signal
# ===================================================================

def plot_variance_error_signal(metrics):
    auc_data = metrics.get("variance_auc", {}).get("baseline_k8", {})
    auc = auc_data.get("auc")
    fpr = auc_data.get("roc_fpr")
    tpr = auc_data.get("roc_tpr")

    if not (fpr and tpr and auc is not None):
        print("  Skipping variance ROC figure: no AUC data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Variance as incorrectness classifier")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _save(fig, "variance_error_signal")


# ===================================================================
# Figure 5: Variance correlation (mini vs full)
# ===================================================================

def plot_variance_correlation(metrics):
    data = _find_collection("base_both")
    if not data:
        print("  Skipping variance correlation: no data")
        return

    from scipy import stats as sp_stats

    has_nano = data and any("nano_scores" in r for r in data)
    n_plots = 2 if has_nano else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    # Mini vs Full
    mini_vars, full_vars = [], []
    for r in data:
        if "mini_scores" not in r or "full_scores" not in r:
            continue
        mv = [s for s in r["mini_scores"][0] if s is not None]
        fv = [s for s in r["full_scores"][0] if s is not None]
        if len(mv) > 1 and len(fv) > 1:
            mini_vars.append(float(np.var(mv)))
            full_vars.append(float(np.var(fv)))

    if len(mini_vars) >= 3:
        corr, _ = sp_stats.pearsonr(mini_vars, full_vars)
        clip = float(np.percentile(mini_vars + full_vars, 99))
        ax = axes[0]
        ax.scatter(mini_vars, full_vars, alpha=0.4, s=20, c="#4C72B0")
        ax.plot([0, clip * 1.1], [0, clip * 1.1], "r--", alpha=0.5, lw=1, label="y = x")
        ax.set_xlabel("Mini model score variance")
        ax.set_ylabel("Full model score variance")
        ax.set_title(f"Mini vs Full (r = {corr:.3f})")
        ax.legend()
        ax.set_xlim(0, clip * 1.1)
        ax.set_ylim(0, clip * 1.1)
        ax.set_aspect("equal")

    # Nano vs Full
    if has_nano:
        nano_vars, full_vars_n = [], []
        for r in data:
            if "nano_scores" not in r or "full_scores" not in r:
                continue
            nv = [s for s in r["nano_scores"][0] if s is not None]
            fv = [s for s in r["full_scores"][0] if s is not None]
            if len(nv) > 1 and len(fv) > 1:
                nano_vars.append(float(np.var(nv)))
                full_vars_n.append(float(np.var(fv)))

        if len(nano_vars) >= 3:
            corr_n, _ = sp_stats.pearsonr(nano_vars, full_vars_n)
            clip_n = float(np.percentile(nano_vars + full_vars_n, 99))
            ax = axes[1]
            ax.scatter(nano_vars, full_vars_n, alpha=0.4, s=20, c="#55A868")
            ax.plot([0, clip_n * 1.1], [0, clip_n * 1.1], "r--", alpha=0.5, lw=1, label="y = x")
            ax.set_xlabel("Nano model score variance")
            ax.set_ylabel("Full model score variance")
            ax.set_title(f"Nano vs Full (r = {corr_n:.3f})")
            ax.legend()
            ax.set_xlim(0, clip_n * 1.1)
            ax.set_ylim(0, clip_n * 1.1)
            ax.set_aspect("equal")

    fig.suptitle("Score Variance Correlation", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "variance_correlation")


# ===================================================================
# Figure 6: Soft blending
# ===================================================================

def plot_soft_blending(metrics):
    data = _find_collection("base_both")
    if not data or "mini_scores" not in data[0]:
        print("  Skipping soft blending: no dual-model data")
        return

    all_stds = _compute_mini_stds(data)
    unique_stds = sorted(set(s for stds in all_stds for s in stds))

    midpoints, accs, weights = [], [], []
    for m in unique_stds:
        r = compute_blend_accuracy(data, m)
        midpoints.append(m)
        accs.append(r["accuracy"])
        weights.append(r["mean_weight"])

    full_acc = compute_accuracy(data, model="full")["overall"]
    mini_acc = compute_accuracy(data, model="mini")["overall"]
    best_idx = max(range(len(accs)), key=lambda i: accs[i])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(weights, accs, color="#C44E52", lw=2)
    ax.scatter([weights[best_idx]], [accs[best_idx]], color="#C44E52", s=80, zorder=5,
               label=f"Best: {accs[best_idx]:.1%} at w={weights[best_idx]:.2f}")
    ax.axhline(full_acc, color="gray", ls=":", alpha=0.7, label=f"Full k=8 ({full_acc:.1%})")
    ax.axhline(mini_acc, color="orange", ls=":", alpha=0.7, label=f"Mini k=8 ({mini_acc:.1%})")
    ax.set_xlabel("Mean blend weight w (0=mini, 1=full)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Soft Blending: Accuracy vs Blend Weight")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "soft_blending")


# ===================================================================
# Figure 7: Escalation Pareto
# ===================================================================

def plot_escalation_pareto(metrics):
    esc = metrics.get("escalation_hard")
    if not esc or not esc.get("thresholds"):
        print("  Skipping escalation Pareto: no data")
        return

    ref = esc["reference"]
    full_acc = ref.get("full_k8", {}).get("overall", 0)
    mini_acc = ref.get("mini_k8", {}).get("overall", 0)

    costs = [t["effective_cost_ratio"] for t in esc["thresholds"]]
    accs = [t["accuracy"] for t in esc["thresholds"]]

    # Pareto
    frontier, best_a = [], -1
    for c, a in sorted(zip(costs, accs)):
        if a > best_a:
            frontier.append((c, a))
            best_a = a

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(costs, accs, s=18, alpha=0.5, c="#4C72B0", label="Threshold sweep")
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, c="#4C72B0", lw=2, label="Pareto frontier")
    ax.axhline(full_acc, color="gray", ls=":", alpha=0.7, label=f"Full k=8 ({full_acc:.1%})")
    ax.axhline(mini_acc, color="orange", ls=":", alpha=0.7, label=f"Mini k=8 ({mini_acc:.1%})")
    ax.set_xlabel("Cost (fraction of full k=8)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Hard Escalation: Accuracy vs Cost")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "per_response_escalation_pareto")


# ===================================================================
# Figure 8: Variance-informed ensembling
# ===================================================================

def plot_var_informed(metrics):
    vie = metrics.get("var_informed_optimised")
    if not vie or not vie.get("grid_results"):
        print("  Skipping var-informed figure: no data")
        return

    grid = vie["grid_results"]
    costs = [g["effective_cost_ratio"] if "effective_cost_ratio" in g else g["mean_n2"] / 8 for g in grid]
    accs = [g["accuracy"] for g in grid]

    # Pareto
    frontier, best_a = [], -1
    for c, a in sorted(zip(costs, accs)):
        if a > best_a:
            frontier.append((c, a))
            best_a = a

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(costs, accs, alpha=0.15, s=12, c="gray", label=f"Grid ({len(grid)})")
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k-", lw=2, label="Pareto frontier")

    tb = vie.get("test_best")
    tbc = vie.get("test_best_constrained")
    if tb:
        ax.scatter([tb.get("mean_n2", 0) / 8], [tb["accuracy"]],
                   s=120, c="red", marker="*", zorder=6, label=f"Best ({tb['accuracy']:.1%})")
    if tbc:
        ax.scatter([tbc.get("mean_n2", 0) / 8], [tbc["accuracy"]],
                   s=120, c="green", marker="*", zorder=6, label=f"Budget ({tbc['accuracy']:.1%})")

    ax.set_xlabel("Cost (fraction of full k=8)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Variance-Informed Ensembling")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "variance_informed_ensembling")


# ===================================================================
# Figure 9: Mini-full convergence
# ===================================================================

def plot_convergence(metrics):
    conv = metrics.get("convergence")
    if not conv or not conv.get("by_k"):
        print("  Skipping convergence: no data")
        return

    conv_nano = metrics.get("convergence_nano")

    ks = [e["k"] for e in conv["by_k"]]
    agr = [e["agreement"] for e in conv["by_k"]]
    rcs = [e.get("rank_correlation") for e in conv["by_k"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, agr, "b-o", lw=2, ms=6, label="Mini agreement")
    ax1.set_xlabel("Ensemble size (k)")
    ax1.set_ylabel("Agreement with full model")
    ax1.set_xticks(range(1, max(ks) + 1))
    ax1.set_ylim(0.4, 1.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    if conv_nano and conv_nano.get("by_k"):
        nano_ks = [e["k"] for e in conv_nano["by_k"]]
        nano_agr = [e["agreement"] for e in conv_nano["by_k"]]
        ax1.plot(nano_ks, nano_agr, "g-^", lw=2, ms=6, label="Nano agreement")

    valid_rcs = [(k, r) for k, r in zip(ks, rcs) if r is not None]
    if valid_rcs:
        ax2 = ax1.twinx()
        ax2.plot([k for k, r in valid_rcs], [r for k, r in valid_rcs],
                 "b--s", lw=1, ms=4, alpha=0.5, label="Mini rank corr")
        ax2.set_ylabel("Spearman rank correlation")
        ax2.set_ylim(0.4, 1.0)

        if conv_nano and conv_nano.get("by_k"):
            nano_rcs = [(e["k"], e.get("rank_correlation")) for e in conv_nano["by_k"]]
            valid_nano_rcs = [(k, r) for k, r in nano_rcs if r is not None]
            if valid_nano_rcs:
                ax2.plot([k for k, r in valid_nano_rcs], [r for k, r in valid_nano_rcs],
                         "g--D", lw=1, ms=4, alpha=0.5, label="Nano rank corr")

    ax1.set_title("Model Agreement with Full (k=8) vs Ensemble Size")
    lines1, labels1 = ax1.get_legend_handles_labels()
    if valid_rcs:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        ax1.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "mini_full_convergence")


# ===================================================================
# Figure 10: Temperature sweep
# ===================================================================

def plot_temperature_sweep(metrics):
    sweep = metrics.get("temperature_sweep")
    if not sweep:
        print("  Skipping temperature sweep: no data")
        return

    # Parse: keys are like "base_full_k8_t0.3_k1", "base_full_k8_t0.3_k8"
    k1_points, k8_points = {}, {}
    for key, data in sweep.items():
        # Extract temperature
        parts = key.split("_t")
        if len(parts) < 2:
            continue
        rest = parts[1]  # "0.3_k1" or "1.0_k8"
        rest_parts = rest.split("_k")
        if len(rest_parts) < 2:
            continue
        temp = float(rest_parts[0])
        k = int(rest_parts[1])
        if k == 1:
            k1_points[temp] = data
        elif k == 8:
            k8_points[temp] = data

    if not k1_points and not k8_points:
        print("  Skipping temperature sweep: no parseable data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if k1_points:
        temps = sorted(k1_points.keys())
        accs = [k1_points[t]["accuracy"] for t in temps]
        ci_lo = [k1_points[t].get("ci_low", accs[i]) for i, t in enumerate(temps)]
        ci_hi = [k1_points[t].get("ci_high", accs[i]) for i, t in enumerate(temps)]
        ax.errorbar(temps, accs, yerr=[np.array(accs) - np.array(ci_lo),
                                        np.array(ci_hi) - np.array(accs)],
                     fmt="b-o", lw=2, ms=6, capsize=4, label="k=1")

    if k8_points:
        temps = sorted(k8_points.keys())
        accs = [k8_points[t]["accuracy"] for t in temps]
        ci_lo = [k8_points[t].get("ci_low", accs[i]) for i, t in enumerate(temps)]
        ci_hi = [k8_points[t].get("ci_high", accs[i]) for i, t in enumerate(temps)]
        ax.errorbar(temps, accs, yerr=[np.array(accs) - np.array(ci_lo),
                                        np.array(ci_hi) - np.array(accs)],
                     fmt="r-s", lw=2, ms=6, capsize=4, label="k=8")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline Accuracy vs Temperature")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.set_xticks([0.0, 0.3, 0.7, 1.0])
    fig.tight_layout()
    _save(fig, "temperature_sweep")


# ===================================================================
# Figure 11: Cross-model generalisation
# ===================================================================

def plot_cross_model_gain(metrics):
    """Bar chart: criteria + ensembling deltas across model families.

    Computes accuracies directly from raw collections so it works
    independently of the (slow) full metrics regeneration.
    """
    from analysis.compute_metrics import load_collection, compute_accuracy

    def _acc(path, model, k):
        try:
            return compute_accuracy(load_collection(path), model=model, k_subset=k)["overall"]
        except FileNotFoundError:
            return None

    spec = [
        ("GPT-5.4",
         ("results/raw/base_both_k8.jsonl",     "full", 1),
         ("results/raw/base_both_k8.jsonl",     "full", 8),
         ("results/raw/criteria_both_k8.jsonl", "full", 8)),
        ("GPT-5.4 mini",
         ("results/raw/base_both_k8.jsonl",     "mini", 1),
         ("results/raw/base_both_k8.jsonl",     "mini", 8),
         ("results/raw/criteria_both_k8.jsonl", "mini", 8)),
        ("Claude Sonnet 4.6",
         ("results/raw/base_claude_both_k8.jsonl",     "full", 1),
         ("results/raw/base_claude_both_k8.jsonl",     "full", 8),
         ("results/raw/criteria_claude_both_k8.jsonl", "full", 8)),
        ("Claude Haiku 4.5",
         ("results/raw/base_claude_both_k8.jsonl",     "mini", 1),
         ("results/raw/base_claude_both_k8.jsonl",     "mini", 8),
         ("results/raw/criteria_claude_both_k8.jsonl", "mini", 8)),
    ]

    rows = []
    for label, base_spec, ens_spec, both_spec in spec:
        b = _acc(*base_spec)
        e = _acc(*ens_spec)
        c = _acc(*both_spec)
        if b is None or e is None or c is None:
            print(f"  cross_model_gain: skipping {label} (missing data)")
            continue
        rows.append((label, b, e, c))

    if len(rows) < 2:
        print("  Skipping cross-model figure: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    n_groups = len(rows)
    x = np.arange(n_groups)
    bar_w = 0.26

    # Use distinct colorblind-safe colors for the three conditions
    cols = {"base": "#0173b2", "ens": "#de8f05", "both": "#d55e00"}

    base_vals = [r[1] for r in rows]
    ens_vals  = [r[2] for r in rows]
    both_vals = [r[3] for r in rows]

    ax.bar(x - bar_w, base_vals, bar_w, label="Baseline ($k{=}1$)",          color=cols["base"])
    ax.bar(x,         ens_vals,  bar_w, label="Ensemble ($k{=}8$)",          color=cols["ens"])
    ax.bar(x + bar_w, both_vals, bar_w, label="Criteria + ensemble ($k{=}8$)", color=cols["both"])

    # Annotate the delta (criteria+ensemble vs baseline) on top of the right-most bar
    for i, (_, b, _, c) in enumerate(rows):
        delta_pp = (c - b) * 100
        ax.annotate(f"+{delta_pp:.1f}pp",
                    xy=(i + bar_w, c), xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.55, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Cross-model accuracy: criteria + ensembling")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    _save(fig, "cross_model_gain")


# ===================================================================
# Figure 12: Cross-model diminishing returns
# ===================================================================

def plot_cross_model_diminishing(metrics):
    """k=1..8 accuracy curves for the five-model cross-family panel."""
    from analysis.compute_metrics import load_collection, compute_accuracy

    def _curve(path, model):
        try:
            data = load_collection(path)
        except FileNotFoundError:
            return None
        return [compute_accuracy(data, model=model, k_subset=k)["overall"] for k in range(1, 9)]

    spec = [
        ("GPT-5.4",          "results/raw/base_both_k8.jsonl",        "full", "#0173b2", "-"),
        ("GPT-5.4 mini",     "results/raw/base_both_k8.jsonl",        "mini", "#de8f05", "-"),
        ("GPT-5.4 nano",     "results/raw/base_nano_k8.jsonl",        "nano", "#ca9161", "-"),
        ("Claude Sonnet 4.6", "results/raw/base_claude_both_k8.jsonl", "full", "#d55e00", "--"),
        ("Claude Haiku 4.5", "results/raw/base_claude_both_k8.jsonl", "mini", "#cc78bc", "--"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for label, path, model, color, style in spec:
        accs = _curve(path, model)
        if accs is None:
            print(f"  cross_model_diminishing: skipping {label}")
            continue
        ax.plot(range(1, 9), accs, marker="o", color=color, linestyle=style, label=label, lw=1.5)
        plotted += 1

    if plotted < 2:
        print("  Skipping cross-model diminishing: insufficient data")
        return

    ax.set_xlabel("Ensemble size $k$")
    ax.set_ylabel("Accuracy")
    ax.set_title("Diminishing returns across model families")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    _save(fig, "cross_model_diminishing")


# ===================================================================
# Main
# ===================================================================

def main():
    print("Loading metrics...")
    metrics = _load_metrics()

    print("\nGenerating figures...")

    print("\n  1. Hero Accuracy")
    plot_hero_accuracy(metrics)

    print("\n  2. Pareto Frontier")
    plot_pareto_frontier(metrics)

    print("\n  3. Diminishing Returns")
    plot_diminishing_returns(metrics)

    print("\n  4. Variance Error Signal")
    plot_variance_error_signal(metrics)

    print("\n  5. Variance Correlation")
    plot_variance_correlation(metrics)

    print("\n  6. Soft Blending")
    plot_soft_blending(metrics)

    print("\n  7. Hard Escalation Pareto")
    plot_escalation_pareto(metrics)

    print("\n  8. Variance-Informed Ensembling")
    plot_var_informed(metrics)

    print("\n  9. Mini-Full Convergence")
    plot_convergence(metrics)

    print("\n  10. Temperature Sweep")
    plot_temperature_sweep(metrics)

    print(f"\nDone! Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()

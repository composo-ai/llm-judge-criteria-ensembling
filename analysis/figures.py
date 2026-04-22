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
    "Baseline": "#4C72B0",
    "Ensemble k=8": "#DD8452",
    "Criteria": "#55A868",
    "Calibration (low)": "#8172B3",
    "Calibration (high)": "#C44E52",
    "Calibration (both)": "#937860",
    "Calibration (cross)": "#DA8BC3",
    "Mini k=8": "#CCB974",
    "Nano k=8": "#64B5F6",
    "Soft blend": "#E5AE38",
    "Combined": "#B07AA1",
    "Criteria k=8": "#76B7B2",
    "Combined + blend": "#FF9DA7",
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
    # Build {condition_name: {subset: accuracy}} from metrics
    conditions = {}
    # Only full-dataset conditions (no test-set to avoid mixing evaluation protocols)
    display_order = [
        ("baseline", "Baseline"),
        ("criteria", "Criteria (k=1)"),
        ("cal_low", "Calibration (low)"),
        ("ensemble_k8", "Ensemble k=8"),
        ("mini_k8", "Mini k=8"),
        ("nano_k8", "Nano k=8"),
        ("criteria_k8", "Criteria k=8"),
        ("combined", "Combined"),
    ]

    for key, label in display_order:
        m = metrics.get(key)
        if m and "accuracy" in m:
            conditions[label] = {
                sub: d["accuracy"]
                for sub, d in m["accuracy"]["by_subset"].items()
            }

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
        color = COLORS.get(name, f"C{i}")
        ax.bar(x + i * bar_w, vals, bar_w, label=name, color=color)

    ax.axhline(0.25, color="gray", linestyle="--", lw=0.8, label="Random (25%)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Condition and Category")
    ax.set_xticks(x + bar_w * (n_cond - 1) / 2)
    ax.set_xticklabels(subsets)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1.0))
    _save(fig, "hero_accuracy")


# ===================================================================
# Figure 2: Pareto frontier
# ===================================================================

def plot_pareto_frontier(metrics):
    points = []  # (cost_per_ex, accuracy, label)

    # Only full-dataset conditions (no test-set numbers to avoid mixing protocols)
    for key, label in [("baseline", "Baseline"), ("criteria", "Criteria (k=1)"),
                       ("criteria_k8", "Criteria k=8"),
                       ("ensemble_k8", "Ensemble k=8"), ("mini_k8", "Mini k=8"),
                       ("nano_k8", "Nano k=8"),
                       ("cal_low", "Calibration (low)"),
                       ("combined", "Combined")]:
        m = metrics.get(key)
        if m and "accuracy" in m and "cost" in m:
            points.append((m["cost"]["cost_per_example"], m["accuracy"]["overall"], label))

    if not points:
        print("  Skipping Pareto: no data")
        return

    baseline_cost = next((c for c, a, l in points if l == "Baseline"), None)
    if not baseline_cost:
        print("  Skipping Pareto: no baseline")
        return

    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(10, 6))
    rel = [(c / baseline_cost, a, l) for c, a, l in points]

    texts = []
    for rc, acc, label in rel:
        color = COLORS.get(label, "gray")
        ax.scatter(rc, acc, s=80, color=color, zorder=5)
        texts.append(ax.text(rc, acc, label, fontsize=8))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Pareto frontier
    sorted_pts = sorted([(c, a) for c, a, _ in rel])
    frontier, best_acc = [], -1
    for c, a in sorted_pts:
        if a > best_acc:
            frontier.append((c, a))
            best_acc = a
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k--", alpha=0.3, lw=1)

    ax.set_xlabel("Cost (x baseline)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost-Accuracy Pareto Frontier")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
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
    data = _find_collection("base_both")
    if not data:
        print("  Skipping variance figure: no base collection")
        return

    correct_stds, incorrect_stds = [], []
    for r in data:
        scores_per_resp = r.get("full_scores")
        if not scores_per_resp:
            continue
        stds, means = [], []
        for scores in scores_per_resp:
            valid = [s for s in scores if s is not None]
            stds.append(float(np.std(valid)) if len(valid) > 1 else 0.0)
            means.append(float(np.mean(valid)) if valid else None)
        if any(m is None for m in means):
            continue
        ex_std = float(np.mean(stds))
        max_s = max(means)
        winners = [i for i, m in enumerate(means) if m == max_s]
        if len(winners) == 1 and winners[0] == 0:
            correct_stds.append(ex_std)
        else:
            incorrect_stds.append(ex_std)

    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot([correct_stds, incorrect_stds], positions=[0, 1],
                          showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_ylabel("Mean score std")
    auc_data = metrics.get("variance_auc", {}).get("baseline_k8", {})
    auc = auc_data.get("auc")
    title = "Variance as Error Signal"
    if auc is not None:
        title += f"  (AUC = {auc:.3f})"
    ax.set_title(title)
    ax.text(0, ax.get_ylim()[1] * 0.95, f"n={len(correct_stds)}", ha="center", fontsize=9)
    ax.text(1, ax.get_ylim()[1] * 0.95, f"n={len(incorrect_stds)}", ha="center", fontsize=9)
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

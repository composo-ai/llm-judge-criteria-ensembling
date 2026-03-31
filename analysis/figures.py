"""Generate figures for RB2 experiment results.

Run as: python analysis/figures.py
Reads from results/tables/all_metrics.json and/or results/raw/*.jsonl.
Saves figures to figures/ as PNG (300 DPI).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from compute_metrics import (
    load_results,
    compute_accuracy,
    compute_escalation_metrics,
    compute_ensemble_convergence,
    compute_variance_metrics,
)

# --- Style ---
CONDITION_COLORS = {
    "Baseline": "#4C72B0",
    "Ensemble k=8": "#DD8452",
    "Criteria": "#55A868",
    "Calibration (high)": "#C44E52",
    "Calibration (low)": "#8172B3",
    "Calibration (both)": "#937860",
    "Calibration (cross)": "#DA8BC3",
    "Escalation": "#8C8C8C",
    "Mini only": "#CCB974",
    "Soft blend": "#E5AE38",
    "Var-Informed": "#76B7B2",
    "Combined (full k=8)": "#B07AA1",
    "Combined (blend)": "#FF9DA7",
}

SUBSET_ORDER = ["Factuality", "Focus", "Math", "Precise IF", "Safety"]

FIGURES_DIR = Path("figures")
RAW_DIR = Path("results/raw")
TABLES_DIR = Path("results/tables")


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"  Saved figures/{name}.png")
    plt.close(fig)


def _load_metrics() -> dict:
    path = TABLES_DIR / "all_metrics.json"
    if not path.exists():
        print(
            "ERROR: results/tables/all_metrics.json not found. Run compute_metrics.py first."
        )
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def _collect_conditions(metrics: dict) -> dict[str, dict]:
    """Map condition display names to their accuracy dicts."""
    conditions = {}
    for key, m in metrics.items():
        if m["type"] == "baseline":
            conditions["Baseline"] = m["accuracy"]
        elif m["type"] == "ensemble":
            conditions["Ensemble k=8"] = m["accuracy"]
        elif m["type"] == "criteria":
            conditions["Criteria"] = m["accuracy"]
        elif m["type"] == "calibration":
            variant = key.split("_")[1] if "_" in key else "random"
            label = f"Calibration ({variant})"
            conditions[label] = m["accuracy"]
        elif m["type"] == "escalation":
            # Always-mini from reference points (full only is redundant with Ensemble k=8)
            esc = m.get("escalation", {})
            ref = esc.get("reference", {})
            if "always_mini" in ref:
                conditions["Mini model k=8"] = ref["always_mini"]

            # Best soft blending point (highest accuracy)
            blended = m.get("escalation_blended", {})
            if blended.get("thresholds"):
                best = max(blended["thresholds"], key=lambda t: t["accuracy"])
                best_acc = {
                    "overall": best["accuracy"],
                    "n": best["n"],
                    "n_correct": sum(
                        s["n_correct"] for s in best["by_subset"].values()
                    ),
                    "n_tied": 0,
                    "by_subset": best["by_subset"],
                }
                conditions["Soft blend"] = best_acc

            # Best variance-informed ensembling
            vie = m.get("variance_informed_ensembling", {})
            vie_best = vie.get("best")
            if vie_best:
                conditions["Var-Informed"] = {
                    "overall": vie_best["accuracy"],
                    "n": vie_best["n"],
                    "n_correct": sum(
                        s["n_correct"] for s in vie_best["by_subset"].values()
                    ),
                    "n_tied": 0,
                    "by_subset": vie_best["by_subset"],
                }
        elif m["type"] == "combined":
            # Full model accuracy (all k)
            esc = m.get("escalation", {})
            ref = esc.get("reference", {})
            if "always_full" in ref:
                conditions["Combined (full k=8)"] = ref["always_full"]

            # Best soft blend
            blended = m.get("escalation_blended", {})
            if blended.get("thresholds"):
                best = max(blended["thresholds"], key=lambda t: t["accuracy"])
                conditions["Combined + soft blend"] = {
                    "overall": best["accuracy"],
                    "n": best["n"],
                    "n_correct": sum(
                        s["n_correct"] for s in best["by_subset"].values()
                    ),
                    "n_tied": 0,
                    "by_subset": best["by_subset"],
                }
    return conditions


def plot_hero_accuracy(metrics: dict):
    """Figure 1: Grouped bar chart of accuracy by condition and category."""
    conditions = _collect_conditions(metrics)
    if not conditions:
        print("  Skipping hero figure: no accuracy data")
        return

    subsets = [
        s
        for s in SUBSET_ORDER
        if any(s in cond.get("by_subset", {}) for cond in conditions.values())
    ]
    cond_names = list(conditions.keys())
    n_cond = len(cond_names)
    n_subsets = len(subsets)

    fig, ax = plt.subplots(figsize=(max(10, n_subsets * 2) + 3, 6))

    bar_width = 0.8 / n_cond
    x = np.arange(n_subsets)

    for i, cond_name in enumerate(cond_names):
        acc_data = conditions[cond_name]
        values = [
            acc_data.get("by_subset", {}).get(s, {}).get("accuracy", 0) for s in subsets
        ]
        color = CONDITION_COLORS.get(cond_name, f"C{i}")
        bars = ax.bar(
            x + i * bar_width, values, bar_width, label=cond_name, color=color
        )

    ax.axhline(
        y=0.25, color="gray", linestyle="--", linewidth=0.8, label="Random (25%)"
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Condition and Category")
    ax.set_xticks(x + bar_width * (n_cond - 1) / 2)
    ax.set_xticklabels(subsets)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    _save(fig, "hero_accuracy")


def plot_pareto_frontier(metrics: dict):
    """Figure 2: Cost-Accuracy Pareto frontier."""
    points = []
    baseline_cost = None

    for key, m in metrics.items():
        if not m.get("accuracy") or not m.get("cost"):
            continue
        cost = m["cost"].get(
            "total_cost", m["cost"].get("cost_per_example", 0) * m.get("n", 1)
        )
        acc = m["accuracy"]["overall"]

        if m["type"] == "baseline":
            baseline_cost = cost
            label = "Baseline"
        elif m["type"] == "ensemble":
            label = "Ensemble k=8"
        elif m["type"] == "criteria":
            label = "Criteria"
        elif m["type"] == "calibration":
            label = "Calibration (variants)"
        elif m["type"] in ("escalation", "combined"):
            esc = m.get("escalation", {})
            ref = esc.get("reference", {})
            # Add mini-only once (from escalation only, not combined)
            if "always_mini" in ref and m["type"] == "escalation":
                mini_cost = m["cost"].get("mini_cost", cost * 0.1)
                points.append(
                    (mini_cost, ref["always_mini"]["overall"], "Mini model k=8")
                )
            # Best cost-efficient hard routing point (from escalation only)
            if m["type"] == "escalation" and esc.get("thresholds"):
                # Find Pareto frontier, pick the point with best accuracy below 0.3 cost ratio
                sorted_thresh = sorted(esc["thresholds"], key=lambda t: t["effective_cost_ratio"])
                best_cheap = None
                for t in sorted_thresh:
                    if t["effective_cost_ratio"] < 0.3:
                        if best_cheap is None or t["accuracy"] > best_cheap["accuracy"]:
                            best_cheap = t
                if best_cheap:
                    hr_cost = cost * best_cheap["effective_cost_ratio"]
                    points.append(
                        (hr_cost, best_cheap["accuracy"], f"Hard routing (θ={best_cheap['threshold']:.2f})")
                    )
            # Best soft blend — use combined if available, else escalation
            blended = m.get("escalation_blended", {})
            if blended.get("thresholds"):
                best_b = max(blended["thresholds"], key=lambda t: t["accuracy"])
                blend_cost = cost * best_b["effective_cost_ratio"]
                label = (
                    "Combined + soft blend" if m["type"] == "combined" else "Soft blend"
                )
                points.append((blend_cost, best_b["accuracy"], label))
            # Best var-informed cost-constrained (escalation only)
            if m["type"] == "escalation":
                vie = m.get("variance_informed_ensembling", {})
                if vie.get("best_cost_constrained"):
                    bc = vie["best_cost_constrained"]
                    vie_cost = cost * bc["effective_cost_ratio"]
                    points.append(
                        (
                            vie_cost,
                            bc["accuracy"],
                            f"Var-informed (n̄={bc['mean_n2']:.1f})",
                        )
                    )
            continue
        else:
            label = key

        points.append((cost, acc, label))

    if not points or baseline_cost is None or baseline_cost == 0:
        print("  Skipping Pareto figure: insufficient data")
        return

    # Collapse calibration variants into a single representative point (best accuracy)
    cal_points = [(c, a) for c, a, l in points if l == "Calibration (variants)"]
    non_cal_points = [(c, a, l) for c, a, l in points if l != "Calibration (variants)"]
    if cal_points:
        best_cal = max(cal_points, key=lambda x: x[1])
        non_cal_points.append((best_cal[0], best_cal[1], "Calibration (variants)"))
    points = non_cal_points

    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(10, 6))

    rel_points = [(c / baseline_cost, a, l) for c, a, l in points]

    x_vals = [c for c, a, l in rel_points]
    x_range = max(x_vals) - min(x_vals)
    nudge = x_range * 0.02

    texts = []
    for rel_cost, acc, label in rel_points:
        color = CONDITION_COLORS.get(label, "gray")
        ax.scatter(rel_cost, acc, s=80, color=color, zorder=5)
        texts.append(ax.text(rel_cost + nudge, acc, label, fontsize=8))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Pareto frontier
    sorted_pts = sorted([(c, a) for c, a, _ in rel_points])
    frontier = []
    best_acc = -1
    for c, a in sorted_pts:
        if a > best_acc:
            frontier.append((c, a))
            best_acc = a
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Cost (× baseline)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost-Accuracy Pareto Frontier")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0.68, 0.88)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}×"))

    _save(fig, "pareto_frontier")


def plot_soft_blending(metrics: dict):
    """Figure: Soft blending accuracy vs cost ratio (sigmoid weighting)."""
    esc_key = None
    for key, m in metrics.items():
        if m["type"] == "escalation" and "escalation_blended" in m:
            esc_key = key
            break

    if esc_key is None:
        print("  Skipping soft blending figure: no blended escalation data")
        return

    data = metrics[esc_key].get("escalation_blended")
    if not data or not data.get("thresholds"):
        print("  Skipping soft blending figure: no threshold data")
        return

    ref = metrics[esc_key]["escalation"]["reference"]
    always_full_acc = ref.get("always_full", {}).get("overall", 0)
    always_mini_acc = ref.get("always_mini", {}).get("overall", 0)

    thresholds = data["thresholds"]
    accs = [t["accuracy"] for t in thresholds]
    weights = [t["mean_weight"] for t in thresholds]

    # Find best point
    best_idx = max(range(len(accs)), key=lambda i: accs[i])

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(weights, accs, color="#C44E52", linewidth=2)
    ax.scatter(
        [weights[best_idx]],
        [accs[best_idx]],
        color="#C44E52",
        s=80,
        zorder=5,
        label=f"Best: {accs[best_idx]:.1%} at w={weights[best_idx]:.2f}",
    )
    ax.axhline(
        y=always_full_acc,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Full model k=8 ({always_full_acc:.1%})",
    )
    ax.axhline(
        y=always_mini_acc,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"Mini model k=8 ({always_mini_acc:.1%})",
    )
    ax.set_xlabel("Mean blend weight $w$ (0 = all mini, 1 = all full)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Soft Blending: Accuracy vs Blend Weight")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "soft_blending")


def plot_per_response_escalation_pareto(metrics: dict):
    """Figure: Pareto frontier for per-response escalation (accuracy vs cost)."""
    esc_key = None
    for key, m in metrics.items():
        if m["type"] == "escalation" and "escalation" in m:
            esc_key = key
            break

    if esc_key is None:
        print("  Skipping per-response escalation pareto: no escalation data")
        return

    data = metrics[esc_key].get("escalation")
    if not data or not data.get("thresholds"):
        print("  Skipping per-response escalation pareto: no threshold data")
        return

    ref = metrics[esc_key]["escalation"]["reference"]
    always_full_acc = ref.get("always_full", {}).get("overall", 0)
    always_mini_acc = ref.get("always_mini", {}).get("overall", 0)

    thresholds = data["thresholds"]
    cost_ratios = [t["effective_cost_ratio"] for t in thresholds]
    accs = [t["accuracy"] for t in thresholds]
    pct_esc = [t["pct_escalated"] for t in thresholds]

    # Compute Pareto frontier (upper-left envelope sorted by cost)
    points = sorted(zip(cost_ratios, accs, pct_esc))
    frontier = []
    best_acc = -1
    for c, a, p in points:
        if a > best_acc:
            frontier.append((c, a))
            best_acc = a

    fig, ax = plt.subplots(figsize=(7, 5))

    # All threshold points
    ax.scatter(
        cost_ratios,
        accs,
        s=18,
        alpha=0.5,
        color="#4C72B0",
        zorder=2,
        label=f"Threshold sweep (n={len(thresholds)})",
    )

    # Pareto frontier
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, color="#4C72B0", linewidth=2, zorder=3, label="Pareto frontier")

    # Reference lines
    ax.axhline(
        y=always_full_acc,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Full model k=8 ({always_full_acc:.1%})",
    )
    ax.axhline(
        y=always_mini_acc,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"Mini model k=8 ({always_mini_acc:.1%})",
    )

    ax.set_xlabel("Cost ratio (relative to full model k=8)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Response Escalation: Accuracy vs Cost")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()

    _save(fig, "per_response_escalation_pareto")


def plot_variance_error_signal(metrics: dict):
    """Figure 4: Variance distributions for correct vs incorrect judgments."""
    # Load raw ensemble data
    ensemble_files = list(RAW_DIR.glob("ensemble_*.jsonl"))
    if not ensemble_files:
        print("  Skipping variance figure: no ensemble data")
        return

    results = load_results(ensemble_files[0])
    if not results:
        print("  Skipping variance figure: no valid ensemble results")
        return

    correct_stds = []
    incorrect_stds = []

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
            valid_mean = np.mean(valid) if valid else None
            means.append(valid_mean)

        if any(m is None for m in means):
            continue

        example_mean_std = float(np.mean(response_stds))
        max_score = max(means)
        winners = [i for i, m in enumerate(means) if m == max_score]
        correct = len(winners) == 1 and winners[0] == 0

        if correct:
            correct_stds.append(example_mean_std)
        else:
            incorrect_stds.append(example_mean_std)

    fig, ax = plt.subplots(figsize=(8, 6))

    parts = ax.violinplot(
        [correct_stds, incorrect_stds],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
    )

    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_ylabel("Mean score std (across ensemble calls)")
    ax.set_title("Variance as Error Signal: Do Uncertain Judgments Predict Errors?")

    # Add n counts
    ax.text(
        0, ax.get_ylim()[1] * 0.95, f"n={len(correct_stds)}", ha="center", fontsize=9
    )
    ax.text(
        1, ax.get_ylim()[1] * 0.95, f"n={len(incorrect_stds)}", ha="center", fontsize=9
    )

    _save(fig, "variance_error_signal")


def plot_mini_full_convergence(metrics: dict):
    """Figure 5: Mini-full agreement vs ensemble size."""
    esc_key = None
    for key, m in metrics.items():
        if m["type"] == "escalation" and "convergence" in m:
            esc_key = key
            break

    if esc_key is None:
        print("  Skipping convergence figure: no escalation data with convergence")
        return

    conv = metrics[esc_key]["convergence"]
    if not conv.get("by_k"):
        print("  Skipping convergence figure: empty convergence data")
        return

    ks = [e["k"] for e in conv["by_k"]]
    agreements = [e["agreement_with_full"] for e in conv["by_k"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, agreements, "b-o", linewidth=2, markersize=6, label="Overall")

    # Per-subset lines
    for subset, entries in conv.get("by_k_and_subset", {}).items():
        sub_ks = [e["k"] for e in entries]
        sub_agr = [e["agreement_with_full"] for e in entries]
        ax.plot(sub_ks, sub_agr, "--", alpha=0.4, linewidth=1, label=subset)

    ax.set_xlabel("Number of mini ensemble calls (k)")
    ax.set_ylabel("Agreement with full model (%)")
    ax.set_title("Mini-Full Model Agreement vs Ensemble Size")
    ax.set_xticks(range(1, max(ks) + 1))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    _save(fig, "mini_full_convergence")


def plot_diminishing_returns(metrics: dict):
    """Figure 6: Accuracy vs ensemble size (diminishing returns)."""
    # Find ensemble metrics with diminishing returns data
    ens_key = None
    for key, m in metrics.items():
        if m["type"] == "ensemble" and "diminishing_returns" in m:
            ens_key = key
            break

    if ens_key is None:
        print("  Skipping diminishing returns figure: no ensemble data")
        return

    dr = metrics[ens_key]["diminishing_returns"]
    ks = sorted(int(k) for k in dr.keys())
    accs = [dr[str(k)]["overall"] for k in ks]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, accs, "b-o", linewidth=2, markersize=6, label="Full model (GPT-5.4)")

    # Per-subset lines
    subsets = list(dr[str(ks[0])].get("by_subset", {}).keys())
    for subset in subsets:
        sub_accs = [dr[str(k)].get("by_subset", {}).get(subset, 0) for k in ks]
        ax.plot(ks, sub_accs, "--", alpha=0.4, linewidth=1, label=subset)

    # Also try to add mini model line from escalation data
    for key, m in metrics.items():
        if m["type"] == "escalation" and "convergence" in m:
            # Use escalation data to show mini accuracy at different k
            esc_results = load_results(RAW_DIR / f"{key}.jsonl")
            if esc_results:
                mini_accs = []
                for k_sub in ks:
                    adapted = [
                        {
                            "id": r["id"],
                            "subset": r["subset"],
                            "all_scores": r["mini_scores"],
                        }
                        for r in esc_results
                    ]
                    mini_acc = compute_accuracy(adapted, k_subset=k_sub)
                    mini_accs.append(mini_acc["overall"])
                ax.plot(
                    ks[: len(mini_accs)],
                    mini_accs,
                    "r-s",
                    linewidth=2,
                    markersize=6,
                    label="Mini model (GPT-5.4 mini)",
                )
            break

    ax.axhline(
        y=0.25, color="gray", linestyle="--", linewidth=0.8, label="Random (25%)"
    )
    ax.set_xlabel("Number of ensemble calls (k)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Diminishing Returns of Ensembling")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    _save(fig, "diminishing_returns")


def plot_variance_correlation(metrics: dict):
    """Figure 8: Mini vs Full variance correlation scatter plot."""
    esc_file = None
    for key, m in metrics.items():
        if m["type"] == "escalation":
            esc_file = RAW_DIR / f"{key}.jsonl"
            break

    if esc_file is None or not esc_file.exists():
        print("  Skipping variance correlation: no escalation data")
        return

    results = load_results(esc_file)
    if not results:
        print("  Skipping variance correlation: no valid results")
        return

    mini_vars = []
    full_vars = []

    for r in results:
        # Use the chosen response (index 0) variance, like the reference figure
        mini_scores = [s for s in r["mini_scores"][0] if s is not None]
        full_scores = [s for s in r["full_scores"][0] if s is not None]

        if len(mini_scores) > 1 and len(full_scores) > 1:
            mini_vars.append(float(np.var(mini_scores)))
            full_vars.append(float(np.var(full_scores)))

    if len(mini_vars) < 3:
        print("  Skipping variance correlation: insufficient data")
        return

    from scipy import stats as sp_stats

    corr, _ = sp_stats.pearsonr(mini_vars, full_vars)

    # Clip to 99th percentile to show the dense cluster clearly
    clip_val = float(np.percentile(mini_vars + full_vars, 99))
    n_outliers = sum(
        1 for m, f in zip(mini_vars, full_vars) if m > clip_val or f > clip_val
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(mini_vars, full_vars, alpha=0.4, s=20, c="#4C72B0", zorder=3)

    # y=x reference line
    max_val = clip_val * 1.1
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, linewidth=1, label="y = x")

    ax.set_xlabel("Variance of chosen scores (GPT-5.4 mini)")
    ax.set_ylabel("Variance of chosen scores (GPT-5.4)")
    ax.set_title(
        f"Chosen Scores Variance Comparison\n(Correlation: {corr:.3f}, {n_outliers} outliers clipped)"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")

    _save(fig, "variance_correlation")


def plot_variance_informed_ensembling(metrics: dict):
    """Figure 7: Pareto frontier for variance-informed ensembling (accuracy vs cost)."""
    vie_data = None
    for key, m in metrics.items():
        if m["type"] == "escalation" and "variance_informed_ensembling" in m:
            vie_data = m["variance_informed_ensembling"]
            break

    if vie_data is None or not vie_data.get("grid_results"):
        print("  Skipping variance-informed ensembling figure: no data")
        return

    grid = vie_data["grid_results"]
    best = vie_data["best"]
    by_n2 = vie_data["by_n2"]
    bc = vie_data.get("best_cost_constrained")

    # Convert mean_n2 to cost ratio (relative to always n2=8)
    grid_costs = [g["effective_cost_ratio"] for g in grid]
    grid_accs = [g["accuracy"] for g in grid]

    # Pareto frontier
    sorted_pts = sorted(zip(grid_costs, grid_accs))
    frontier = []
    best_acc = -1
    for c, a in sorted_pts:
        if a > best_acc:
            frontier.append((c, a))
            best_acc = a

    # Fixed-k reference line (cost ratio = n2 / n_max where n_max=8)
    n_max = max(e["n2"] for e in by_n2)
    ref_costs = [e["n2"] / n_max for e in by_n2]
    ref_accs = [e["accuracy"] for e in by_n2]

    always_full_acc = (
        vie_data.get("reference", {}).get("always_full", {}).get("overall")
    )
    always_mini_acc = (
        vie_data.get("reference", {}).get("always_mini", {}).get("overall")
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        grid_costs,
        grid_accs,
        alpha=0.15,
        s=12,
        c="gray",
        zorder=2,
        label=f"Grid search (n={len(grid)})",
    )

    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k-", linewidth=2, zorder=4, label="Pareto frontier")

    ax.plot(
        ref_costs,
        ref_accs,
        "b-o",
        linewidth=2,
        markersize=6,
        zorder=5,
        label="Fixed k (full model only)",
    )

    if best:
        ax.scatter(
            [best["effective_cost_ratio"]],
            [best["accuracy"]],
            s=120,
            c="red",
            marker="*",
            zorder=6,
            label=f"Best ({best['accuracy']:.1%})",
        )
    if bc:
        ax.scatter(
            [bc["effective_cost_ratio"]],
            [bc["accuracy"]],
            s=120,
            c="green",
            marker="*",
            zorder=6,
            label=f"Budget-constrained ({bc['accuracy']:.1%})",
        )

    if always_full_acc:
        ax.axhline(
            y=always_full_acc,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"Full model k=8 ({always_full_acc:.1%})",
        )
    if always_mini_acc:
        ax.axhline(
            y=always_mini_acc,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"Mini model k=8 ({always_mini_acc:.1%})",
        )

    ax.set_xlabel("Cost ratio (relative to full model k=8)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Variance-Informed Ensembling: Accuracy vs Cost")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save(fig, "variance_informed_ensembling")


def main():
    print("Loading metrics...")
    metrics = _load_metrics()

    print("\nGenerating figures...")

    print("\nFigure 1: Hero Accuracy")
    plot_hero_accuracy(metrics)

    print("\nFigure 2: Pareto Frontier")
    plot_pareto_frontier(metrics)

    print("\nFigure 3b: Per-Response Escalation Pareto")
    plot_per_response_escalation_pareto(metrics)

    print("\nFigure 3c: Soft Blending")
    plot_soft_blending(metrics)

    print("\nFigure 4: Variance Error Signal")
    plot_variance_error_signal(metrics)

    print("\nFigure 5: Mini-Full Convergence")
    plot_mini_full_convergence(metrics)

    print("\nFigure 6: Diminishing Returns")
    plot_diminishing_returns(metrics)

    print("\nFigure 7: Variance-Informed Ensembling")
    plot_variance_informed_ensembling(metrics)

    print("\nFigure 8: Variance Correlation (mini vs full)")
    plot_variance_correlation(metrics)

    print(f"\nDone! Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()

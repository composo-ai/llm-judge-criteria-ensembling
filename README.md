# Practical Techniques for Improving LLM-as-Judge Accuracy on RewardBench 2

**Author:** Ryan Lail<br>
**Affiliation:** Composo AI

Ablation study testing practical techniques to improve LLM-as-judge accuracy on [RewardBench 2](https://huggingface.co/datasets/allenai/reward-bench-2) (ratings mode). See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for full methodology, mathematical derivations, and analysis.

## Setup

**Requirements**: Python 3.13, Azure OpenAI access.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your Azure OpenAI credentials in .env
```

> **Note:** `run_all.sh` calls bare `python`, so make sure the venv is activated (or prepend the venv to your PATH: `PATH=".venv/bin:$PATH" bash run_all.sh`).

The baseline, ensemble, criteria, and calibration conditions need only the `AZURE_OPENAI_*` vars (full model). The escalation and combined conditions additionally require `AZURE_OPENAI_MINI_*` (a separate mini deployment).

## Quick Start

```bash
# Run all experiments (resume-safe, ~2-3 hours total)
python run_baselines.py
python run_ensemble.py
python run_criteria.py
python run_calibration.py
python run_escalation.py
python run_combined.py

# Compute metrics and generate figures
python analysis/compute_metrics.py
python analysis/figures.py
```

## Results

| Condition | N | Overall | Factuality | Focus | Math | Precise IF | Safety | $/example | vs Baseline |
|-----------|---|---------|------------|-------|------|------------|--------|-----------|-------------|
| Baseline | 1753 | 72.1% | 77.3% | 71.3% | 64.5% | 26.4% | 87.1% | $0.0134 | 1.0× |
| Criteria | 1747 | 75.3% | 79.8% | 73.5% | 70.5% | 34.4% | 89.7% | $0.0140 | 1.0× |
| Calibration (high) | 1614 | 73.0% | 78.4% | 67.7% | 67.8% | 36.9% | 88.2% | $0.0206 | 1.5× |
| Calibration (low) | 1545 | 74.2% | 77.9% | 68.3% | 69.9% | 33.1% | 91.2% | $0.0211 | 1.6× |
| Calibration (both) | 1516 | 74.1% | 77.7% | 71.2% | 67.1% | 32.7% | 88.6% | $0.0285 | 2.1× |
| Calibration (cross-category) | 1614 | 73.0% | 79.1% | 70.8% | 66.3% | 32.9% | 87.4% | $0.0209 | 1.6× |
| Ensemble k=8 | 1754 | 80.8% | 86.1% | 80.6% | 76.5% | 40.9% | 91.6% | $0.0667 | 5.0× |
| Mini model k=8 † | 1706 | 79.0% | 83.4% | 79.6% | 68.4% | 39.3% | 91.6% | $0.0051 | 0.4× |
| Soft blend (best) † | 1721 | 83.7% | **89.1%** | 83.3% | 78.0% | 50.7% | 92.0% | $0.0715 | 5.4× |
| Var-informed (≤2 calls) † | 1702 | 75.4% | 79.6% | 73.0% | 67.1% | 38.0% | 90.5% | $0.0216 | 1.6× |
| Combined | 1698 | 83.0% | 86.9% | 82.1% | 76.5% | 50.7% | 93.8% | $0.0773 | 5.8× |
| **Combined + soft blend** † | 1698 | **84.9%** | 88.2% | **85.3%** | **79.4%** | **51.3%** | **94.8%** | $0.0773 | 5.8× |

## File Structure

```
experiments/llm-judge-ablations/
├── judge.py                    # Shared judge logic: prompts, scoring, retry
├── run_baselines.py            # Condition 0: vanilla baseline
├── run_ensemble.py             # Condition 1: ensemble (n=k completions)
├── run_escalation.py           # Condition 2: mini + full model scoring
├── run_criteria.py             # Condition 3: task-specific criteria
├── run_calibration.py          # Condition 4: calibration context
├── run_combined.py             # Combined: criteria + calibration + escalation
├── run_all.sh                  # Run all conditions sequentially
├── analysis/
│   ├── compute_metrics.py      # All metric computation + CLI
│   └── figures.py              # All figure generation
├── results/
│   ├── raw/                    # JSONL results (gitignored)
│   └── tables/
│       └── all_metrics.json    # Computed metrics
└── figures/                    # Generated PNG figures
```

## Infrastructure Notes

- All experiment scripts write results **incrementally** and support **resume** — if interrupted, restart the same command and it skips completed examples.
- Run experiments in `tmux` to survive SSH disconnections.
- The escalation and combined experiments collect both mini and full scores for every example, enabling offline analysis of escalation strategies without re-running API calls.

## Citation

```bibtex
@misc{lail2025llmjudge,
  title={Practical Techniques for Improving LLM-as-Judge Accuracy on RewardBench 2},
  author={Ryan Lail},
  year={2026},
  url={https://github.com/composo-ai/llm-judge-ablations}
}
```

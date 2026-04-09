# A Systematic Evaluation of LLM-as-Judge Improvement Techniques on RewardBench 2

**Author:** Ryan Lail<br>
**Affiliation:** Composo AI

We systematically tested five techniques for improving LLM judge accuracy on [RewardBench 2](https://huggingface.co/datasets/allenai/reward-bench-2) and found that three simple, drop-in changes improve accuracy from 71.7% to 83.6%. No fine-tuning required.

**Blog post:** [Improving LLM Judges With Experiments, Not Vibes](https://www.composo.ai/post/llm-judge-criteria-ensembling/) | **Full methodology:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

## What Works

1. **Ask more than once.** LLM judges give different scores on every call. Request k=8 independent scores and average them — the noise cancels out. **+9.8pp at 5× cost.** Most of the gain comes by k=3.

2. **Try mini models.** GPT-5.4 mini with k=8 achieves 79.2% at **1.2× baseline cost** — at one-quarter the full ensemble's cost. Add criteria and it hits 81.5%.

3. **Be specific.** The standard judge prompt asks for generic qualities like "helpfulness, relevance, accuracy." Add a single sentence specifying what actually matters for each task. **+3.0pp at near-zero cost.** Criteria were pre-registered — no post-hoc tuning.

Combined, criteria + ensembling reach **83.6%** accuracy at 5.3× baseline cost.

## Setup

**Requirements**: Python 3.13, Azure OpenAI access (GPT-5.4, GPT-5.4 mini, and optionally GPT-5.4 nano deployments).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your Azure OpenAI credentials in .env
```

## Data Collection

All experiments use a unified collection script. Each collection gathers k=8 scores from both mini and full models, enabling offline derivation of many experimental conditions from a single API run.

```bash
# Core collections (each runs the full RB2 dataset with both models, k=8)
python collect.py --prompt base --models both --k 8        # Vanilla RB2 prompt
python collect.py --prompt criteria --models both --k 8    # + task-specific criteria
python collect.py --prompt cal-low --models both --k 8     # + calibration (low anchors)
python collect.py --prompt cal-high --models both --k 8    # + calibration (high anchors)
python collect.py --prompt cal-both --models both --k 8    # + calibration (both anchors)
python collect.py --prompt cal-cross --models both --k 8   # + calibration (cross-category)
python collect.py --prompt combined --models both --k 8    # + criteria + calibration

# Nano model (base prompt only — merged with base_both by ID in analysis)
python collect.py --prompt base --models nano --k 8

# Temperature sweep (base prompt, full model only)
python collect.py --prompt base --models full --k 8 --temperature 0.0
python collect.py --prompt base --models full --k 8 --temperature 0.3
python collect.py --prompt base --models full --k 8 --temperature 0.7

# Or run everything:
bash run_all.sh
```

All collections write incrementally and support **resume** — restart after interruption and completed examples are skipped.

## Analysis

```bash
python -m analysis.compute_metrics
python -m analysis.figures
```

## Results

All accuracy deltas in percentage points (pp). 95% bootstrap CIs shown. Conditions marked ‡ report test-set accuracy (20% held-out).

**Recommended techniques:**

| Condition | N | Overall (95% CI) | $/example | vs Baseline |
|-----------|---|------------------|-----------|-------------|
| Baseline (full k=1) | 1729 | 71.7% (±2.0pp) | $0.0133 | 1.0× |
| Criteria (full k=1) | 1738 | 74.7% (±1.9pp) | $0.0140 | 1.1× |
| Ensemble (full k=8) | 1730 | 81.5% (±1.8pp) | $0.0663 | 5.0× |
| **Criteria (full k=8)** | 1741 | **83.6%** (±1.6pp) | $0.0702 | 5.3× |
| Mini model k=8 | 1730 | 79.2% (±1.9pp) | $0.0154 | 1.2× |
| Criteria (mini k=8) | 1742 | 81.5% (±2.0pp) | $0.0160 | 1.2× |
| Nano model k=8 | 1705 | 71.4% (±2.0pp) | $0.0057 | 0.4× |
| Nano model k=1 | 1700 | 52.3% (±2.4pp) | $0.0011 | 0.1× |

**Investigated techniques (did not improve on criteria k=8):**

| Condition | N | Overall (95% CI) | $/example | vs Baseline |
|-----------|---|------------------|-----------|-------------|
| Calibration low (k=1) | 1737 | 73.8% (±2.0pp) | $0.0198 | 1.5× |
| Calibration low (k=8) | 1737 | 81.7% (±1.7pp) | $0.0744 | 5.6× |
| Combined (full k=8) | 1746 | 82.6% (±1.6pp) | $0.0913 | 6.8× |
| Combined + blend (test) ‡ | ~349 | 84.8% | $0.0913 | 6.8× |

> **‡** Blend parameters optimised on 80% train split, accuracy on held-out 20%. See report for caveats.

![Hero Accuracy](figures/hero_accuracy.png)

## File Structure

```
├── judge.py                    # Shared judge logic: prompts, scoring, retry
├── collect.py                  # Unified data collection
├── run_all.sh                  # Run all collections sequentially
├── requirements.txt            # Python dependencies
├── .env.example                # Template for Azure OpenAI credentials
├── TECHNICAL_REPORT.md         # Full methodology and analysis
├── LICENSE
├── analysis/
│   ├── compute_metrics.py      # Derive conditions, compute metrics + CIs
│   └── figures.py              # Generate all figures
├── results/
│   ├── raw/                    # JSONL collections per prompt variant (gitignored)
│   └── tables/
│       └── all_metrics.json    # Computed metrics (included in repo)
└── figures/                    # Pre-generated PNG figures (included in repo)
```

## Experimental Design

Each `collect.py` invocation collects k=8 scores from both mini and full models for every example. From this data, all conditions are derived offline:

- **Baseline** = base collection, full model, subsample k=1
- **Ensemble k=N** = base collection, full model, subsample k=N
- **Criteria** = criteria collection, full model
- **Calibration** = cal-* collection, full model
- **Combined** = combined collection, all techniques

This design minimises API calls while maximising the number of conditions that can be analysed.

## Infrastructure Notes

- All collections write results **incrementally** (JSONL, flushed per example) and support **resume**.
- Run in `tmux` to survive SSH disconnections.
- By default, all examples are collected. Use `--sample-size N` to cap examples per subset for quick tests.

## Citation

```bibtex
@misc{lail2026llmjudge,
  title={Criteria Injection and Ensembling Are All You Need: A Systematic Evaluation of LLM Judge Techniques on RewardBench 2},
  author={Ryan Lail},
  year={2026},
  url={https://github.com/composo-ai/llm-judge-criteria-ensembling}
}
```

#!/bin/bash
set -e

echo "============================================"
echo "RB2 Judge Experiments — Data Collection"
echo "============================================"

# -----------------------------------------------
# Core collections (mini + full, k=8, temp=1.0)
# Each collection enables many offline experiments
# -----------------------------------------------

echo ""
echo ">>> Collection 1/7: Base prompt (vanilla RB2)"
python collect.py --prompt base --models both --k 8

echo ""
echo ">>> Collection 2/7: Criteria prompt"
python collect.py --prompt criteria --models both --k 8

echo ""
echo ">>> Collection 3/7: Calibration (low)"
python collect.py --prompt cal-low --models both --k 8

echo ""
echo ">>> Collection 4/7: Calibration (high)"
python collect.py --prompt cal-high --models both --k 8

echo ""
echo ">>> Collection 5/7: Calibration (both)"
python collect.py --prompt cal-both --models both --k 8

echo ""
echo ">>> Collection 6/7: Calibration (cross-category)"
python collect.py --prompt cal-cross --models both --k 8

echo ""
echo ">>> Collection 7/7: Combined (criteria + cal-low)"
python collect.py --prompt combined --models both --k 8

# -----------------------------------------------
# Nano model (base prompt only, k=8)
# Merged with base_both by ID in analysis
# -----------------------------------------------

echo ""
echo ">>> Nano model: Base prompt"
python collect.py --prompt base --models nano --k 8

# -----------------------------------------------
# Temperature sweep (base prompt, full model, k=8)
# Subsampled to k=1 in analysis
# -----------------------------------------------

echo ""
echo ">>> Temperature sweep"
for TEMP in 0.0 0.3 0.7; do
    echo "  temp=$TEMP"
    python collect.py --prompt base --models full --k 8 --temperature "$TEMP"
done

echo ""
echo "============================================"
echo "All collections complete!"
echo "Results in results/raw/"
ls -lh results/raw/*.jsonl
echo ""
echo "Next: python analysis/compute_metrics.py && python analysis/figures.py"
echo "============================================"

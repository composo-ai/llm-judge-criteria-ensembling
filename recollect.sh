#!/bin/bash
set -e

echo "============================================"
echo "Full re-collection + analysis"
echo "============================================"
echo ""
echo "This will:"
echo "  1. Delete all old data in results/raw/"
echo "  2. Collect 7 prompt variants (both models, k=8)"
echo "  3. Collect temperature sweep (full model, k=8)"
echo "  4. Run analysis + generate figures"
echo ""
read -p "Press Enter to start (Ctrl+C to abort)..."

# -----------------------------------------------
# 0. Clear old data
# -----------------------------------------------
echo ""
echo ">>> Clearing old data..."
rm -f results/raw/*.jsonl
echo "  Done."

# -----------------------------------------------
# 1. Core collections (mini + full, k=8)
# -----------------------------------------------

echo ""
echo ">>> [1/7] Base prompt"
python collect.py --prompt base --models both --k 8

echo ""
echo ">>> [2/7] Criteria prompt"
python collect.py --prompt criteria --models both --k 8

echo ""
echo ">>> [3/7] Calibration (low)"
python collect.py --prompt cal-low --models both --k 8

echo ""
echo ">>> [4/7] Calibration (high)"
python collect.py --prompt cal-high --models both --k 8

echo ""
echo ">>> [5/7] Calibration (both)"
python collect.py --prompt cal-both --models both --k 8

echo ""
echo ">>> [6/7] Calibration (cross-category)"
python collect.py --prompt cal-cross --models both --k 8

echo ""
echo ">>> [7/7] Combined (criteria + cal-low)"
python collect.py --prompt combined --models both --k 8

# -----------------------------------------------
# 2. Temperature sweep (base prompt, full model only)
# -----------------------------------------------

echo ""
echo ">>> Temperature sweep"
for TEMP in 0.0 0.3 0.7; do
    echo ""
    echo "  >>> temp=$TEMP"
    python collect.py --prompt base --models full --k 8 --temperature "$TEMP"
done

# -----------------------------------------------
# 3. Analysis + figures
# -----------------------------------------------

echo ""
echo ">>> Running analysis..."
python -m analysis.compute_metrics

echo ""
echo ">>> Generating figures..."
python -m analysis.figures

# -----------------------------------------------
# Done
# -----------------------------------------------

echo ""
echo "============================================"
echo "All done!"
echo "============================================"
echo ""
ls -lh results/raw/*.jsonl
echo ""
echo "Metrics: results/tables/all_metrics.json"
echo "Figures: figures/*.png"

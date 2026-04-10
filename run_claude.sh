#!/bin/bash
set -e

echo "=============================================="
echo "Claude Generalisability Experiments"
echo "  Full:  Claude Sonnet 4.6"
echo "  Mini:  Claude Haiku 4.5"
echo "=============================================="

echo ""
echo ">>> Condition 1/2: Base prompt (k=8, both models)"
python collect_claude.py --prompt base --k 8

echo ""
echo ">>> Condition 2/2: Criteria prompt (k=8, both models)"
python collect_claude.py --prompt criteria --k 8

echo ""
echo ">>> Computing metrics..."
python analysis/compute_metrics.py

echo ""
echo "Done. Results in results/tables/all_metrics.json"

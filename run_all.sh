#!/bin/bash
set -e

SAMPLE_SIZE=${1:-50}

echo "============================================"
echo "Running all RB2 experiments (sample_size=$SAMPLE_SIZE)"
echo "============================================"

echo ""
echo ">>> Condition 0: Baseline"
python run_baselines.py --sample-size "$SAMPLE_SIZE"

echo ""
echo ">>> Condition 1: Ensemble (k=8)"
python run_ensemble.py --sample-size "$SAMPLE_SIZE"

echo ""
echo ">>> Condition 2: Adaptive Escalation (mini → full)"
python run_escalation.py --sample-size "$SAMPLE_SIZE"

echo ""
echo ">>> Condition 3: Task-Specific Criteria"
python run_criteria.py --sample-size "$SAMPLE_SIZE"

echo ""
echo ">>> Condition 4A: Calibration (high)"
python run_calibration.py --cal-variant high --sample-size "$SAMPLE_SIZE"

echo ""
echo ">>> Condition 4B: Calibration (low)"
python run_calibration.py --cal-variant low --sample-size "$SAMPLE_SIZE"

echo ""
echo ">>> Condition 4C: Calibration (both)"
python run_calibration.py --cal-variant both --sample-size "$SAMPLE_SIZE"

echo ""
echo ">>> Condition 4D: Calibration (cross-category)"
python run_calibration.py --cal-variant cross-category --sample-size "$SAMPLE_SIZE"

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Results in results/raw/"
ls -lh results/raw/*.jsonl
echo "============================================"

#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_baseline.sh
bash scripts/run_opt_tree.sh
bash scripts/run_ddd.sh
bash scripts/run_combo.sh
python scripts/collect_results.py results/*.jsonl --output results/summary.csv

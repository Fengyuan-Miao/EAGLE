#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
EA_MODEL_PATH="${EA_MODEL_PATH:-/home/miaofy/models/Qwen3-1.7B_eagle3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-float16}"
OPT_TREE_BUDGET="${OPT_TREE_BUDGET:-32}"
OPT_TREE_DEPTH="${OPT_TREE_DEPTH:-3}"

python scripts/benchmark.py \
  --decode-mode eagle \
  --spec-mode opt_tree \
  --base-model-path "$BASE_MODEL_PATH" \
  --ea-model-path "$EA_MODEL_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --depth "$OPT_TREE_DEPTH" \
  --enable-profile \
  --profile-output results/profile_opt_tree_b${OPT_TREE_BUDGET}.jsonl \
  --opt-tree-budget "$OPT_TREE_BUDGET" \
  --opt-tree-overexpand-factor 1.0 \
  --output results/opt_tree_b${OPT_TREE_BUDGET}.jsonl

#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
EA_MODEL_PATH="${EA_MODEL_PATH:-/home/miaofy/models/Qwen3-1.7B_eagle3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-float16}"
DDD_MAX_DEPTH="${DDD_MAX_DEPTH:-4}"

python scripts/benchmark.py \
  --decode-mode eagle \
  --spec-mode ddd \
  --base-model-path "$BASE_MODEL_PATH" \
  --ea-model-path "$EA_MODEL_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --enable-profile \
  --profile-output results/profile_ddd_medium.jsonl \
  --ddd-max-depth "$DDD_MAX_DEPTH" \
  --ddd-checkpoints 3,5,7 \
  --ddd-thresholds=-100,-100,-100 \
  --output results/ddd_medium.jsonl

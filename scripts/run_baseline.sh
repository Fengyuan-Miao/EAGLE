#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
EA_MODEL_PATH="${EA_MODEL_PATH:-/home/miaofy/models/Qwen3-1.7B_eagle3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-float16}"

python scripts/benchmark.py \
  --decode-mode vanilla_ar \
  --base-model-path "$BASE_MODEL_PATH" \
  --ea-model-path "$EA_MODEL_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --output results/vanilla_ar.jsonl

python scripts/benchmark.py \
  --decode-mode eagle \
  --spec-mode baseline \
  --base-model-path "$BASE_MODEL_PATH" \
  --ea-model-path "$EA_MODEL_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --enable-profile \
  --profile-output results/profile_baseline.jsonl \
  --output results/eagle_baseline.jsonl

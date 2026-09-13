#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/c/Users/bjw-0/Downloads/ControlFlow-G-v2"
lock_dir="$repo/state/v21_gpu.lock.d"
env_dir="$HOME/.venvs/controlflow-g-v2"
model="Qwen/Qwen3-4B-Instruct-2507"
revision="cdbee75f17c01a7cc42f958dc650907174af0554"
export VLLM_USE_V2_MODEL_RUNNER=0

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "V2.1 GPU semaphore is already held: $lock_dir" >&2
  exit 73
fi
cleanup() { rmdir "$lock_dir"; }
trap cleanup EXIT INT TERM

"$env_dir/bin/vllm" serve "$model" \
  --revision "$revision" \
  --served-model-name controlflow-g-v21-qwen3-4b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --structured-outputs-config.backend xgrammar \
  --host 127.0.0.1 \
  --port 8001

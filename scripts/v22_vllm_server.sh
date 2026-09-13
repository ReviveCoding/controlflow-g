#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/c/Users/bjw-0/Downloads/ControlFlow-G-v2"
lock_dir="$repo/state/gpu.lock"
env_dir="$HOME/.venvs/controlflow-g-v2"
model="Qwen/Qwen3-4B-Instruct-2507"
revision="cdbee75f17c01a7cc42f958dc650907174af0554"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU has an existing compute consumer; refusing to launch or kill it" >&2
  exit 74
fi
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "repository GPU semaphore is already held: $lock_dir" >&2
  exit 73
fi
cleanup() { rmdir "$lock_dir"; }
trap cleanup EXIT INT TERM

export VLLM_USE_V2_MODEL_RUNNER=0
"$env_dir/bin/vllm" serve "$model" \
  --revision "$revision" \
  --served-model-name controlflow-g-v22-qwen3-4b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.72 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --structured-outputs-config.backend xgrammar \
  --host 127.0.0.1 \
  --port 8022

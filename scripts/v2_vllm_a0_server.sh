#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/c/Users/bjw-0/Downloads/ControlFlow-G-v2"
lock_dir="$repo/state/v2_gpu.lock.d"
env_dir="$HOME/.venvs/controlflow-g-v2"
model="Qwen/Qwen2.5-0.5B-Instruct"
revision="7ae557604adf67be50417f59c2c2f167def9a775"

export VLLM_USE_V2_MODEL_RUNNER=0

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "V2 GPU semaphore is already held: $lock_dir" >&2
  exit 73
fi
cleanup() {
  rmdir "$lock_dir"
}
trap cleanup EXIT INT TERM

"$env_dir/bin/vllm" serve "$model" \
  --revision "$revision" \
  --served-model-name controlflow-g-v2-a0 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --host 127.0.0.1 \
  --port 8000

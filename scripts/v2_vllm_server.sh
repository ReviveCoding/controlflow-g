#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/c/Users/bjw-0/Downloads/ControlFlow-G-v2"
lock_dir="$repo/state/v2_gpu.lock.d"
env_dir="$HOME/.venvs/controlflow-g-v2"
model="Qwen/Qwen3-4B-Instruct-2507"
revision="cdbee75f17c01a7cc42f958dc650907174af0554"

# vLLM's V2 runner requires UVA buffers that WSL2 does not currently expose.
# The supported V1 runner preserves GPU execution and structured decoding.
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
  --served-model-name controlflow-g-v2-qwen3-4b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --host 127.0.0.1 \
  --port 8000

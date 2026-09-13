#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/c/Users/bjw-0/Downloads/ControlFlow-G-v2"
lock_dir="$repo/state/gpu.lock"
env_dir="$HOME/.venvs/controlflow-g-v2"
model="Qwen/Qwen3-4B-Instruct-2507"
revision="cdbee75f17c01a7cc42f958dc650907174af0554"

performance_mode="${V23_PERFORMANCE_MODE:-balanced}"
batched_tokens="${V23_BATCHED_TOKENS:-2048}"
optimization_level="${V23_OPTIMIZATION_LEVEL:-2}"
prefix_caching="${V23_PREFIX_CACHING:-1}"
port="${V23_PORT:-8023}"
run_role="${V23_RUN_ROLE:-development}"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU has an existing compute consumer; refusing to launch or kill it" >&2
  exit 74
fi
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "repository GPU semaphore is already held: $lock_dir" >&2
  exit 73
fi
cleanup() { rmdir "$lock_dir" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

prefix_args=(--no-enable-prefix-caching)
if [[ "$prefix_caching" == "1" ]]; then prefix_args=(--enable-prefix-caching); fi
batched_args=()
if [[ "$batched_tokens" != "default" ]]; then batched_args=(--max-num-batched-tokens "$batched_tokens"); fi

export VLLM_USE_V2_MODEL_RUNNER=0
export V23_RUN_ROLE="$run_role"
"$env_dir/bin/vllm" serve "$model" \
  --revision "$revision" \
  --served-model-name controlflow-g-v23-qwen3-4b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.72 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --structured-outputs-config.backend xgrammar \
  --enable-chunked-prefill \
  "${prefix_args[@]}" \
  --enable-per-request-metrics \
  --enable-request-id-headers \
  --performance-mode "$performance_mode" \
  --optimization-level "$optimization_level" \
  "${batched_args[@]}" \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port "$port"

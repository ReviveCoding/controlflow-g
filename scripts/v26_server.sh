#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/c/Users/bjw-0/Downloads/ControlFlow-G-v2"
lock_dir="$repo/state/gpu.lock"
env_dir="$HOME/.venvs/controlflow-g-v2"
model="Qwen/Qwen3-4B-Instruct-2507"
revision="cdbee75f17c01a7cc42f958dc650907174af0554"

if [[ "${V26_RUN_ROLE:-}" != "development" && "${V26_RUN_ROLE:-}" != "qualification" && "${V26_RUN_ROLE:-}" != "final" ]]; then
  echo "V26 run role must be development, qualification, or final" >&2
  exit 72
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "Unrelated GPU compute consumer present" >&2
  exit 74
fi
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "Repository GPU semaphore already held" >&2
  exit 73
fi
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -f "$lock_dir/owner.pid"
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export VLLM_USE_V2_MODEL_RUNNER=0
export V23_RUN_ROLE="$V26_RUN_ROLE"
"$env_dir/bin/vllm" serve "$model" \
  --revision "$revision" \
  --served-model-name controlflow-g-v23-qwen3-4b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.72 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --structured-outputs-config.backend xgrammar \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-per-request-metrics \
  --enable-request-id-headers \
  --performance-mode interactivity \
  --optimization-level 2 \
  --max-num-batched-tokens 2048 \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port 8023 &
server_pid=$!
printf '%s\n' "$server_pid" > "$lock_dir/owner.pid"
wait "$server_pid"

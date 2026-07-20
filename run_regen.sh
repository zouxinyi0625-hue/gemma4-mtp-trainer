#!/usr/bin/env bash
# One-command regen: launch 8 single-GPU vLLM servers (TP=1, FP8, MTP on) and
# regenerate 26B answers across all of them in parallel, then clean up.
#
# WHY: each server owns one GPU (TP=1), so 8 GPUs = 8 independent OpenAI
# endpoints. regen.py round-robins requests across them -> ~8x throughput.
# MTP (speculative decoding) is enabled purely to speed up generation; it does
# NOT change the sampled answers, only how fast they're produced.
#
# USAGE (on the 8-GPU server, in the vLLM venv):
#   INPUT=/path/to/dspark_prompts.jsonl \
#   OUTPUT=/path/to/regen_26b.jsonl \
#   bash run_regen.sh
#
# Override anything via env: TARGET, ASSISTANT, NGPU, BASE_PORT, CONCURRENCY,
# TEMPERATURE, TOP_P, MAX_TOKENS, MAX_MODEL_LEN, GPU_UTIL.
set -euo pipefail

# ---- config (override via env) --------------------------------------------
TARGET="${TARGET:-google/gemma-4-26B-A4B-it-text-only}"
ASSISTANT="${ASSISTANT:-google/gemma-4-26B-A4B-it-assistant}"  # MTP draft
TOKENIZER="${TOKENIZER:-google/gemma-4-26B-A4B-it}"
SERVED_NAME="${SERVED_NAME:-gemma4}"

INPUT="${INPUT:?set INPUT=/path/to/dspark_prompts.jsonl}"
OUTPUT="${OUTPUT:?set OUTPUT=/path/to/regen_26b.jsonl}"

NGPU="${NGPU:-8}"
BASE_PORT="${BASE_PORT:-8100}"
CONCURRENCY="${CONCURRENCY:-64}"          # in-flight requests PER server
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
GPU_UTIL="${GPU_UTIL:-0.9}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"           # MTP k
READY_TIMEOUT="${READY_TIMEOUT:-900}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-regen_server_logs}"
SOURCE_LAYER="${SOURCE_LAYER:-}"          # optional tag for build_split stats

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

mkdir -p "$SERVER_LOG_DIR"

echo "=== regen: $NGPU single-GPU servers (TP1, FP8, MTP k=$SPEC_TOKENS) ==="
echo "  target=$TARGET"
echo "  assistant(MTP)=$ASSISTANT"
echo "  input=$INPUT  output=$OUTPUT"
echo "  ports=$BASE_PORT..$((BASE_PORT+NGPU-1))  concurrency=$CONCURRENCY/server"
echo ""

# ---- pre-clean stale servers/ports ----------------------------------------
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

PIDS=()
SERVERS=()

cleanup() {
  echo ""
  echo "Stopping $NGPU servers..."
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 5
  pkill -9 -f "vllm serve" 2>/dev/null || true
}
trap cleanup EXIT

# ---- launch one server per GPU --------------------------------------------
for ((i=0; i<NGPU; i++)); do
  port=$((BASE_PORT + i))
  log="$SERVER_LOG_DIR/regen_server_gpu${i}_port${port}.log"
  echo "  [gpu $i] starting vllm serve on port $port -> $log"
  CUDA_VISIBLE_DEVICES="$i" vllm serve "$TARGET" \
    --served-model-name "$SERVED_NAME" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --dtype auto \
    --quantization fp8 \
    --kv-cache-dtype auto \
    --tokenizer "$TOKENIZER" \
    --trust-remote-code \
    --spec-model "$ASSISTANT" --spec-tokens "$SPEC_TOKENS" \
    --no-enable-log-requests \
    >"$log" 2>&1 &
  PIDS+=($!)
  SERVERS+=("http://localhost:${port}/v1")
done

# ---- wait for readiness ----------------------------------------------------
echo ""
echo "Waiting for $NGPU servers to be ready (timeout ${READY_TIMEOUT}s each)..."
for ((i=0; i<NGPU; i++)); do
  port=$((BASE_PORT + i))
  url="http://localhost:${port}/v1/models"
  ok=0
  for ((s=0; s<READY_TIMEOUT; s++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "  [gpu $i] ready after ${s}s"
      ok=1; break
    fi
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "[FATAL] gpu $i server exited before readiness. Last 40 log lines:" >&2
      tail -40 "$SERVER_LOG_DIR/regen_server_gpu${i}_port${port}.log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "[FATAL] gpu $i not ready after ${READY_TIMEOUT}s" >&2
    exit 1
  fi
done

# ---- run regen across all servers -----------------------------------------
echo ""
echo "=== all servers ready; running regen ==="
SRC_ARG=()
[[ -n "$SOURCE_LAYER" ]] && SRC_ARG=(--source-layer "$SOURCE_LAYER")

python -m gemma4_mtp.regen \
  --model "$SERVED_NAME" \
  --server "${SERVERS[@]}" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --concurrency "$CONCURRENCY" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --resume \
  "${SRC_ARG[@]}"

echo ""
echo "=== regen done -> $OUTPUT ==="

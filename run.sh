#!/usr/bin/env bash
# Full-scale target-cache generation + training, 8-GPU.
#
# Two stages:
#   1) prepare_cache: run the frozen 26B target once over all samples, stream
#      last_hidden + shared KV to a sharded on-mount cache (async writer, large
#      shards, no per-sample .pt).
#   2) train: read the cache (no 26B loaded) and fine-tune the assistant on 8
#      GPUs, export a stock-config checkpoint for vLLM.
#
# Usage:
#   bash run.sh              # both stages
#   STAGE=cache bash run.sh  # only generate the cache
#   STAGE=train bash run.sh  # only train (cache must already exist)
#
# Override any path via env, e.g.:
#   OUT_DIR=/my/cache DATA=/my/data.jsonl bash run.sh
set -euo pipefail

# ---- config (override via env) --------------------------------------------
NPROC="${NPROC:-8}"
TARGET="${TARGET:-/tmp/models/gemma4/text_only}"
ASSISTANT="${ASSISTANT:-/tmp/models/gemma4/assistant}"
DATA="${DATA:-./data/mtp_short/train_maiprofile_short_26b.jsonl}"

# Cache lives on the mount (large; slow writes handled by the async writer).
MNT="${MNT:-$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile}"
DATE="${DATE:-$(date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-$MNT/mtp_cache/$DATE/short_train}"

CKPT_DIR="${CKPT_DIR:-./out/mtp_maiprofile_$DATE}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

# Training hyperparams.
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
TTT_STEPS="${TTT_STEPS:-5}"

STAGE="${STAGE:-all}"   # all | cache | train

# Quieten NCCL topology spam; only warn on real problems.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

filter() { grep -v "NCCL INFO" || true; }

echo "=== config ==="
echo "  NPROC      = $NPROC"
echo "  TARGET     = $TARGET"
echo "  ASSISTANT  = $ASSISTANT"
echo "  DATA       = $DATA"
echo "  OUT_DIR    = $OUT_DIR   (cache)"
echo "  CKPT_DIR   = $CKPT_DIR  (checkpoints)"
echo "  STAGE      = $STAGE"
echo ""

# ---- preflight ------------------------------------------------------------
if [[ ! -f "$DATA" ]]; then
  echo "!! DATA not found: $DATA"
  echo "   Generate it first with gemma4_mtp.build_split (see its docstring)."
  exit 1
fi

# ---- stage 1: prepare cache ----------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "cache" ]]; then
  if [[ -f "$OUT_DIR/manifest.json" ]]; then
    echo "=== [cache] SKIP: manifest already exists at $OUT_DIR ==="
    echo "    (delete the dir to regenerate: rm -rf $OUT_DIR)"
  else
    # prepare_cache requires an empty/non-existent OUT_DIR.
    if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]]; then
      echo "!! OUT_DIR exists and is not empty: $OUT_DIR"
      echo "   Use a fresh dir or: rm -rf $OUT_DIR"
      exit 1
    fi
    echo "=== [cache] generating -> $OUT_DIR ==="
    torchrun --standalone --nproc_per_node "$NPROC" -m gemma4_mtp.prepare_cache \
        --target "$TARGET" \
        --data "$DATA" \
        --out-dir "$OUT_DIR" \
        --max-length "$MAX_LENGTH" \
        --bf16 2>&1 | filter
    echo "=== [cache] done ==="
    ls -la "$OUT_DIR"
  fi
fi

# ---- stage 2: train -------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  if [[ ! -f "$OUT_DIR/manifest.json" ]]; then
    echo "!! no cache manifest at $OUT_DIR — run STAGE=cache first."
    exit 1
  fi
  echo "=== [train] 8-GPU from cache -> $CKPT_DIR ==="
  torchrun --standalone --nproc_per_node "$NPROC" -m gemma4_mtp.train \
      --cache-dir "$OUT_DIR" \
      --target "$TARGET" \
      --assistant "$ASSISTANT" \
      --output "$CKPT_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --grad-accum "$GRAD_ACCUM" \
      --lr "$LR" \
      --ttt-steps "$TTT_STEPS" \
      --max-length "$MAX_LENGTH" \
      --bf16 --log-every 10 2>&1 | filter
  echo "=== [train] done. checkpoint at $CKPT_DIR ==="
fi

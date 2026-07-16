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
# Both the cache and the checkpoints are written directly to the mount. The
# cache write is async (won't block the GPU). Checkpoints are large single-file
# writes; to avoid stalling training on the slow mount we save infrequently
# (only at the end by default; set SAVE_EVERY for periodic saves).
#
# Hyperparameters follow DeepSpec's dspark_gemma4_12b config
# (config/dspark/dspark_gemma4_12b.py): lr=6e-4, 10 epochs, global batch 512,
# warmup 4%, weight_decay 0, grad_clip 1.0.
#
# Usage:
#   bash run.sh              # both stages
#   STAGE=cache bash run.sh  # only generate the cache
#   STAGE=train bash run.sh  # only train (cache must already exist)
#
# Override any path/hyperparam via env, e.g.:
#   LR=2e-4 EPOCHS=2 bash run.sh
set -euo pipefail

# ---- paths (override via env) ---------------------------------------------
NPROC="${NPROC:-8}"
TARGET="${TARGET:-/tmp/models/gemma4/text_only}"
ASSISTANT="${ASSISTANT:-/tmp/models/gemma4/assistant}"
DATA="${DATA:-./data/mtp_short/train_maiprofile_short_26b.jsonl}"

# Everything persistent lives on the MSN.DnI mount.
MNT="${MNT:-$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile}"
# Timestamped run tag so each launch gets a fresh dir (avoids colliding with a
# previous run's leftover _tmp shards, which would trip the "not empty" guard).
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$MNT/mtp_cache/$RUN_TAG/short_train}"        # sharded cache
CKPT_DIR="${CKPT_DIR:-$MNT/checkpoints/mtp_maiprofile/$RUN_TAG}" # checkpoints (mount)

MAX_LENGTH="${MAX_LENGTH:-4096}"

# ---- hyperparams (aligned to dspark_gemma4_12b) ---------------------------
EPOCHS="${EPOCHS:-10}"
LOCAL_BATCH="${LOCAL_BATCH:-2}"           # per-GPU micro-batch
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"       # dspark global_batch_size
LR="${LR:-6e-4}"                          # dspark lr
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-}"          # if empty, computed as 4% of total steps
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"      # dspark warmup_ratio
TTT_STEPS="${TTT_STEPS:-5}"
SAVE_EVERY="${SAVE_EVERY:-0}"             # 0 = save only at end (avoid frequent mount writes)
LOG_EVERY="${LOG_EVERY:-10}"

# grad_accum so that local_batch * nproc * grad_accum == global_batch.
GRAD_ACCUM=$(( GLOBAL_BATCH / (LOCAL_BATCH * NPROC) ))
if (( GRAD_ACCUM < 1 )); then GRAD_ACCUM=1; fi

STAGE="${STAGE:-all}"   # all | cache | train
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
filter() { grep -v "NCCL INFO" || true; }

echo "=== config ==="
echo "  NPROC       = $NPROC"
echo "  TARGET      = $TARGET"
echo "  ASSISTANT   = $ASSISTANT"
echo "  DATA        = $DATA"
echo "  RUN_TAG     = $RUN_TAG   (set RUN_TAG=... to reuse a cache across stages)"
echo "  OUT_DIR     = $OUT_DIR   (cache, mount)"
echo "  CKPT_DIR    = $CKPT_DIR  (checkpoints, mount)"
echo "  hyperparams : lr=$LR epochs=$EPOCHS local_batch=$LOCAL_BATCH"
echo "                global_batch=$GLOBAL_BATCH -> grad_accum=$GRAD_ACCUM"
echo "                ttt_steps=$TTT_STEPS save_every=$SAVE_EVERY"
echo "  STAGE       = $STAGE"
echo ""

# ---- preflight ------------------------------------------------------------
# DATA (the raw jsonl) is only needed to BUILD the cache. STAGE=train reads the
# already-built cache, so don't require DATA there.
if [[ "$STAGE" == "all" || "$STAGE" == "cache" ]]; then
  if [[ ! -f "$DATA" ]]; then
    echo "!! DATA not found: $DATA"
    echo "   Generate it first with gemma4_mtp.build_split (see its docstring)."
    exit 1
  fi
fi

# ---- stage 1: prepare cache ----------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "cache" ]]; then
  if [[ -f "$OUT_DIR/manifest.json" ]]; then
    echo "=== [cache] SKIP: manifest already exists at $OUT_DIR ==="
    echo "    (delete the dir to regenerate: rm -rf $OUT_DIR)"
  else
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

  # Compute warmup steps from the ratio if not given explicitly.
  # total optim steps = ceil(num_samples / global_batch) * epochs.
  if [[ -z "$WARMUP_STEPS" ]]; then
    NUM_SAMPLES=$(python -c "import json,sys; print(json.load(open('$OUT_DIR/manifest.json'))['num_samples'])")
    WARMUP_STEPS=$(python -c "
import math
steps_per_epoch = max(1, math.ceil($NUM_SAMPLES / $GLOBAL_BATCH))
total = steps_per_epoch * $EPOCHS
print(max(1, round(total * $WARMUP_RATIO)))
")
    echo "  [train] num_samples=$NUM_SAMPLES -> warmup_steps=$WARMUP_STEPS (ratio=$WARMUP_RATIO)"
  fi

  SAVE_ARG=""
  if (( SAVE_EVERY > 0 )); then SAVE_ARG="--save-every $SAVE_EVERY"; fi

  mkdir -p "$CKPT_DIR"
  echo "=== [train] 8-GPU from cache -> $CKPT_DIR ==="
  torchrun --standalone --nproc_per_node "$NPROC" -m gemma4_mtp.train \
      --cache-dir "$OUT_DIR" \
      --target "$TARGET" \
      --assistant "$ASSISTANT" \
      --output "$CKPT_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$LOCAL_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --lr "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --warmup-steps "$WARMUP_STEPS" \
      --ttt-steps "$TTT_STEPS" \
      --max-length "$MAX_LENGTH" \
      $SAVE_ARG \
      --bf16 --log-every "$LOG_EVERY" 2>&1 | filter
  echo "=== [train] done. checkpoint at $CKPT_DIR ==="
fi

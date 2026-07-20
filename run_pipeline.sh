#!/usr/bin/env bash
# End-to-end pipeline for the EXPANDED MAI Profile data (regen -> split ->
# cache -> train), in one command.
#
# Prereq: regen has already produced per-layer answers (run_regen.sh) under
# REGEN_DIR. This script then:
#   1) split : random stratified train/eval split of the regen data
#   2) cache : run the frozen 26B target over the train split, store
#              last_hidden + shared-KV to a sharded cache
#   3) train : fine-tune the assistant (MTP draft) from that cache
#
# Each stage is skippable via STAGE (split|cache|train|all) and is a no-op if
# its output already exists, so you can resume.
#
# USAGE
#   REGEN_DIR=/tmp/regen \
#   DATA_DIR=/tmp/mtp_26b_data \
#   OUT_DIR=/tmp/mtp_cache_26b \
#   CKPT_DIR="$AZURE_ML_INPUT_ukwdata/maiprofile/checkpoints/mtp_26b/$(date +%Y%m%d_%H%M%S)" \
#   bash run_pipeline.sh
#
#   # only one stage:
#   STAGE=split ... bash run_pipeline.sh
set -euo pipefail

# ---- models --------------------------------------------------------------
NPROC="${NPROC:-8}"
TARGET="${TARGET:-/tmp/models/gemma4/text_only}"
ASSISTANT="${ASSISTANT:-/tmp/models/gemma4/assistant}"

# ---- data paths ----------------------------------------------------------
REGEN_DIR="${REGEN_DIR:-/tmp/regen}"          # per-layer *_regen.jsonl (from run_regen.sh)
DATA_DIR="${DATA_DIR:-/tmp/mtp_26b_data}"     # split train/eval land here (local, fast)
OUT_DIR="${OUT_DIR:-/tmp/mtp_cache_26b}"      # sharded target cache (local)
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR=/path/to/checkpoints (e.g. on the mount)}"
TRAIN_JSONL="$DATA_DIR/train_maiprofile_26b.jsonl"
EVAL_JSONL="$DATA_DIR/eval_maiprofile_26b.jsonl"

MAX_LENGTH="${MAX_LENGTH:-4096}"
EVAL_FRAC="${EVAL_FRAC:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-0}"

# ---- train hyperparams (same defaults as run.sh) -------------------------
EPOCHS="${EPOCHS:-3}"
LOCAL_BATCH="${LOCAL_BATCH:-2}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
TTT_STEPS="${TTT_STEPS:-5}"
SAVE_EVERY="${SAVE_EVERY:-0}"
LOG_EVERY="${LOG_EVERY:-1}"
NUM_ANCHORS="${NUM_ANCHORS:-128}"
ANCHOR_CHUNK="${ANCHOR_CHUNK:-8}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-4.0}"
ARGMAX_CE="${ARGMAX_CE:-1.0}"
SOFT_CE="${SOFT_CE:-0.0}"
L1_WEIGHT="${L1_WEIGHT:-0.0}"
HARD_CE="${HARD_CE:-0.0}"

GRAD_ACCUM=$(( GLOBAL_BATCH / (LOCAL_BATCH * NPROC) ))
if (( GRAD_ACCUM < 1 )); then GRAD_ACCUM=1; fi

STAGE="${STAGE:-all}"   # all | split | cache | train
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
filter() { grep -v "NCCL INFO" || true; }

echo "=== pipeline config ==="
echo "  REGEN_DIR = $REGEN_DIR"
echo "  DATA_DIR  = $DATA_DIR   (train/eval split)"
echo "  OUT_DIR   = $OUT_DIR    (target cache)"
echo "  CKPT_DIR  = $CKPT_DIR   (checkpoints)"
echo "  eval_frac=$EVAL_FRAC seed=$SPLIT_SEED  epochs=$EPOCHS lr=$LR"
echo "  global_batch=$GLOBAL_BATCH -> grad_accum=$GRAD_ACCUM"
echo "  num_anchors=$NUM_ANCHORS anchor_chunk=$ANCHOR_CHUNK gamma=$LOSS_DECAY_GAMMA"
echo "  STAGE     = $STAGE"
echo ""

# ---- stage 1: split ------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "split" ]]; then
  if [[ -f "$TRAIN_JSONL" && -f "$EVAL_JSONL" ]]; then
    echo "=== [split] SKIP: $TRAIN_JSONL already exists ==="
  else
    echo "=== [split] random stratified -> $DATA_DIR ==="
    python -m gemma4_mtp.split_regen \
        --regen "$REGEN_DIR" \
        --out-dir "$DATA_DIR" \
        --eval-frac "$EVAL_FRAC" \
        --seed "$SPLIT_SEED"
    echo "=== [split] done ==="
  fi
fi

# ---- stage 2: cache ------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "cache" ]]; then
  if [[ ! -f "$TRAIN_JSONL" ]]; then
    echo "!! no train split at $TRAIN_JSONL — run STAGE=split first."; exit 1
  fi
  if [[ -f "$OUT_DIR/manifest.json" ]]; then
    echo "=== [cache] SKIP: manifest exists at $OUT_DIR (rm -rf to regen) ==="
  else
    if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]]; then
      echo "!! OUT_DIR exists and not empty: $OUT_DIR (rm -rf it)"; exit 1
    fi
    echo "=== [cache] generating -> $OUT_DIR ==="
    torchrun --standalone --nproc_per_node "$NPROC" -m gemma4_mtp.prepare_cache \
        --target "$TARGET" \
        --data "$TRAIN_JSONL" \
        --out-dir "$OUT_DIR" \
        --max-length "$MAX_LENGTH" \
        --bf16 2>&1 | filter
    echo "=== [cache] done ==="
    ls -la "$OUT_DIR"
  fi
fi

# ---- stage 3: train ------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  if [[ ! -f "$OUT_DIR/manifest.json" ]]; then
    echo "!! no cache at $OUT_DIR — run STAGE=cache first."; exit 1
  fi
  if [[ -z "$WARMUP_STEPS" ]]; then
    NUM_SAMPLES=$(python -c "import json; print(json.load(open('$OUT_DIR/manifest.json'))['num_samples'])")
    WARMUP_STEPS=$(python -c "
import math
spe = max(1, math.ceil($NUM_SAMPLES / $GLOBAL_BATCH))
print(max(1, round(spe * $EPOCHS * $WARMUP_RATIO)))
")
    echo "  [train] num_samples=$NUM_SAMPLES -> warmup_steps=$WARMUP_STEPS"
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
      --num-anchors "$NUM_ANCHORS" \
      --anchor-chunk "$ANCHOR_CHUNK" \
      --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
      --argmax-ce-weight "$ARGMAX_CE" \
      --soft-ce-weight "$SOFT_CE" \
      --l1-weight "$L1_WEIGHT" \
      --hard-ce-weight "$HARD_CE" \
      --max-length "$MAX_LENGTH" \
      $SAVE_ARG \
      --bf16 --log-every "$LOG_EVERY" 2>&1 | filter
  echo "=== [train] done. checkpoint at $CKPT_DIR ==="
fi

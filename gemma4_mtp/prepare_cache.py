#!/usr/bin/env python3
"""Offline target-signal cache for MTP training (the DSpark prepare_data pattern).

MTP training's cost is dominated by the frozen 26B target forward that produces,
per sample:
  - last_hidden        (T, H)          -> draft step-0 input
  - shared_kv_states   {full,sliding}  -> draft cross-attention KV (constant)
  - target soft labels (T, V=262144)   -> distillation target

Recomputing this every step/epoch is wasteful and OOMs (V=262144 softmax). The
target is FROZEN, so these signals are identical every epoch. We precompute them
ONCE here, 8-GPU data-parallel, and stream to disk (mount). train.py then reads
the cache and only runs the small assistant — no 26B target, no OOM.

Storage note (why not store logits): full (T, 262144) logits are ~1GB/sample.
We DON'T store them at all — target soft labels are recomputed at train time as
lm_head(last_hidden), a single frozen matmul evaluated only on supervised
(loss_mask==1) positions. This gives the FULL distribution, not a top-K approx,
and keeps the cache small.

Per-sample cache (one .pt file):
  input_ids      (T,)        int32
  loss_mask      (T,)        int8
  last_hidden    (T, H)      bf16
  kv_full_k/v    (Hkv, T, D) bf16   (last full-attention layer KV)
  kv_slide_k/v   (Hkv, T, D) bf16   (last sliding-attention layer KV)

Run 8-GPU:
  MNT=$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile
  torchrun --standalone --nproc_per_node 8 -m gemma4_mtp.prepare_cache \
      --target /tmp/models/gemma4/text_only \
      --data ./data/mtp_short/train_maiprofile_short_26b.jsonl \
      --out-dir $MNT/mtp_cache/20260615/short_train \
      --max-length 4096 --bf16
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--data", required=True, help="conversations JSONL")
    ap.add_argument("--out-dir", required=True, help="cache output dir (mount ok)")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap #samples (debug)")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma4_mtp.data import DataConfig, Gemma4ConversationParser, iter_jsonl
    from gemma4_mtp.training_step import locate_target_parts

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world, device = 0, 1, "cuda"

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    log(f"=== Loading target (world={world}) ===")
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, trust_remote_code=True).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    target_base, _, _, _ = locate_target_parts(target)

    parser = Gemma4ConversationParser(tok, max_length=args.max_length)

    # Load + tokenize all rows (cheap vs target forward), then shard by rank.
    log("=== Tokenizing rows ===")
    rows = []
    for obj in iter_jsonl(args.data):
        if obj.get("status") not in (None, "success"):
            continue
        conv = obj.get("conversations")
        if not conv:
            continue
        try:
            parsed = parser.parse(conv)
        except Exception:
            continue
        if int(parsed["loss_mask"].sum()) == 0:
            continue
        rows.append(parsed)
        if args.limit and len(rows) >= args.limit:
            break
    log(f"  {len(rows)} usable samples")

    # Shard across ranks.
    my_rows = list(range(rank, len(rows), world))
    log(f"=== Rank {rank} caching {len(my_rows)} samples ===")

    done = 0
    for idx in my_rows:
        out_path = os.path.join(args.out_dir, f"sample_{idx:07d}.pt")
        if os.path.exists(out_path):
            done += 1
            continue
        parsed = rows[idx]
        input_ids = parsed["input_ids"].unsqueeze(0).to(device)     # (1, T)
        attn = parsed["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            base_out = target_base(
                input_ids=input_ids, attention_mask=attn,
                return_shared_kv_states=True, use_cache=False)
            last_hidden = base_out.last_hidden_state[0]              # (T, H)
            skv = base_out.shared_kv_states
            # NO logits stored: target_logits = lm_head(last_hidden) is a single
            # matmul recomputed at train time, only on mask==1 positions. lm_head
            # is frozen (tied to embed), so this is cheap + gives the FULL
            # distribution (better than a top-K approximation).

            rec = {
                "input_ids": parsed["input_ids"].to(torch.int32),
                "loss_mask": parsed["loss_mask"].to(torch.int8),
                "last_hidden": last_hidden.to(torch.bfloat16).cpu(),
            }
            # shared_kv: {"full_attention": (K,V), "sliding_attention": (K,V)}
            # each K/V shape (1, Hkv, T, D) -> store (Hkv, T, D).
            fk, fv = skv["full_attention"]
            sk, sv = skv["sliding_attention"]
            rec["kv_full_k"] = fk[0].to(torch.bfloat16).cpu()
            rec["kv_full_v"] = fv[0].to(torch.bfloat16).cpu()
            rec["kv_slide_k"] = sk[0].to(torch.bfloat16).cpu()
            rec["kv_slide_v"] = sv[0].to(torch.bfloat16).cpu()

        tmp = out_path + ".tmp"
        torch.save(rec, tmp)
        os.replace(tmp, out_path)
        done += 1
        if rank == 0 and done % 50 == 0:
            print(f"[rank0] cached {done}/{len(my_rows)}", flush=True)

    log(f"=== Rank {rank} done: {done} samples ===")
    if ddp:
        dist.barrier()
        if rank == 0:
            meta = {
                "data": args.data,
                "num_samples": len(rows),
                "max_length": args.max_length,
                "dtype": "bf16" if args.bf16 else "fp32",
            }
            with open(os.path.join(args.out_dir, "cache_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[done] cache at {args.out_dir} ({len(rows)} samples)", flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

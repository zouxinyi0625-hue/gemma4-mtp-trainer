#!/usr/bin/env python3
"""Offline target-signal cache for MTP training (async sharded storage).

MTP training's cost is dominated by the frozen 26B target forward that produces,
per sample:
  - last_hidden        (T, H)          -> draft step-0 input
  - shared_kv_states   {full,sliding}  -> draft cross-attention KV (constant)
  - target soft labels (T, V=262144)   -> distillation target

Recomputing this every step/epoch is wasteful and OOMs (V=262144 softmax). The
target is FROZEN, so these signals are identical every epoch. We precompute them
ONCE here, 8-GPU data-parallel, and stream to a sharded on-mount cache. train.py
then reads the cache and only runs the small assistant — no 26B target, no OOM.

Storage (see gemma4_mtp.target_cache): samples are packed into a few large
`shard-NNNNN.bin` files with a fixed-width `samples.idx` and `manifest.json`,
written OFF the GPU thread by a background writer draining a bounded queue. This
replaces the old per-sample .pt files (slow on a network mount) and never blocks
the GPU forward on disk IO; a slow mount just throttles the producer via queue
backpressure.

We DON'T store logits (~1GB/sample): target soft labels are recomputed at train
time as lm_head(last_hidden), a single frozen matmul on supervised positions —
full distribution, small cache.

Per-sample cache fields:
  input_ids      (T,)        int32
  loss_mask      (T,)        uint8
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
import os


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--data", required=True, help="conversations JSONL")
    ap.add_argument("--out-dir", required=True,
                    help="cache output dir (mount ok); must be empty")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap #samples (debug)")
    ap.add_argument("--max-shard-bytes", type=int, default=64 * 1024 ** 3,
                    help="roll to a new shard past this size (default 64 GiB)")
    ap.add_argument("--max-queue-size", type=int, default=64,
                    help="async writer queue depth; bounds in-flight samples and "
                         "provides backpressure when the mount is slow")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma4_mtp.data import Gemma4ConversationParser, iter_jsonl
    from gemma4_mtp.training_step import locate_target_parts
    from gemma4_mtp import target_cache as tc

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        # Large timeout: the only collectives are an early barrier (after rank0
        # makes the output dir) — cheap — but writing the full cache to a slow
        # mount can take a long time, and we don't want the NCCL watchdog to
        # tear down the group during that window. Global finalize uses a
        # filesystem poll, not a collective (see the tail of main()).
        import datetime
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(hours=6))
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

    # Rank 0 prepares the (empty) output dir + _tmp/; all ranks wait, then each
    # rank writes into its own _tmp/rank_N/ so shards never collide.
    if rank == 0:
        tc.prepare_output_dir(args.out_dir)
    if ddp:
        dist.barrier()
    rank_dir = os.path.join(args.out_dir, "_tmp", f"rank_{rank}")
    os.makedirs(rank_dir, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    log(f"=== Loading target (world={world}) ===")
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, trust_remote_code=True).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    target_base, _, _, _ = locate_target_parts(target)

    parser = Gemma4ConversationParser(tok, max_length=args.max_length)

    writer = tc.AsyncCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=args.max_shard_bytes,
        max_queue_size=args.max_queue_size,
    )

    hidden_size = None
    kv_dims = None
    log(f"=== Rank {rank}: tokenizing + caching own shard ===")
    uidx = -1          # global index over USABLE rows
    done = 0
    try:
        for obj in iter_jsonl(args.data):
            if obj.get("status") not in (None, "success"):
                continue
            conv = obj.get("conversations")
            if not conv:
                continue
            uidx += 1
            if args.limit and uidx >= args.limit:
                break
            if uidx % world != rank:
                continue          # not our shard — skip tokenization entirely
            try:
                parsed = parser.parse(conv)
            except Exception:
                continue
            if int(parsed["loss_mask"].sum()) == 0:
                continue
            input_ids = parsed["input_ids"].unsqueeze(0).to(device)     # (1, T)
            attn = parsed["attention_mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                base_out = target_base(
                    input_ids=input_ids, attention_mask=attn,
                    return_shared_kv_states=True, use_cache=False)
                last_hidden = base_out.last_hidden_state[0]              # (T, H)
                skv = base_out.shared_kv_states
                fk, fv = skv["full_attention"]        # each (1, Hkv, T, D)
                sk, sv = skv["sliding_attention"]

            if hidden_size is None:
                hidden_size = int(last_hidden.shape[-1])
                # full and sliding KV have DIFFERENT head counts / head dims
                # (e.g. full (2, T, 512) vs sliding (8, T, 256)), so capture each
                # field's own (Hkv, D). fk/fv/sk/sv are (1, Hkv, T, D).
                kv_dims = {
                    "kv_full_k": (int(fk.shape[1]), int(fk.shape[-1])),
                    "kv_full_v": (int(fv.shape[1]), int(fv.shape[-1])),
                    "kv_slide_k": (int(sk.shape[1]), int(sk.shape[-1])),
                    "kv_slide_v": (int(sv.shape[1]), int(sv.shape[-1])),
                }

            # write_sample converts to CPU bytes off the GPU thread; slice batch
            # dim off the KV so they are (Hkv, T, D).
            writer.write_sample(
                input_ids=parsed["input_ids"],
                loss_mask=parsed["loss_mask"],
                last_hidden=last_hidden,
                kv_full_k=fk[0], kv_full_v=fv[0],
                kv_slide_k=sk[0], kv_slide_v=sv[0],
            )
            done += 1
            if rank == 0 and done % 50 == 0:
                print(f"[rank0] cached {done}", flush=True)
    finally:
        writer.close()

    # Per-rank summary for the global merge. source_sample_start=rank gives a
    # deterministic rank ordering (samples are interleaved by modulo sharding;
    # training shuffles, so the global ordering is irrelevant). Tensor dims go
    # in the summary too (identical across ranks; rank0 reads any one) — we do
    # NOT use an NCCL all_reduce for this, because it would have to wait for
    # every rank to finish its slow-mount writes and blows past the NCCL
    # watchdog timeout when ranks finish at very different times.
    import json
    summary = tc.LocalWriteSummary(
        global_rank=rank,
        source_sample_start=rank,
        source_sample_end=rank + 1,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
        hidden_size=hidden_size or 0,
        kv_dims={name: [int(kv_dims[name][0]), int(kv_dims[name][1])]
                 for name in tc._KV_FIELDS} if kv_dims else {},
    )
    # Atomic write so rank0's poll never sees a half-written summary.
    summary_path = os.path.join(rank_dir, "summary.json")
    tmp_summary = summary_path + ".tmp"
    with open(tmp_summary, "w") as f:
        json.dump(summary.to_json(), f, indent=2)
    os.replace(tmp_summary, summary_path)

    log(f"=== Rank {rank} done: {done} samples ===")

    # Global finalize on rank 0. Instead of an NCCL barrier (which times out
    # when ranks finish slow-mount writes minutes apart), rank 0 polls the
    # filesystem until every rank's summary.json exists. Non-zero ranks are
    # done — they just exit after destroying the process group.
    if rank == 0:
        import time
        summaries = []
        for r in range(world):
            r_dir = os.path.join(args.out_dir, "_tmp", f"rank_{r}")
            r_summary = os.path.join(r_dir, "summary.json")
            waited = 0
            while not os.path.exists(r_summary):
                time.sleep(5)
                waited += 5
                if waited % 60 == 0:
                    print(f"[rank0] waiting for rank {r} summary "
                          f"({waited}s)...", flush=True)
            summaries.append(tc.load_local_summary(r_dir))

        # Tensor dims from any rank that actually wrote samples.
        dim_src = next((s for s in summaries if s.get("kv_dims")), None)
        assert dim_src is not None, "no rank produced any samples; check --data"
        hidden_size = int(dim_src["hidden_size"])
        kv_dims = {name: (int(h), int(d))
                   for name, (h, d) in dim_src["kv_dims"].items()}

        shard_map, shards = tc.build_global_shard_map(summaries)
        for summary_json in summaries:
            r_dir = os.path.join(args.out_dir, "_tmp",
                                 f"rank_{int(summary_json['global_rank'])}")
            tc.rename_local_shards(output_dir=args.out_dir, rank_dir=r_dir,
                                   summary=summary_json, shard_map=shard_map)
        num_samples = tc.finalize_index(
            output_dir=args.out_dir, summaries=summaries, shard_map=shard_map)
        manifest = tc.build_manifest(
            num_samples=num_samples, shards=shards,
            hidden_size=hidden_size, kv_dims=kv_dims,
            extra_fields={
                "data": args.data,
                "max_length": args.max_length,
                "dtype": "bf16" if args.bf16 else "fp32",
                "target_model_name_or_path": args.target,
            },
        )
        tc.write_manifest(output_dir=args.out_dir, manifest=manifest)
        tc.cleanup_tmp_dir(args.out_dir)
        print(f"[done] cache at {args.out_dir} "
              f"({num_samples} samples, {len(shards)} shards)", flush=True)

    # No final NCCL barrier: rank0 syncs via the filesystem poll above, and a
    # barrier here would again wait on ranks slow to reach it. Each rank tears
    # down its own process group independently (a local op, no collective).
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

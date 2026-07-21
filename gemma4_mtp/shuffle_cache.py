#!/usr/bin/env python3
"""Offline shuffle of the sharded target cache (block/buffer shuffle).

WHY: training reads the cache SEQUENTIALLY (random reads over 68 GB mount
shards stall — see bench_cache / ShardOrderSampler). But the cache is written
layer-by-layer, so sequential batches are all one layer, making per-step accept
swing wildly (0.88 <-> 0.60). This rewrites the cache once with samples shuffled,
so afterwards a sequential read == a shuffled order: no mount random reads AND
mixed-layer batches, at zero training-time cost.

HOW (block/buffer shuffle, since 9.6 TB can't fit in RAM): read the old cache
SEQUENTIALLY into a bounded in-memory buffer of BUFFER samples; whenever the
buffer is full, pop a RANDOM one and write it to the new cache (also sequential
appends). This decorrelates the layer ordering over a window of BUFFER samples
— big enough to span several layer boundaries. Both reads and writes stay
sequential, so the mount runs at full bandwidth.

USAGE
  python -m gemma4_mtp.shuffle_cache \
      --in-cache  $MNT/mtp_26b/cache \
      --out-cache $MNT/mtp_26b/cache_shuf \
      --buffer 20000 --seed 0

  # then train against --cache-dir $MNT/mtp_26b/cache_shuf
"""

from __future__ import annotations

import argparse
import os
import random


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-cache", required=True, help="existing sharded cache dir")
    ap.add_argument("--out-cache", required=True, help="new shuffled cache dir")
    ap.add_argument("--buffer", type=int, default=20000,
                    help="shuffle-buffer size in samples (bigger = better mixing, "
                         "more RAM; each sample is small on CPU as bf16 tensors)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3,
                    help="new shard size cap (default 64 GiB, like prepare_cache)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="stop after N (debug)")
    return ap.parse_args()


def main():
    args = parse_args()
    from tqdm import tqdm
    import gemma4_mtp.target_cache as tc
    from gemma4_mtp.data import CacheDataset

    ds = CacheDataset(args.in_cache)
    n = len(ds)
    if args.max_samples:
        n = min(n, args.max_samples)
    print(f"=== shuffle_cache: {args.in_cache} ({n:,} samples) "
          f"-> {args.out_cache} (buffer={args.buffer:,}) ===", flush=True)

    # single-writer setup, mirroring prepare_cache rank-0 layout
    tc.prepare_output_dir(args.out_cache)
    rank_dir = os.path.join(args.out_cache, "_tmp", "rank_0")
    os.makedirs(rank_dir, exist_ok=True)
    writer = tc.AsyncCacheWriter(rank_dir=rank_dir,
                                 max_shard_bytes=args.max_shard_bytes)

    hidden_size = int(ds.hidden_size)
    kv_dims = {name: [int(h), int(d)] for name, (h, d) in ds.kv_dims.items()}

    rng = random.Random(args.seed)

    def emit(sample):
        # CacheDataset returns kv as (Hkv, T, D); write_sample expects the same
        # (Hkv, T, D) — it transposes to (T, Hkv, D) internally. Direct pass-through.
        writer.write_sample(
            input_ids=sample["input_ids"],
            loss_mask=sample["loss_mask"],
            last_hidden=sample["last_hidden"],
            kv_full_k=sample["kv_full_k"], kv_full_v=sample["kv_full_v"],
            kv_slide_k=sample["kv_slide_k"], kv_slide_v=sample["kv_slide_v"],
        )

    buf = []
    written = 0
    pbar = tqdm(total=n, desc="shuffle", unit="row")
    for i in range(n):
        buf.append(ds[i])                      # sequential READ
        if len(buf) >= args.buffer:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            emit(buf.pop())                    # sequential WRITE (random pick)
            written += 1
            pbar.update(1)
    # drain remaining buffer in random order
    rng.shuffle(buf)
    for sample in buf:
        emit(sample)
        written += 1
        pbar.update(1)
    pbar.close()

    writer.close()
    print(f"  wrote {written:,} samples; finalizing index ...", flush=True)

    # writer.close() does NOT write summary.json — build it like prepare_cache.
    import json
    summary_obj = tc.LocalWriteSummary(
        global_rank=0,
        source_sample_start=0,
        source_sample_end=1,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
        hidden_size=hidden_size or 0,
        kv_dims={name: [int(kv_dims[name][0]), int(kv_dims[name][1])]
                 for name in tc._KV_FIELDS} if kv_dims else {},
    )
    with open(os.path.join(rank_dir, "summary.json"), "w") as f:
        json.dump(summary_obj.to_json(), f, indent=2)

    # finalize (single rank_0 summary)
    summary = tc.load_local_summary(rank_dir)
    summaries = [summary]
    shard_map, shards = tc.build_global_shard_map(summaries)
    tc.rename_local_shards(output_dir=args.out_cache, rank_dir=rank_dir,
                           summary=summary, shard_map=shard_map)
    num_samples = tc.finalize_index(output_dir=args.out_cache,
                                    summaries=summaries, shard_map=shard_map)
    manifest = tc.build_manifest(
        num_samples=num_samples, shards=shards,
        hidden_size=hidden_size, kv_dims=kv_dims,
        extra_fields={"shuffled_from": args.in_cache,
                      "buffer": args.buffer, "seed": args.seed})
    tc.write_manifest(output_dir=args.out_cache, manifest=manifest)
    tc.cleanup_tmp_dir(args.out_cache)
    print(f"[done] shuffled cache at {args.out_cache} "
          f"({num_samples:,} samples, {len(shards)} shards)", flush=True)


if __name__ == "__main__":
    main()

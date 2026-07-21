#!/usr/bin/env python3
"""Benchmark CacheDataset read speed — is the training bottleneck the cache IO?

Training was ~25000 s/step with GPUs mostly idle, which points at the cache
reads (9.6 TB of 68 GB shards on the mount) stalling the DataLoader. This
measures it directly: load one sample at a time and report rows/sec, both
RANDOM order (what DistributedSampler(shuffle=True) does) and SEQUENTIAL order
(shards read front-to-back). A big gap = random mmap reads over the mount are
the problem; both slow = the mount is just slow; both fast = look elsewhere.

USAGE
  python -m gemma4_mtp.bench_cache --cache-dir <cache> --n 200
  # compare against a local copy of a few shards, or a different --cache-dir
"""

from __future__ import annotations

import argparse
import random
import time


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True, help="sharded cache dir (manifest.json)")
    ap.add_argument("--n", type=int, default=200, help="samples to read per mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["both", "random", "seq"], default="both")
    ap.add_argument("--warmup", type=int, default=5, help="untimed reads first")
    return ap.parse_args()


def _time_reads(ds, indices, warmup):
    import torch  # noqa: F401 (CacheDataset returns tensors)
    # warmup (open shards / prime any caches) — not timed
    for i in indices[:warmup]:
        _ = ds[i]
    idxs = indices[warmup:]
    n = len(idxs)
    nbytes = 0
    t0 = time.time()
    slow = []  # (idx, seconds) for the slowest reads
    for i in idxs:
        s = time.time()
        item = ds[i]
        dt = time.time() - s
        # rough bytes moved this sample
        for v in item.values() if isinstance(item, dict) else []:
            try:
                nbytes += v.numel() * v.element_size()
            except Exception:
                pass
        if len(slow) < 5 or dt > slow[-1][1]:
            slow.append((int(i), dt))
            slow.sort(key=lambda x: -x[1])
            slow[:] = slow[:5]
    el = time.time() - t0
    rps = n / el if el > 0 else 0.0
    mbps = (nbytes / 1e6) / el if el > 0 else 0.0
    return n, el, rps, mbps, slow


def main():
    args = parse_args()
    from gemma4_mtp.data import CacheDataset

    print(f"=== bench cache: {args.cache_dir} ===", flush=True)
    ds = CacheDataset(args.cache_dir)
    total = len(ds)
    print(f"  num_samples={total:,}", flush=True)

    rng = random.Random(args.seed)

    if args.mode in ("random", "both"):
        idx = [rng.randrange(total) for _ in range(args.n + args.warmup)]
        print(f"\n--- RANDOM order ({args.n} timed reads) ---", flush=True)
        n, el, rps, mbps, slow = _time_reads(ds, idx, args.warmup)
        print(f"  {n} reads in {el:.1f}s -> {rps:.2f} rows/s, {mbps:.1f} MB/s", flush=True)
        print(f"  slowest: " + ", ".join(f"idx{i}={d:.2f}s" for i, d in slow), flush=True)

    if args.mode in ("seq", "both"):
        start = rng.randrange(max(1, total - args.n - args.warmup))
        idx = list(range(start, start + args.n + args.warmup))
        print(f"\n--- SEQUENTIAL order ({args.n} timed reads from idx {start}) ---", flush=True)
        n, el, rps, mbps, slow = _time_reads(ds, idx, args.warmup)
        print(f"  {n} reads in {el:.1f}s -> {rps:.2f} rows/s, {mbps:.1f} MB/s", flush=True)
        print(f"  slowest: " + ", ".join(f"idx{i}={d:.2f}s" for i, d in slow), flush=True)

    print("\n=== interpretation ===")
    print("  random << sequential  -> random mmap reads over the mount are the bottleneck")
    print("  both slow             -> the mount itself is slow (copy cache local, or shrink it)")
    print("  both fast             -> IO is fine; bottleneck is elsewhere (compute/DDP)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe the target-cache on the mount: file sizes, sequential read speed, and
whether random-access reads (what training does) hang.

Usage:
  python probe_cache.py <CACHE_DIR>
    e.g. python probe_cache.py "$AZURE_ML_INPUT_msndni/.../mtp_cache/20260715_014947/short_train"
"""
import json
import os
import sys
import time

CACHE = sys.argv[1]

print("=" * 64)
print("cache dir:", CACHE)
print("=" * 64)

# 1. manifest
man_path = os.path.join(CACHE, "manifest.json")
with open(man_path) as f:
    man = json.load(f)
print(f"manifest: num_samples={man['num_samples']} num_shards={man['num_shards']} "
      f"hidden={man['hidden_size']} kv_dims={man.get('kv_dims')}")

# 2. file sizes (os.path.getsize works even when du reports 0 on blobfuse)
idx_path = os.path.join(CACHE, "samples.idx")
idx_bytes = os.path.getsize(idx_path)
print(f"samples.idx: {idx_bytes/1e6:.1f} MB "
      f"({idx_bytes} bytes, {idx_bytes // 72} records @ 72B)")

shard_files = sorted(
    os.path.join(CACHE, s["file_name"]) for s in man["shards"])
total = 0
print("shards:")
for p in shard_files:
    sz = os.path.getsize(p)
    total += sz
    print(f"  {os.path.basename(p):20s} {sz/1e9:6.2f} GB")
print(f"TOTAL cache size: {total/1e9:.2f} GB")

# 3. sequential read speed on the first shard (what `cp` would see)
print()
print("--- sequential read test (first shard, 512 MB) ---")
test_shard = shard_files[0]
CHUNK = 512 * 1024 * 1024
t0 = time.time()
read = 0
with open(test_shard, "rb") as f:
    buf = f.read(CHUNK)
    read = len(buf)
dt = time.time() - t0
print(f"read {read/1e6:.0f} MB in {dt:.1f}s -> {read/1e6/dt:.1f} MB/s")

# 4. random-access read test (what training's mmap does — the thing that hangs)
print()
print("--- random-access read test (10 random samples via samples.idx) ---")
import struct
REC = struct.Struct("<QII" + "Q" * 7)
with open(idx_path, "rb") as f:
    idx_data = f.read()
n = len(idx_data) // 72
print(f"index has {n} records")

# read 10 evenly-spaced samples, seeking into shards like training does
import mmap
opened = {}
def shard_mmap(shard_id):
    if shard_id not in opened:
        fh = open(shard_files[shard_id], "rb")
        opened[shard_id] = (fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ))
    return opened[shard_id][1]

t0 = time.time()
for i in range(0, n, max(1, n // 10))[:10]:
    rec = REC.unpack_from(idx_data, i * 72)
    sample_id, shard_id, seq_len = rec[0], rec[1], rec[2]
    off = rec[3]  # input_ids offset
    mm = shard_mmap(shard_id)
    _ = bytes(mm[off:off + seq_len * 4])  # read input_ids
    print(f"  sample {sample_id}: shard {shard_id}, seq_len {seq_len} — OK")
dt = time.time() - t0
print(f"10 random samples read in {dt:.2f}s ({dt/10*1000:.0f} ms/sample)")
print()
print("If this finished fast, the mount is responsive right now and training")
print("should run. If ms/sample is huge (>1000ms), mount random-read is the")
print("bottleneck — copy the cache to local /tmp first.")

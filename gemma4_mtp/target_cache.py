"""Async sharded target-signal cache for MTP training.

Ports the DeepSpec target-cache storage engine (async writer thread + bounded
queue backpressure + large binary shards + fixed-width struct index + mmap
reader) to this repo, adapted to the 7 per-sample tensors MTP training needs.

Slow-mount problem: writing one .pt per sample straight to a cosmos/ADLS mount
is dominated by per-file metadata ops. Instead we pack many samples into a few
large `shard-NNNNN.bin` files with a `samples.idx` fixed-width index and a
`manifest.json`, and write them off the GPU thread via a background writer that
drains a bounded queue (so a slow mount throttles the producer instead of
blowing up memory).

Per-sample fields (one index record + a run of bytes in a shard):
  input_ids    (T,)        int32
  loss_mask    (T,)        uint8   (stored from int8/long loss_mask)
  last_hidden  (T, H)      bf16
  kv_full_k    (Hkv, T, D) bf16    stored transposed as (T, Hkv, D)
  kv_full_v    (Hkv, T, D) bf16    stored transposed as (T, Hkv, D)
  kv_slide_k   (Hkv, T, D) bf16    stored transposed as (T, Hkv, D)
  kv_slide_v   (Hkv, T, D) bf16    stored transposed as (T, Hkv, D)

KV is stored T-major so the shard layout is uniformly (T, ...) and the index
only needs seq_len; Hkv and D live in the manifest (constant across samples).
The reader transposes back to (Hkv, T, D) for collate_cache.
"""

from __future__ import annotations

import json
import mmap
import os
import queue
import shutil
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass

import torch

# numpy is used by the mmap reader (CacheDataset) for zero-copy views; the
# writer path falls back to a pure-torch byte path when numpy is absent.
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


CACHE_VERSION = 1

# sample_id, shard_id, seq_len, + 7 field byte-offsets within the shard.
INDEX_RECORD_STRUCT = struct.Struct("<QII" + "Q" * 7)
INDEX_RECORD_SIZE = INDEX_RECORD_STRUCT.size

_FIELDS = (
    "input_ids",
    "loss_mask",
    "last_hidden",
    "kv_full_k",
    "kv_full_v",
    "kv_slide_k",
    "kv_slide_v",
)
_KV_FIELDS = ("kv_full_k", "kv_full_v", "kv_slide_k", "kv_slide_v")


def atomic_json_dump(payload, path: str):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def expected_tensor_nbytes(*, seq_len, hidden_size, kv_dims):
    """kv_dims: {field_name: (num_kv_heads, head_dim)} — full and sliding differ."""
    out = {
        "input_ids": int(seq_len) * 4,
        "loss_mask": int(seq_len) * 1,
        "last_hidden": int(seq_len) * int(hidden_size) * 2,
    }
    for name in _KV_FIELDS:
        heads, dim = kv_dims[name]
        out[name] = int(seq_len) * int(heads) * int(dim) * 2
    return out


def pack_index_record(*, sample_id, shard_id, seq_len, offsets):
    return INDEX_RECORD_STRUCT.pack(
        int(sample_id),
        int(shard_id),
        int(seq_len),
        *(int(offsets[name]) for name in _FIELDS),
    )


def unpack_index_record(buffer, offset: int = 0):
    values = INDEX_RECORD_STRUCT.unpack_from(buffer, offset)
    sample_id, shard_id, seq_len = values[0], values[1], values[2]
    offsets = {name: values[3 + i] for i, name in enumerate(_FIELDS)}
    return {"sample_id": sample_id, "shard_id": shard_id,
            "seq_len": seq_len, "offsets": offsets}


def _tensor_bytes(tensor, dtype):
    """Serialize a tensor to raw little-endian bytes of `dtype`, numpy-optional."""
    t = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    raw = t.view(torch.uint8) if dtype != torch.bfloat16 else t.view(torch.uint16).view(torch.uint8)
    if _HAS_NUMPY:
        return raw.numpy().tobytes()
    return bytes(raw.reshape(-1).tolist())


def _int_tensor_to_bytes(tensor, dtype):
    return _tensor_bytes(tensor, dtype)


def _bf16_to_bytes(tensor):
    return _tensor_bytes(tensor, torch.bfloat16)


def compute_local_sample_range(*, num_samples, rank, world_size):
    base = int(num_samples) // int(world_size)
    remainder = int(num_samples) % int(world_size)
    start = rank * base + min(rank, remainder)
    local_count = base + (1 if rank < remainder else 0)
    return start, start + local_count


def prepare_output_dir(output_dir: str):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        existing = sorted(os.listdir(output_dir))
        if existing:
            raise FileExistsError(
                f"Target cache output dir is not empty: {output_dir}. "
                "Use a new output directory."
            )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "_tmp"), exist_ok=True)


@dataclass(frozen=True)
class SampleBytes:
    seq_len: int
    payloads: dict  # field name -> bytes, in _FIELDS order


def build_sample_bytes(*, input_ids, loss_mask, last_hidden,
                       kv_full_k, kv_full_v, kv_slide_k, kv_slide_v):
    kv = {"kv_full_k": kv_full_k, "kv_full_v": kv_full_v,
          "kv_slide_k": kv_slide_k, "kv_slide_v": kv_slide_v}
    payloads = {
        "input_ids": _int_tensor_to_bytes(input_ids, torch.int32),
        "loss_mask": _int_tensor_to_bytes(loss_mask, torch.uint8),
        "last_hidden": _bf16_to_bytes(last_hidden),
    }
    for name in _KV_FIELDS:
        # (Hkv, T, D) -> (T, Hkv, D) so the shard layout is T-major.
        payloads[name] = _bf16_to_bytes(kv[name].transpose(0, 1).contiguous())
    return SampleBytes(seq_len=int(input_ids.shape[0]), payloads=payloads)


@dataclass
class LocalWriteSummary:
    global_rank: int
    source_sample_start: int
    source_sample_end: int
    num_local_samples: int
    num_local_shards: int
    local_shard_files: list
    hidden_size: int = 0
    kv_dims: dict = None  # {field_name: [num_kv_heads, head_dim]}

    def to_json(self):
        return {
            "global_rank": self.global_rank,
            "source_sample_start": self.source_sample_start,
            "source_sample_end": self.source_sample_end,
            "num_local_samples": self.num_local_samples,
            "num_local_shards": self.num_local_shards,
            "local_shard_files": list(self.local_shard_files),
            "hidden_size": int(self.hidden_size),
            "kv_dims": self.kv_dims or {},
        }


class LocalCacheWriter:
    def __init__(self, *, rank_dir: str, max_shard_bytes: int):
        self.rank_dir = rank_dir
        self.max_shard_bytes = int(max_shard_bytes)
        self.local_index_path = os.path.join(rank_dir, "samples.local.idx")
        self.index_handle = open(self.local_index_path, "wb")
        self.current_shard_id = -1
        self.current_shard_handle = None
        self.current_shard_size = 0
        self.local_shard_files = []
        self.num_local_samples = 0

    def close(self):
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
            self.current_shard_handle = None
        if getattr(self, "index_handle", None) is not None:
            self.index_handle.flush()
            os.fsync(self.index_handle.fileno())
            self.index_handle.close()
            self.index_handle = None

    def _open_new_shard(self):
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
        self.current_shard_id += 1
        file_name = f"shard-local-{self.current_shard_id:05d}.bin"
        self.current_shard_handle = open(os.path.join(self.rank_dir, file_name), "wb")
        self.current_shard_size = 0
        self.local_shard_files.append(file_name)

    def _ensure_shard(self, sample_nbytes: int):
        if self.current_shard_handle is None:
            self._open_new_shard()
            return
        if (self.current_shard_size > 0
                and self.current_shard_size + int(sample_nbytes) > self.max_shard_bytes):
            self._open_new_shard()

    def write_sample_bytes(self, sample: SampleBytes):
        sample_nbytes = sum(len(sample.payloads[name]) for name in _FIELDS)
        self._ensure_shard(sample_nbytes)
        offsets = {}
        for name in _FIELDS:
            offsets[name] = self.current_shard_size
            payload = sample.payloads[name]
            self.current_shard_handle.write(payload)
            self.current_shard_size += len(payload)
        self.index_handle.write(
            pack_index_record(
                sample_id=self.num_local_samples,
                shard_id=self.current_shard_id,
                seq_len=sample.seq_len,
                offsets=offsets,
            )
        )
        self.num_local_samples += 1


class AsyncCacheWriter:
    def __init__(self, *, rank_dir: str, max_shard_bytes: int, max_queue_size: int = 128):
        self.writer = LocalCacheWriter(rank_dir=rank_dir, max_shard_bytes=max_shard_bytes)
        # Queue CPU byte records only; never hold CUDA tensor references here.
        self.queue = queue.Queue(maxsize=int(max_queue_size))
        self.sentinel = object()
        self.num_local_samples = 0
        self._closed = False
        self._exception = None
        self.thread = threading.Thread(
            target=self._run, name=f"target-cache-writer-{os.path.basename(rank_dir)}")
        self.thread.start()

    @property
    def local_shard_files(self):
        return self.writer.local_shard_files

    def _run(self):
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self.sentinel:
                        break
                    self.writer.write_sample_bytes(item)
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self._exception = exc
        finally:
            try:
                self.writer.close()
            except BaseException as exc:
                if self._exception is None:
                    self._exception = exc

    def _raise_if_failed(self):
        if self._exception is not None:
            raise RuntimeError("Async target cache writer failed.") from self._exception

    def _put(self, item):
        while True:
            self._raise_if_failed()
            try:
                self.queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    def write_sample(self, *, input_ids, loss_mask, last_hidden,
                     kv_full_k, kv_full_v, kv_slide_k, kv_slide_v):
        sample = build_sample_bytes(
            input_ids=input_ids, loss_mask=loss_mask, last_hidden=last_hidden,
            kv_full_k=kv_full_k, kv_full_v=kv_full_v,
            kv_slide_k=kv_slide_k, kv_slide_v=kv_slide_v)
        self._put(sample)
        self.num_local_samples += 1

    def close(self):
        if self._closed:
            self._raise_if_failed()
            return
        if self._exception is None:
            self._put(self.sentinel)
        self.thread.join()
        self._closed = True
        self._raise_if_failed()
        assert self.writer.num_local_samples == self.num_local_samples, (
            "Async target cache writer lost samples: "
            f"{self.writer.num_local_samples} != {self.num_local_samples}"
        )


# --- global finalization (main rank, after barrier) ------------------------

def load_local_summary(rank_dir: str):
    with open(os.path.join(rank_dir, "summary.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_global_shard_map(summaries):
    shard_map = {}
    shards = []
    next_shard_id = 0
    for summary in sorted(summaries, key=lambda item: int(item["source_sample_start"])):
        local_map = []
        for _ in summary["local_shard_files"]:
            local_map.append(next_shard_id)
            shards.append({"shard_id": next_shard_id,
                           "file_name": f"shard-{next_shard_id:05d}.bin"})
            next_shard_id += 1
        shard_map[int(summary["global_rank"])] = local_map
    return shard_map, shards


def rename_local_shards(*, output_dir, rank_dir, summary, shard_map):
    local_map = shard_map[int(summary["global_rank"])]
    for local_shard_id, file_name in enumerate(summary["local_shard_files"]):
        source = os.path.join(rank_dir, file_name)
        target = os.path.join(output_dir, f"shard-{local_map[local_shard_id]:05d}.bin")
        os.replace(source, target)


def finalize_index(*, output_dir, summaries, shard_map):
    index_tmp_path = os.path.join(output_dir, "samples.idx.tmp")
    next_sample_id = 0
    with open(index_tmp_path, "wb") as output_handle:
        for summary in sorted(summaries, key=lambda item: int(item["source_sample_start"])):
            rank_dir = os.path.join(output_dir, "_tmp", f"rank_{int(summary['global_rank'])}")
            with open(os.path.join(rank_dir, "samples.local.idx"), "rb") as local_handle:
                local_bytes = local_handle.read()
            assert len(local_bytes) % INDEX_RECORD_SIZE == 0, (
                f"Local index has invalid size: {rank_dir}")
            expected_local = 0
            for offset in range(0, len(local_bytes), INDEX_RECORD_SIZE):
                record = unpack_index_record(local_bytes, offset)
                assert int(record["sample_id"]) == expected_local, (
                    "Local index not ordered by local sample_id: "
                    f"got {record['sample_id']}, expected {expected_local}")
                global_shard_id = shard_map[int(summary["global_rank"])][int(record["shard_id"])]
                output_handle.write(pack_index_record(
                    sample_id=next_sample_id,
                    shard_id=global_shard_id,
                    seq_len=record["seq_len"],
                    offsets=record["offsets"],
                ))
                expected_local += 1
                next_sample_id += 1
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(index_tmp_path, os.path.join(output_dir, "samples.idx"))
    return next_sample_id


def build_manifest(*, num_samples, shards, hidden_size, kv_dims,
                   extra_fields=None):
    """kv_dims: {field_name: (num_kv_heads, head_dim)}. full/sliding differ, so
    each KV field stores its own head count and head dim."""
    manifest = {
        "version": CACHE_VERSION,
        "num_samples": int(num_samples),
        "num_shards": len(shards),
        "index_record_size": INDEX_RECORD_SIZE,
        "hidden_size": int(hidden_size),
        "kv_dims": {name: [int(kv_dims[name][0]), int(kv_dims[name][1])]
                    for name in _KV_FIELDS},
        "shards": shards,
    }
    if extra_fields:
        manifest.update(extra_fields)
    return manifest


def write_manifest(*, output_dir, manifest):
    atomic_json_dump(manifest, os.path.join(output_dir, "manifest.json"))


def cleanup_tmp_dir(output_dir: str):
    tmp_dir = os.path.join(output_dir, "_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


def load_manifest(cache_dir: str):
    manifest_path = os.path.join(cache_dir, "manifest.json")
    assert os.path.exists(manifest_path), f"Missing target cache manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert int(manifest["version"]) == CACHE_VERSION, (
        f"Unsupported cache version: {manifest['version']} != {CACHE_VERSION}")
    assert int(manifest["index_record_size"]) == INDEX_RECORD_SIZE, (
        "index_record_size mismatch: "
        f"{manifest['index_record_size']} != {INDEX_RECORD_SIZE}")
    index_path = os.path.join(cache_dir, "samples.idx")
    assert os.path.exists(index_path), f"Missing index: {index_path}"
    index_size = os.path.getsize(index_path)
    assert index_size == int(manifest["num_samples"]) * INDEX_RECORD_SIZE, (
        f"samples.idx size {index_size} != num_samples * record_size")
    for expected_id, shard in enumerate(manifest["shards"]):
        assert int(shard["shard_id"]) == expected_id, "shard ids must be contiguous from 0"
        shard_path = os.path.join(cache_dir, shard["file_name"])
        assert os.path.exists(shard_path), f"Missing shard file: {shard_path}"
    return manifest


# --- read side -------------------------------------------------------------

class CacheDataset(torch.utils.data.Dataset):
    """mmap reader for the sharded target cache (LRU over open shards)."""

    def __init__(self, cache_dir: str, max_open_shards: int = 4):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_dir)
        self.manifest = load_manifest(self.cache_dir)
        self.num_samples = int(self.manifest["num_samples"])
        self.hidden_size = int(self.manifest["hidden_size"])
        self.kv_dims = {name: (int(h), int(d))
                        for name, (h, d) in self.manifest["kv_dims"].items()}
        self.index_path = os.path.join(self.cache_dir, "samples.idx")
        self.index_file = None
        self.index_mmap = None
        self.max_open_shards = max_open_shards
        self.shard_handles = OrderedDict()
        self.shard_mmaps = OrderedDict()
        self.shard_paths = {
            int(shard["shard_id"]): os.path.join(self.cache_dir, shard["file_name"])
            for shard in self.manifest["shards"]
        }

    def __len__(self):
        return self.num_samples

    def close(self):
        for shard_mmap in getattr(self, "shard_mmaps", {}).values():
            shard_mmap.close()
        for handle in getattr(self, "shard_handles", {}).values():
            handle.close()
        if hasattr(self, "shard_mmaps"):
            self.shard_mmaps.clear()
        if hasattr(self, "shard_handles"):
            self.shard_handles.clear()
        if getattr(self, "index_mmap", None) is not None:
            self.index_mmap.close()
            self.index_mmap = None
        if getattr(self, "index_file", None) is not None:
            self.index_file.close()
            self.index_file = None

    def __del__(self):  # pragma: no cover
        self.close()

    def __getstate__(self):  # pragma: no cover
        state = dict(self.__dict__)
        state["index_file"] = None
        state["index_mmap"] = None
        state["shard_handles"] = OrderedDict()
        state["shard_mmaps"] = OrderedDict()
        return state

    def _ensure_index_mmap(self):
        if self.index_mmap is None:
            self.index_file = open(self.index_path, "rb")
            self.index_mmap = mmap.mmap(self.index_file.fileno(), 0, access=mmap.ACCESS_READ)

    def _get_shard_mmap(self, shard_id: int):
        shard_id = int(shard_id)
        if shard_id in self.shard_mmaps:
            self.shard_mmaps.move_to_end(shard_id)
            self.shard_handles.move_to_end(shard_id)
            return self.shard_mmaps[shard_id]
        handle = open(self.shard_paths[shard_id], "rb")
        self.shard_handles[shard_id] = handle
        self.shard_mmaps[shard_id] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        while len(self.shard_mmaps) > self.max_open_shards:
            evicted_id, evicted_mmap = self.shard_mmaps.popitem(last=False)
            evicted_mmap.close()
            self.shard_handles.pop(evicted_id).close()
        return self.shard_mmaps[shard_id]

    def _read_record(self, index: int):
        self._ensure_index_mmap()
        offset = int(index) * INDEX_RECORD_SIZE
        record = unpack_index_record(self.index_mmap, offset)
        assert int(record["sample_id"]) == int(index), (
            "Index not dense/sorted by sample_id: "
            f"record sample_id={record['sample_id']}, expected {index}")
        return record

    def _read_int(self, *, shard_mmap, offset, count, np_dtype, torch_dtype, nbytes):
        assert int(offset) + int(nbytes) <= shard_mmap.size(), "tensor beyond shard size"
        array = np.frombuffer(shard_mmap, dtype=np_dtype, count=int(count),
                              offset=int(offset)).copy()
        return torch.from_numpy(array).to(dtype=torch_dtype)

    def _read_bf16(self, *, shard_mmap, offset, shape, nbytes):
        assert int(offset) + int(nbytes) <= shard_mmap.size(), "tensor beyond shard size"
        array = np.frombuffer(shard_mmap, dtype=np.uint16,
                              count=int(np.prod(shape)), offset=int(offset)).copy()
        return torch.from_numpy(array).view(torch.bfloat16).view(*shape)

    def __getitem__(self, index: int):
        if not (0 <= int(index) < self.num_samples):
            raise IndexError(index)
        record = self._read_record(int(index))
        seq_len = int(record["seq_len"])
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        off = record["offsets"]
        shard_mmap = self._get_shard_mmap(int(record["shard_id"]))
        nbytes = expected_tensor_nbytes(
            seq_len=seq_len, hidden_size=self.hidden_size, kv_dims=self.kv_dims)

        input_ids = self._read_int(
            shard_mmap=shard_mmap, offset=off["input_ids"], count=seq_len,
            np_dtype=np.int32, torch_dtype=torch.int32, nbytes=nbytes["input_ids"])
        loss_mask = self._read_int(
            shard_mmap=shard_mmap, offset=off["loss_mask"], count=seq_len,
            np_dtype=np.uint8, torch_dtype=torch.uint8, nbytes=nbytes["loss_mask"])
        last_hidden = self._read_bf16(
            shard_mmap=shard_mmap, offset=off["last_hidden"],
            shape=(seq_len, self.hidden_size), nbytes=nbytes["last_hidden"])

        out = {"input_ids": input_ids, "loss_mask": loss_mask, "last_hidden": last_hidden}
        for name in _KV_FIELDS:
            heads, dim = self.kv_dims[name]
            # stored as (T, Hkv, D); transpose back to (Hkv, T, D) for collate.
            kv = self._read_bf16(
                shard_mmap=shard_mmap, offset=off[name],
                shape=(seq_len, heads, dim), nbytes=nbytes[name])
            out[name] = kv.transpose(0, 1).contiguous()
        return out


def collate_cache(batch, pad_token_id: int):
    """Right-pad cached samples (variable T) into batched tensors + shared_kv.

    Output contract matches training_step_from_cache: input_ids (B,T),
    loss_mask (B,T), last_hidden (B,T,H), shared_kv_states with each K/V padded
    to (B, Hkv, T, D). Target soft labels are recomputed at train time via
    lm_head(last_hidden), so no logits are stored or batched here.
    """
    B = len(batch)
    T = max(b["input_ids"].shape[0] for b in batch)
    H = batch[0]["last_hidden"].shape[-1]

    input_ids = torch.full((B, T), pad_token_id, dtype=torch.long)
    loss_mask = torch.zeros((B, T), dtype=torch.long)
    last_hidden = torch.zeros((B, T, H), dtype=batch[0]["last_hidden"].dtype)

    kv_shapes = {k: batch[0][k].shape for k in _KV_FIELDS}  # (Hkv, T, D)
    kv = {k: torch.zeros((B, v[0], T, v[2]), dtype=batch[0][k].dtype)
          for k, v in kv_shapes.items()}

    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"].long()
        loss_mask[i, :n] = b["loss_mask"].long()
        last_hidden[i, :n] = b["last_hidden"]
        for k in kv:
            kv[k][i, :, :n, :] = b[k]

    shared_kv_states = {
        "full_attention": (kv["kv_full_k"], kv["kv_full_v"]),
        "sliding_attention": (kv["kv_slide_k"], kv["kv_slide_v"]),
    }
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "last_hidden": last_hidden,
        "shared_kv_states": shared_kv_states,
    }

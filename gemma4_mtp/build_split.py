#!/usr/bin/env python3
"""Split the 26B regen data into train/eval using DSpark's EXISTING short-layers
split (by id = "{layer}:{prompt_hash}"), keeping the 26B responses.

We do NOT re-run DSpark's RNG. DSpark already wrote its split to
    prepared_prompts/20260615/short_layers/
        train_<layer>.jsonl              (train prompts, each has "id")
        eval_datasets/maiprofile_<layer>.jsonl   (eval prompts, each has "id")
Both carry id = "{layer}:{prompt_hash}", identical to the regen file's id. So we
just collect DSpark's train_ids / eval_ids and route each regen row (which has
the 26B response) accordingly. Regen rows whose id is in neither set (i.e. not
part of the short-layers subset) are dropped.

Output rows are exactly what gemma4_mtp.data expects:
    {"conversations": [...system/user/assistant...], "status": "success", "id": ...}

Usage (inspect first — writes small samples + counts, not full files):
    python -m gemma4_mtp.build_split \
        --split-dir $MNT/prepared_prompts/20260615/short_layers \
        --regen     $MNT/eagle3/20260615/regen_26b/train_all_layers_regen_26b.jsonl \
        --out-dir   ./data/mtp_short \
        --sample-only

Then run for real (writes full train/eval jsonl):
    python -m gemma4_mtp.build_split \
        --split-dir $MNT/prepared_prompts/20260615/short_layers \
        --regen     $MNT/eagle3/20260615/regen_26b/train_all_layers_regen_26b.jsonl \
        --out-dir   ./data/mtp_short
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_ids(paths):
    """Collect the 'id' field from a list of JSONL files into a set."""
    ids = set()
    n_files = 0
    for p in paths:
        if not p.exists():
            continue
        n_files += 1
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = obj.get("id")
                if rid:
                    ids.add(rid)
    return ids, n_files


def find_split_files(split_dir: Path):
    """Locate DSpark train_*.jsonl and eval_datasets/*.jsonl under split_dir."""
    train_files = sorted(split_dir.glob("train_layer*.jsonl"))
    eval_dir = split_dir / "eval_datasets"
    eval_files = sorted(eval_dir.glob("maiprofile_*.jsonl")) if eval_dir.exists() else []
    return train_files, eval_files


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True,
                    help="DSpark short_layers dir (train_<layer>.jsonl + eval_datasets/)")
    ap.add_argument("--regen", required=True,
                    help="26B regen JSONL (has id + conversations w/ 26B response)")
    ap.add_argument("--out-dir", required=True, help="output dir for train/eval jsonl")
    ap.add_argument("--sample-only", action="store_true",
                    help="don't write full files; print counts + a few samples")
    ap.add_argument("--sample-n", type=int, default=2)
    args = ap.parse_args()

    split_dir = Path(args.split_dir).expanduser()
    regen = Path(args.regen).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    # 1. Collect DSpark's train/eval id sets (per-layer train_ files exclude the
    #    per-layer eval already, so train_ids and eval_ids are disjoint).
    train_files, eval_files = find_split_files(split_dir)
    print("=== DSpark short-layers split files ===", flush=True)
    print(f"  train files: {[p.name for p in train_files]}")
    print(f"  eval  files: {[p.name for p in eval_files]}")

    train_ids, n_tr = collect_ids(train_files)
    eval_ids, n_ev = collect_ids(eval_files)
    overlap = train_ids & eval_ids
    print(f"  train_ids={len(train_ids)} (from {n_tr} files)")
    print(f"  eval_ids ={len(eval_ids)} (from {n_ev} files)")
    print(f"  overlap  ={len(overlap)} (should be 0)")

    # 2. Route each regen row by id.
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_maiprofile_short_26b.jsonl"
    eval_out = out_dir / "eval_maiprofile_short_26b.jsonl"

    n_regen = 0
    n_train = 0
    n_eval = 0
    n_drop = 0
    train_samples = []
    eval_samples = []
    by_layer = {}

    tw = None if args.sample_only else train_out.open("w", encoding="utf-8")
    ew = None if args.sample_only else eval_out.open("w", encoding="utf-8")
    try:
        with regen.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_regen += 1
                obj = json.loads(line)
                rid = obj.get("id")
                layer = obj.get("source_layer", "?")
                out_row = {
                    "id": rid,
                    "source_layer": layer,
                    "conversations": obj.get("conversations"),
                    "status": "success",
                }
                if rid in train_ids:
                    n_train += 1
                    by_layer.setdefault(layer, {"train": 0, "eval": 0})["train"] += 1
                    if tw:
                        tw.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    elif len(train_samples) < args.sample_n:
                        train_samples.append(out_row)
                elif rid in eval_ids:
                    n_eval += 1
                    by_layer.setdefault(layer, {"train": 0, "eval": 0})["eval"] += 1
                    if ew:
                        ew.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    elif len(eval_samples) < args.sample_n:
                        eval_samples.append(out_row)
                else:
                    n_drop += 1
    finally:
        if tw:
            tw.close()
        if ew:
            ew.close()

    print("\n=== Routing result ===", flush=True)
    print(f"  regen rows read : {n_regen}")
    print(f"  -> train        : {n_train}")
    print(f"  -> eval         : {n_eval}")
    print(f"  -> dropped (not in short subset): {n_drop}")
    print("  by layer (short subset only):")
    for layer in sorted(by_layer):
        c = by_layer[layer]
        print(f"    {layer:32s} train={c['train']:6d} eval={c['eval']:5d}")

    # 3. Show what the data looks like.
    def show(samples, tag):
        print(f"\n--- {tag} sample ---")
        for i, row in enumerate(samples):
            convs = row.get("conversations") or []
            roles = [m.get("role") for m in convs]
            print(f"  [{i}] id={row['id']}")
            print(f"      roles={roles}")
            for m in convs:
                if m.get("role") == "assistant":
                    c = m.get("content", "")
                    prev = c[:200] + ("..." if len(c) > 200 else "")
                    print(f"      assistant(26B): {prev!r}")

    if args.sample_only:
        show(train_samples, "TRAIN")
        show(eval_samples, "EVAL")
        print("\n[sample-only] No full files written. Review the samples above; "
              "re-run without --sample-only to write:")
        print(f"    {train_out}")
        print(f"    {eval_out}")
    else:
        print(f"\nWrote:\n  {train_out} ({n_train} rows)\n  {eval_out} ({n_eval} rows)")
        # Also print one sample from disk so we can eyeball it.
        with train_out.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first:
            row = json.loads(first)
            show([row], "TRAIN (from file)")


if __name__ == "__main__":
    main()

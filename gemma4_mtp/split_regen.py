#!/usr/bin/env python3
"""Randomly split regen data into train / eval JSONL for MTP training.

Replaces the old id-matched split (build_split.py, which routed rows by
DSpark's precomputed train_ids/eval_ids). The expanded MAI Profile data has no
precomputed split, so we split RANDOMLY here.

Stratified by layer: each layer is split independently by the same eval
fraction, so both train and eval cover every layer proportionally (a plain
global shuffle could starve a small layer). Reproducible via --seed.

INPUT
  --regen a directory of <layer>_regen.jsonl files (produced by run_regen.sh),
  or a single regen JSONL. Each row: {conversations:[system,user,assistant],
  status, id, source_layer}. Layer is taken from source_layer, or the filename
  (<layer>_regen.jsonl -> <layer>), or "all" if neither.

OUTPUT (in --out-dir)
  train_maiprofile_26b.jsonl
  eval_maiprofile_26b.jsonl
  (row schema unchanged: what gemma4_mtp.data expects.)

USAGE
  # dir of per-layer files, 10% eval, reproducible
  python -m gemma4_mtp.split_regen \
      --regen /tmp/regen --out-dir ./data/mtp_26b --eval-frac 0.1 --seed 0

  # single file, fixed eval count per layer
  python -m gemma4_mtp.split_regen \
      --regen /tmp/regen/all_regen.jsonl --out-dir ./data/mtp_26b --eval-n 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen", required=True,
                    help="regen dir (<layer>_regen.jsonl files) or single JSONL")
    ap.add_argument("--out-dir", required=True, help="output dir for train/eval")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--eval-frac", type=float, default=0.1,
                     help="fraction of each layer held out for eval (default 0.1)")
    grp.add_argument("--eval-n", type=int, default=None,
                     help="fixed #eval rows per layer (overrides --eval-frac)")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed (reproducible)")
    ap.add_argument("--only-success", action="store_true", default=True,
                    help="keep only status==success rows (default on)")
    ap.add_argument("--glob", default="*_regen.jsonl",
                    help="glob for per-layer files when --regen is a dir")
    return ap.parse_args()


def layer_of(row, fallback):
    return row.get("source_layer") or fallback


def iter_rows(path, glob):
    """Yield (row, layer_fallback) from a file or a dir of files."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob(glob))
        if not files:
            raise SystemExit(f"no files matching {glob} under {p}")
        for f in files:
            # filename <layer>_regen.jsonl -> <layer>
            fb = f.stem[:-6] if f.stem.endswith("_regen") else f.stem
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield line, fb
    elif p.is_file():
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line, "all"
    else:
        raise SystemExit(f"--regen not found: {p}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # 1. bucket rows by layer
    by_layer: dict[str, list[dict]] = {}
    n_read = n_bad = n_skip = 0
    for line, fb in iter_rows(args.regen, args.glob):
        n_read += 1
        try:
            row = json.loads(line)
        except Exception:
            n_bad += 1
            continue
        if args.only_success and row.get("status") not in (None, "success"):
            n_skip += 1
            continue
        if not row.get("conversations"):
            n_skip += 1
            continue
        lyr = layer_of(row, fb)
        by_layer.setdefault(lyr, []).append(row)

    if not by_layer:
        raise SystemExit("no usable rows found")

    # 2. stratified split per layer
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_maiprofile_26b.jsonl"
    eval_path = out_dir / "eval_maiprofile_26b.jsonl"

    n_train = n_eval = 0
    per_layer = []
    with (train_path.open("w", encoding="utf-8") as tw,
          eval_path.open("w", encoding="utf-8") as ew):
        for lyr in sorted(by_layer):
            rows = by_layer[lyr]
            rng.shuffle(rows)
            if args.eval_n is not None:
                k = min(args.eval_n, len(rows))
            else:
                k = int(round(len(rows) * args.eval_frac))
                # keep at least 1 eval per non-trivial layer, but never all
                k = max(0, min(k, len(rows) - 1)) if len(rows) > 1 else 0
            eval_rows = rows[:k]
            train_rows = rows[k:]
            for r in train_rows:
                tw.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in eval_rows:
                ew.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_train += len(train_rows)
            n_eval += len(eval_rows)
            per_layer.append((lyr, len(train_rows), len(eval_rows)))

    # 3. report
    print("=== random stratified split ===")
    print(f"  rows read     : {n_read:,}")
    print(f"  malformed     : {n_bad:,}")
    print(f"  skipped       : {n_skip:,} (non-success / no conversations)")
    split_desc = (f"eval_n={args.eval_n}/layer" if args.eval_n is not None
                  else f"eval_frac={args.eval_frac}")
    print(f"  split         : {split_desc}  seed={args.seed}")
    print(f"  -> train      : {n_train:,}  -> {train_path}")
    print(f"  -> eval       : {n_eval:,}  -> {eval_path}")
    print("  by layer:")
    for lyr, nt, ne in per_layer:
        print(f"    {lyr:32s} train={nt:8,} eval={ne:6,}")


if __name__ == "__main__":
    main()

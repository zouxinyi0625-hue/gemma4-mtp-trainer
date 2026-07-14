#!/usr/bin/env python3
"""Reproduce DSpark's MAI Profile train/eval split, then map it onto the 26B
regen data (which carries 26B-A4B responses instead of DSpark's 12B output).

We only borrow DSpark's SPLIT (which prompts are train vs eval), not its
responses. DSpark splits purely on the raw prompts, keyed by
    id = f"{layer}:{prompt_hash}"
using random.Random(980406), per-layer shuffle, first --eval-size as eval.

This script:
  1. Reproduces that split from raw_data -> sets of train_ids / eval_ids
     (byte-identical logic to DSpark prepare_maiprofile_splits.py).
  2. INSPECTS the 26B regen file: prints the first few records' keys + a sample
     so we can confirm how to align regen rows back to the split id
     (via prompt_hash / source_layer / conversations).
  3. Optionally (once alignment is confirmed) writes train/eval JSONL using the
     26B responses but DSpark's split membership.

STEP 1 (inspection only) — run this first and share the output:
    python -m gemma4_mtp.build_split \
        --raw-dir   $AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/raw_data/20260615 \
        --regen     $AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/eagle3/20260615/regen_26b/train_all_layers_regen_26b.jsonl \
        --layers short \
        --inspect-only

Do NOT set output paths yet — we decide those after seeing the regen structure.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# DSpark's "short" layer set + exact order (order matters: one shared RNG is
# consumed across layers in sequence). Mirror DSpark's parse_layers("short").
SHORT_LAYERS = [
    "layer1_actual",
    "layer1_delta",
    "layer1_intent",
    "layer2_coarse_interest",
    "layer2_temporal",
]

# Full layer set (used by the eagle3 all-layers split). Only used if
# --layers all is passed; keep in sync with the eagle3 split script.
ALL_LAYERS = [
    "layer1_actual", "layer1_delta", "layer1_intent",
    "layer2_coarse_interest", "layer2_temporal",
    "layer3_commercial_interests", "layer3_persona", "layer3_seasonality",
    "layer4_biography", "layer4_commercial_preference",
]


def parse_layers(raw: str) -> list[str]:
    if raw == "short":
        return list(SHORT_LAYERS)
    if raw == "all":
        return list(ALL_LAYERS)
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_messages(record):
    messages = record.get("prompt_messages") or record.get("conversations") or []
    if not isinstance(messages, list):
        return []
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if not isinstance(role, str):
            continue
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if role not in {"system", "user", "assistant"}:
            continue
        out.append({"role": role, "content": content})
    return out


def read_layer_records(input_dir: Path, layer: str):
    """Byte-identical to DSpark read_layer_records: yields records with the same
    id scheme so the split matches exactly."""
    path = input_dir / f"{layer}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = normalize_messages(record)
            if not messages:
                continue
            if not any(m["role"] == "user" for m in messages):
                continue
            yield {
                "id": f"{layer}:{record.get('prompt_hash') or line_number}",
                "source_layer": layer,
                "prompt_hash": record.get("prompt_hash"),
                "user_id": record.get("user_id"),
                "conversations": messages,
            }


def reproduce_split(raw_dir: Path, layers, seed: int, eval_size: int):
    """Reproduce DSpark's split. Returns (train_ids, eval_ids) sets."""
    rng = random.Random(seed)
    train_ids, eval_ids = set(), set()
    per_layer = {}
    for layer in layers:
        records = list(read_layer_records(raw_dir, layer) or [])
        rng.shuffle(records)
        ev = records[:eval_size]
        tr = records[eval_size:]
        for r in ev:
            eval_ids.add(r["id"])
        for r in tr:
            train_ids.add(r["id"])
        per_layer[layer] = {"raw": len(records), "train": len(tr), "eval": len(ev)}
    # DSpark also shuffles the merged train list, but that only affects ORDER,
    # not membership; membership (which id is train/eval) is what we align on.
    return train_ids, eval_ids, per_layer


def inspect_regen(regen_path: Path, n: int = 3):
    """Print the first n regen records' keys + a compact sample."""
    print(f"\n=== Inspecting regen file: {regen_path} ===", flush=True)
    total = 0
    shown = 0
    keys_seen = set()
    with regen_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            keys_seen.update(rec.keys())
            if shown < n:
                shown += 1
                print(f"\n--- regen record #{shown} ---")
                print("keys:", sorted(rec.keys()))
                for k, v in rec.items():
                    if isinstance(v, str):
                        preview = v[:120] + ("..." if len(v) > 120 else "")
                        print(f"  {k}: str[{len(v)}] {preview!r}")
                    elif isinstance(v, list):
                        print(f"  {k}: list[{len(v)}]")
                        for i, item in enumerate(v[:4]):
                            if isinstance(item, dict):
                                role = item.get("role")
                                c = item.get("content", "")
                                c = c[:80] + ("..." if len(c) > 80 else "")
                                print(f"     [{i}] role={role} content={c!r}")
                            else:
                                print(f"     [{i}] {str(item)[:80]!r}")
                    else:
                        print(f"  {k}: {v!r}")
    print(f"\nregen total records: {total}")
    print(f"regen union of keys : {sorted(keys_seen)}")
    print("\n>>> KEY QUESTION: does each regen record carry a 'prompt_hash' or")
    print(">>> 'source_layer' we can use to map it back to id=f'{layer}:{hash}'?")
    print(">>> If not, we must align by the user/system prompt text instead.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True,
                    help="DSpark raw_data dir (per-layer <layer>.jsonl)")
    ap.add_argument("--regen", required=True,
                    help="26B regen JSONL (train_all_layers_regen_26b.jsonl)")
    ap.add_argument("--layers", default="short", help="short | all | comma list")
    ap.add_argument("--seed", type=int, default=980406, help="DSpark split seed")
    ap.add_argument("--eval-size", type=int, default=200,
                    help="eval samples per layer (DSpark default 200)")
    ap.add_argument("--inspect-only", action="store_true",
                    help="only reproduce split counts + inspect regen structure")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).expanduser()
    regen = Path(args.regen).expanduser()
    layers = parse_layers(args.layers)

    print("=== Reproducing DSpark split ===", flush=True)
    print(f"  seed={args.seed} eval_size={args.eval_size} layers={layers}")
    train_ids, eval_ids, per_layer = reproduce_split(
        raw_dir, layers, args.seed, args.eval_size)
    for layer, c in per_layer.items():
        print(f"  [{layer}] raw={c['raw']} train={c['train']} eval={c['eval']}")
    print(f"  TOTAL train_ids={len(train_ids)} eval_ids={len(eval_ids)}")

    inspect_regen(regen)

    if args.inspect_only:
        print("\n[inspect-only] No files written. Share the regen structure "
              "above and we'll decide the alignment key + output paths.")
        return

    print("\n[note] Alignment/writing not implemented yet — run with "
          "--inspect-only first and confirm the regen structure.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Statistics for MAI Profile raw-data JSONL (DSpark raw-data schema).

Schema (per line, same as DSpark download_and_split.py output):
    {"id": <int|str>, "conversations": [
        {"role": "system"|"user"|"assistant", "content": "<str>"}, ...]}

Reports, with a tqdm progress bar over lines:
  - files scanned, total rows, malformed rows (bad JSON / missing fields)
  - role counts (system/user/assistant/other) and per-conversation turn stats
  - content length distribution (chars, and whitespace-split word count as a
    cheap token proxy), overall and per role
  - conversation shape: first-role distribution, #user turns per row
  - a few structural warnings (empty conversations, first role != user)

USAGE (on the server; the mount env var expands to the AzureML input path):

  # default: scan the 20260616 raw_data dir on the mount
  python -m gemma4_mtp.stat_data

  # explicit dir or file(s):
  python -m gemma4_mtp.stat_data --path /some/dir_or_file.jsonl
  python -m gemma4_mtp.stat_data --path "$AZURE_ML_INPUT_ukwdata/maiprofile/data/raw_data/20260616"

  # optional: exact token counts with a tokenizer (slower)
  python -m gemma4_mtp.stat_data --tokenizer /tmp/models/gemma4/text_only
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def default_dir() -> str | None:
    mount = os.environ.get("AZURE_ML_INPUT_ukwdata")
    if not mount:
        return None
    return str(Path(mount) / "maiprofile/data/raw_data/20260616")


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None,
                    help="raw-data dir or JSONL file. Default: "
                         "$AZURE_ML_INPUT_ukwdata/maiprofile/data/raw_data/20260616")
    ap.add_argument("--glob", default="*.jsonl",
                    help="glob for JSONL files when --path is a dir (default *.jsonl)")
    ap.add_argument("--tokenizer", default=None,
                    help="optional HF tokenizer path/id for exact token counts "
                         "(slower); if unset, uses whitespace word count as proxy")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="stop after N rows (debug)")
    return ap.parse_args()


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _summarize(name, vals):
    if not vals:
        print(f"  {name:22s} (none)")
        return
    s = sorted(vals)
    total = sum(s)
    print(f"  {name:22s} n={len(s):>9,} sum={total:>13,} "
          f"mean={total/len(s):>9.1f} "
          f"min={s[0]:>7,} p50={_pct(s,0.5):>7,} p90={_pct(s,0.9):>7,} "
          f"p99={_pct(s,0.99):>8,} max={s[-1]:>8,}")


def main():
    args = parse_args()
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit("tqdm required: pip install tqdm") from exc

    path = args.path or default_dir()
    if not path:
        raise SystemExit(
            "no --path and AZURE_ML_INPUT_ukwdata not set; pass --path explicitly")
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob(args.glob))
    elif p.is_file():
        files = [p]
    else:
        raise SystemExit(f"path not found: {p}")
    if not files:
        raise SystemExit(f"no files matching {args.glob} under {p}")

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def count_tokens(text: str) -> int:
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False))
        return len(text.split())  # whitespace proxy

    # total line count for the progress bar
    print(f"=== MAI Profile raw-data stats ===")
    print(f"  path : {p}")
    print(f"  files: {len(files)}")
    print("  counting lines for progress bar...", flush=True)
    total_lines = 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for _ in fh:
                total_lines += 1
    print(f"  total lines: {total_lines:,}")

    # accumulators
    n_rows = 0
    n_bad = 0
    role_counts = Counter()
    first_role = Counter()
    turns_per_row = []          # #messages per conversation
    user_turns_per_row = []
    asst_turns_per_row = []
    empty_conv = 0
    first_not_user = 0
    content_len_chars = {"system": [], "user": [], "assistant": [], "other": []}
    content_len_toks = {"system": [], "user": [], "assistant": [], "other": []}
    row_total_toks = []         # sum of tokens across a conversation
    id_seen = set()
    dup_ids = 0

    pbar = tqdm(total=(args.max_rows or total_lines), desc="scanning", unit="row")
    stop = False
    for f in files:
        if stop:
            break
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if args.max_rows is not None and n_rows >= args.max_rows:
                    stop = True
                    break
                try:
                    row = json.loads(line)
                    convs = row["conversations"]
                    assert isinstance(convs, list)
                except Exception:
                    n_bad += 1
                    n_rows += 1
                    pbar.update(1)
                    continue

                n_rows += 1
                rid = row.get("id")
                if rid is not None:
                    if rid in id_seen:
                        dup_ids += 1
                    else:
                        id_seen.add(rid)

                if not convs:
                    empty_conv += 1
                    turns_per_row.append(0)
                    pbar.update(1)
                    continue

                first = convs[0].get("role")
                first_role[first] += 1
                if first != "user":
                    first_not_user += 1

                nu = na = 0
                row_toks = 0
                for m in convs:
                    role = m.get("role", "?")
                    content = m.get("content", "")
                    if not isinstance(content, str):
                        content = str(content)
                    bucket = role if role in content_len_chars else "other"
                    clen = len(content)
                    tlen = count_tokens(content)
                    role_counts[role] += 1
                    content_len_chars[bucket].append(clen)
                    content_len_toks[bucket].append(tlen)
                    row_toks += tlen
                    if role == "user":
                        nu += 1
                    elif role == "assistant":
                        na += 1

                turns_per_row.append(len(convs))
                user_turns_per_row.append(nu)
                asst_turns_per_row.append(na)
                row_total_toks.append(row_toks)
                pbar.update(1)
    pbar.close()

    # ---- report --------------------------------------------------------------
    print("\n=== ROWS ===")
    print(f"  rows scanned      : {n_rows:,}")
    print(f"  malformed rows    : {n_bad:,}")
    print(f"  unique ids        : {len(id_seen):,}  (duplicate ids: {dup_ids:,})")
    print(f"  empty conversation: {empty_conv:,}")
    print(f"  first role != user: {first_not_user:,}")

    print("\n=== ROLE COUNTS (messages) ===")
    for role, c in role_counts.most_common():
        print(f"  {role:12s} {c:>12,}")

    print("\n=== FIRST ROLE (per row) ===")
    for role, c in first_role.most_common():
        print(f"  {role:12s} {c:>12,}")

    print("\n=== TURNS PER ROW ===")
    _summarize("messages/row", turns_per_row)
    _summarize("user turns/row", user_turns_per_row)
    _summarize("assistant turns/row", asst_turns_per_row)

    unit = "tokens" if tok is not None else "words(proxy)"
    print(f"\n=== CONTENT LENGTH — chars ===")
    for role in ("system", "user", "assistant", "other"):
        _summarize(f"{role} chars", content_len_chars[role])
    print(f"\n=== CONTENT LENGTH — {unit} ===")
    for role in ("system", "user", "assistant", "other"):
        _summarize(f"{role} {unit}", content_len_toks[role])
    print(f"\n=== CONVERSATION TOTAL — {unit} ===")
    _summarize(f"row total {unit}", row_total_toks)

    print("\n(done)")


if __name__ == "__main__":
    main()

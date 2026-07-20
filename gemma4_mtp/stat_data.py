#!/usr/bin/env python3
"""Statistics for MAI Profile raw-data JSONL.

Handles both the RAW MAI Profile schema and DSpark's normalized schema:
    raw:        {"user_id":..., "prompt_hash":..., "prompt_messages":[
                    {"role": "system"|"user", "content": "<str>"}, ...]}
                (prompt-only — no assistant answer yet; this is regen input)
    normalized: {"id":..., "conversations":[{"role":..., "content":...}, ...]}

The conversation field is auto-detected (prompt_messages / conversations /
messages); id is prompt_hash or id.

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
                    help="stop after N rows per file (debug)")
    ap.add_argument("--per-file", dest="per_file", action="store_true",
                    default=None,
                    help="report each file (layer) separately + a comparison "
                         "table. Default: ON when scanning a dir of >1 files.")
    ap.add_argument("--combined", dest="per_file", action="store_false",
                    help="force a single combined report over all files.")
    return ap.parse_args()


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _new_stats():
    return {
        "n_rows": 0, "n_bad": 0, "empty_conv": 0, "first_not_user": 0,
        "dup_ids": 0, "id_seen": set(),
        "role_counts": Counter(), "first_role": Counter(),
        "turns_per_row": [], "user_turns_per_row": [], "asst_turns_per_row": [],
        "chars": {"system": [], "user": [], "assistant": [], "other": []},
        "toks": {"system": [], "user": [], "assistant": [], "other": []},
        "row_total_toks": [],
    }


def scan_file(path, count_tokens, pbar, st, max_rows=None):
    """Accumulate stats for one JSONL file into st (see _new_stats)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if max_rows is not None and st["n_rows"] >= max_rows:
                break
            try:
                row = json.loads(line)
                # Field name varies: raw MAI Profile uses "prompt_messages"
                # (prompt-only, no assistant answer yet); DSpark normalized data
                # uses "conversations"; some use "messages".
                convs = (row.get("prompt_messages")
                         or row.get("conversations")
                         or row.get("messages"))
                assert isinstance(convs, list)
            except Exception:
                st["n_bad"] += 1
                st["n_rows"] += 1
                pbar.update(1)
                continue

            st["n_rows"] += 1
            rid = row.get("id") or row.get("prompt_hash")
            if rid is not None:
                if rid in st["id_seen"]:
                    st["dup_ids"] += 1
                else:
                    st["id_seen"].add(rid)

            if not convs:
                st["empty_conv"] += 1
                st["turns_per_row"].append(0)
                pbar.update(1)
                continue

            first = convs[0].get("role")
            st["first_role"][first] += 1
            if first != "user":
                st["first_not_user"] += 1

            nu = na = 0
            row_toks = 0
            for m in convs:
                role = m.get("role", "?")
                content = m.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                bucket = role if role in st["chars"] else "other"
                st["role_counts"][role] += 1
                st["chars"][bucket].append(len(content))
                tlen = count_tokens(content)
                st["toks"][bucket].append(tlen)
                row_toks += tlen
                if role == "user":
                    nu += 1
                elif role == "assistant":
                    na += 1

            st["turns_per_row"].append(len(convs))
            st["user_turns_per_row"].append(nu)
            st["asst_turns_per_row"].append(na)
            st["row_total_toks"].append(row_toks)
            pbar.update(1)
    return st


def report(st, unit, title=None):
    if title:
        print(f"\n########## {title} ##########")
    print("\n=== ROWS ===")
    print(f"  rows scanned      : {st['n_rows']:,}")
    print(f"  malformed rows    : {st['n_bad']:,}")
    print(f"  unique ids        : {len(st['id_seen']):,}  "
          f"(duplicate ids: {st['dup_ids']:,})")
    print(f"  empty conversation: {st['empty_conv']:,}")
    print(f"  first role != user: {st['first_not_user']:,}")

    print("\n=== ROLE COUNTS (messages) ===")
    for role, c in st["role_counts"].most_common():
        print(f"  {role:12s} {c:>12,}")

    print("\n=== FIRST ROLE (per row) ===")
    for role, c in st["first_role"].most_common():
        print(f"  {role:12s} {c:>12,}")

    print("\n=== TURNS PER ROW ===")
    _summarize("messages/row", st["turns_per_row"])
    _summarize("user turns/row", st["user_turns_per_row"])
    _summarize("assistant turns/row", st["asst_turns_per_row"])

    print(f"\n=== CONTENT LENGTH — chars ===")
    for role in ("system", "user", "assistant", "other"):
        _summarize(f"{role} chars", st["chars"][role])
    print(f"\n=== CONTENT LENGTH — {unit} ===")
    for role in ("system", "user", "assistant", "other"):
        _summarize(f"{role} {unit}", st["toks"][role])
    print(f"\n=== CONVERSATION TOTAL — {unit} ===")
    _summarize(f"row total {unit}", st["row_total_toks"])


def _p(vals, q):
    if not vals:
        return 0
    return _pct(sorted(vals), q)


def per_layer_summary(rows, unit):
    """One-line-per-layer comparison table across layers."""
    print(f"\n########## PER-LAYER SUMMARY ({unit}) ##########")
    hdr = (f"{'layer':28s} {'rows':>9s} {'sys_'+unit[:4]:>9s} "
           f"{'usr_p50':>8s} {'usr_p90':>8s} {'usr_p99':>8s} "
           f"{'row_p50':>8s} {'row_p90':>8s} {'row_p99':>8s} {'row_max':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for name, st in rows:
        sys_toks = st["toks"]["system"]
        usr = st["toks"]["user"]
        rt = st["row_total_toks"]
        sysv = int(sum(sys_toks) / len(sys_toks)) if sys_toks else 0
        print(f"{name:28s} {st['n_rows']:>9,} {sysv:>9,} "
              f"{_p(usr,.5):>8,} {_p(usr,.9):>8,} {_p(usr,.99):>8,} "
              f"{_p(rt,.5):>8,} {_p(rt,.9):>8,} {_p(rt,.99):>8,} {_p(rt,1.0):>8,}")



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

    # per-file (per-layer) unless forced combined; default ON for a dir of >1.
    per_file = args.per_file
    if per_file is None:
        per_file = len(files) > 1

    print(f"=== MAI Profile raw-data stats ===")
    print(f"  path : {p}")
    print(f"  files: {len(files)}   mode: {'per-file (per-layer)' if per_file else 'combined'}")
    print("  counting lines for progress bar...", flush=True)
    total_lines = 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for _ in fh:
                total_lines += 1
    print(f"  total lines: {total_lines:,}")

    unit = "tokens" if tok is not None else "words(proxy)"
    pbar = tqdm(total=(args.max_rows and args.max_rows * len(files)) or total_lines,
                desc="scanning", unit="row")

    if per_file:
        per_layer = []
        for f in files:
            st = _new_stats()
            scan_file(f, count_tokens, pbar, st, max_rows=args.max_rows)
            layer = f.stem  # filename without .jsonl == layer name
            report(st, unit, title=f"LAYER: {layer}  ({f.name})")
            per_layer.append((layer, st))
        pbar.close()
        per_layer_summary(per_layer, unit)
    else:
        st = _new_stats()
        for f in files:
            scan_file(f, count_tokens, pbar, st, max_rows=args.max_rows)
        pbar.close()
        report(st, unit)

    print("\n(done)")


if __name__ == "__main__":
    main()

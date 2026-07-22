#!/usr/bin/env python3
"""Read-only confirmation of the step0 train-vs-vLLM divergence candidates.

Bench #3 showed training regresses stock at EVERY position including pos0. Two
line-by-line audits of the training forward (training_step.py) vs the deployed
vLLM forward (gemma4_mtp.py @ 6cbe448ee) surfaced three suspects. This script
gathers the FACTS needed to confirm which are real — it does NOT change anything.

WHAT IT CHECKS
  A. Model configs: sliding_window, final_logit_softcapping, use_ordered_embeddings,
     hidden sizes. (Do the two divergences even apply?)
  B. Anchor→0 distance distribution over the cache: for how many anchors is
     (anchor_pos > sliding_window)? Those are exactly the anchors where the
     trainer's UN-cropped sliding mask lets the draft's sliding layers see the
     WHOLE prefix while vLLM only sees the last `sliding_window` tokens. If this
     fraction is large, suspect #1 (sliding mask) is a real, widespread bug.
  C. Prints the exact step0 position the trainer uses (anchor+1) so you can eyeball
     it against vLLM (audit said anchor). Also dumps a couple sample anchor rows.

USAGE (on the server, in the training venv):
  python confirm_step0.py \
      --target   /tmp/models/gemma4/text_only \
      --assistant /tmp/models/gemma4/assistant \
      --cache-dir "$AZURE_ML_INPUT_ukwdata/maiprofile/mtp_26b/cache" \
      --num-samples 2000

Paste the whole stdout back.
"""
from __future__ import annotations

import argparse
import json
import os


def load_cfg(path):
    """Load a HF config.json (local dir or file), return the dict + text_config."""
    p = path
    if os.path.isdir(path):
        p = os.path.join(path, "config.json")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    # Gemma4 nests the LM params under text_config for the multimodal wrapper.
    text = cfg.get("text_config", cfg)
    return cfg, text


def dump_cfg(name, path):
    print(f"\n===== CONFIG: {name}  ({path}) =====")
    try:
        cfg, text = load_cfg(path)
    except Exception as e:
        print(f"  [ERR] could not load config: {e}")
        return {}
    keys = [
        "hidden_size", "sliding_window", "sliding_window_pattern",
        "final_logit_softcapping", "attn_logit_softcapping",
        "use_ordered_embeddings", "vocab_size", "num_hidden_layers",
        "layer_types", "head_dim", "rope_theta", "query_pre_attn_scalar",
    ]
    picked = {}
    for k in keys:
        if k in text:
            v = text[k]
            # layer_types can be long — summarize
            if k == "layer_types" and isinstance(v, list):
                from collections import Counter
                print(f"  {k:28s} = {dict(Counter(v))}  (len={len(v)})")
            else:
                print(f"  {k:28s} = {v}")
            picked[k] = v
        else:
            print(f"  {k:28s} = <absent>")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=2000,
                    help="how many cache samples to scan for the anchor stats")
    ap.add_argument("--num-anchors", type=int, default=128,
                    help="anchors/seq the trainer samples (for context only)")
    args = ap.parse_args()

    print("############ STEP0 CONFIRMATION (read-only) ############")

    # ---- A. configs ----
    tgt = dump_cfg("TARGET", args.target)
    asst = dump_cfg("ASSISTANT", args.assistant)

    sw = tgt.get("sliding_window") or asst.get("sliding_window")
    print("\n----- divergence applicability -----")
    print(f"  sliding_window (effective) = {sw}")
    softcap = (tgt.get("final_logit_softcapping"),
               asst.get("final_logit_softcapping"))
    print(f"  final_logit_softcapping (target, assistant) = {softcap}")
    print(f"  use_ordered_embeddings (target, assistant) = "
          f"({tgt.get('use_ordered_embeddings')}, {asst.get('use_ordered_embeddings')})")

    # ---- B + C. anchor distance distribution over the cache ----
    print("\n===== ANCHOR STATS (scanning cache) =====")
    try:
        import torch
        from gemma4_mtp.target_cache import CacheDataset
    except Exception as e:
        print(f"  [ERR] cannot import CacheDataset / torch: {e}")
        return

    ds = CacheDataset(args.cache_dir)
    n_total = len(ds)
    n_scan = min(args.num_samples, n_total)
    print(f"  cache: {n_total:,} samples; scanning first {n_scan:,}")
    print(f"  hidden_size={ds.hidden_size}  kv_dims={ds.kv_dims}")

    if sw is None:
        print("  [WARN] sliding_window is None in config — suspect #1 (sliding "
              "mask crop) would NOT apply. Reporting anchor stats anyway.")
        sw_int = None
    else:
        sw_int = int(sw)

    # A position t is a valid anchor iff loss_mask[t]==1 AND loss_mask[t+1]==1
    # (matches training_step.sample_anchors). We count, over all valid anchors:
    #   - anchor position distribution
    #   - fraction with anchor > sliding_window (where the un-cropped sliding
    #     mask diverges from vLLM)
    import numpy as np
    total_valid = 0
    beyond_sw = 0            # anchors with anchor_pos > sliding_window
    anchor_positions = []    # subsample for percentiles
    seq_lens = []
    example_printed = 0

    from tqdm import tqdm
    for i in tqdm(range(n_scan), desc="scan", unit="row"):
        rec = ds[i]
        lm = rec["loss_mask"].to(torch.bool)
        T = lm.numel()
        seq_lens.append(T)
        # valid anchor positions t: lm[t] and lm[t+1]
        valid = lm[:-1] & lm[1:]
        pos = torch.nonzero(valid, as_tuple=False).flatten()
        total_valid += int(pos.numel())
        if sw_int is not None:
            beyond_sw += int((pos > sw_int).sum().item())
        # keep a subsample of positions for percentiles (cap memory)
        if len(anchor_positions) < 200000 and pos.numel():
            anchor_positions.append(pos.numpy())
        # print one concrete example: first valid anchor, its step0 pos the
        # trainer would use (anchor+1) vs what vLLM audit claims (anchor)
        if example_printed < 3 and pos.numel():
            a = int(pos[0].item())
            print(f"\n  [example {example_printed}] seq_len={T}  first_anchor={a}")
            print(f"      trainer step0 position_id = anchor+1 = {a+1} "
                  f"(training_step.py:559)")
            print(f"      vLLM audit says position   = anchor   = {a}   <-- CONFIRM")
            if sw_int is not None:
                print(f"      anchor {'>' if a > sw_int else '<='} sliding_window "
                      f"({sw_int}) -> sliding mask "
                      f"{'DIVERGES' if a > sw_int else 'ok'}")
            example_printed += 1

    print("\n----- anchor position distribution -----")
    print(f"  total valid anchors scanned: {total_valid:,}")
    if seq_lens:
        sl = np.array(seq_lens)
        print(f"  seq_len:   min={sl.min()} p50={int(np.percentile(sl,50))} "
              f"p90={int(np.percentile(sl,90))} p99={int(np.percentile(sl,99))} "
              f"max={sl.max()}")
    if anchor_positions:
        ap_arr = np.concatenate(anchor_positions)
        print(f"  anchor_pos: min={ap_arr.min()} p50={int(np.percentile(ap_arr,50))} "
              f"p90={int(np.percentile(ap_arr,90))} p99={int(np.percentile(ap_arr,99))} "
              f"max={ap_arr.max()}  (subsample n={ap_arr.size:,})")

    if sw_int is not None and total_valid:
        frac = beyond_sw / total_valid
        print("\n----- SUSPECT #1: sliding-window mask crop -----")
        print(f"  sliding_window = {sw_int}")
        print(f"  anchors with anchor_pos > sliding_window: "
              f"{beyond_sw:,} / {total_valid:,} = {frac:.1%}")
        if frac > 0.05:
            print(f"  >>> {frac:.1%} of anchors have the draft's SLIDING layers "
                  f"attending the WHOLE prefix in training, but only the last "
                  f"{sw_int} tokens in vLLM. This is a real, widespread mismatch "
                  f"at step0 -> STRONG regression suspect. <<<")
        else:
            print(f"  >>> Only {frac:.1%} of anchors exceed the window; the "
                  f"sliding-mask bug affects few anchors -> weaker suspect. <<<")

    print("\n############ END — paste all of the above back ############")


if __name__ == "__main__":
    main()

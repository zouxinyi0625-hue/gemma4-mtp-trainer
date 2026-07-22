#!/usr/bin/env python3
"""Prove (empirically) whether the un-cropped sliding mask is the regression root
cause — by running the REAL assistant forward two ways on the SAME anchor and
comparing draft logits. Read-only w.r.t. training; loads models to probe.

confirm_step0.py already established the STATIC fact: 86.6% of anchors sit beyond
sliding_window=1024, and the trainer feeds ONE causal mask (kv_pos<=anchor) to
both full and sliding draft layers (training_step.py:545). What we DON'T yet know
is whether the assistant model INTERNALLY re-crops sliding layers to the window
(in which case the trainer's mask wouldn't matter) or uses the passed mask as-is
(in which case training's sliding layers see context that vLLM's never do → the
root cause).

METHOD (causal test):
  Pick a cached sample whose first valid anchor a > sliding_window. Run ONE step0
  draft forward two ways, everything else identical:
    (A) trainer mask   : sliding sees kv_pos<=a          (WHOLE prefix)
    (B) vLLM semantics : sliding sees a-window<kv_pos<=a (LAST `window` only);
                         full still sees kv_pos<=a
  If logits(A) != logits(B) on a stock (untrained) assistant, the model uses the
  passed mask verbatim and the trainer has been optimizing on context that does
  not exist at inference → sliding-mask crop is the root cause. If logits are
  ~identical, the model self-crops and we must look elsewhere.

This mirrors _assistant_step exactly (concat[embed(t0), hidden], shared_kv_states,
position, attention_mask) so the comparison is apples-to-apples.

USAGE (server, training venv, stock assistant):
  python probe_sliding_rootcause.py \
      --target    /tmp/models/gemma4/text_only \
      --assistant /tmp/models/gemma4/assistant \
      --cache-dir "$AZURE_ML_INPUT_ukwdata/maiprofile/mtp_26b/cache" \
      --num-scan 500
Paste the whole stdout back.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-scan", type=int, default=500,
                    help="scan up to this many cache rows to find beyond-window anchors")
    ap.add_argument("--n-examples", type=int, default=5,
                    help="how many beyond-window anchors to test")
    ap.add_argument("--sliding-window", type=int, default=None,
                    help="override; else read from assistant config")
    args = ap.parse_args()

    import json
    import os
    import torch
    from transformers import AutoModelForCausalLM
    from gemma4_mtp.target_cache import CacheDataset
    from gemma4_mtp.training_step import locate_target_parts

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16

    # sliding window from assistant config
    sw = args.sliding_window
    if sw is None:
        cfgp = os.path.join(args.assistant, "config.json")
        cfg = json.load(open(cfgp))
        tc = cfg.get("text_config", cfg)
        sw = int(tc["sliding_window"])
    print(f"sliding_window = {sw}  device={dev}")

    print("loading target (for embed + lm_head + shared-kv contract) ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=dt, trust_remote_code=True).to(dev).eval()
    _, target_lm_head, _, _ = locate_target_parts(target)
    target_embed = target.get_input_embeddings()

    print("loading STOCK assistant ...", flush=True)
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, torch_dtype=dt, trust_remote_code=True).to(dev).eval()

    ds = CacheDataset(args.cache_dir)
    n = min(args.num_scan, len(ds))
    print(f"scanning up to {n} cache rows for anchors > {sw} ...", flush=True)

    def draft_logits(sample, anchor, crop_sliding):
        """One step0 draft forward, mirroring _assistant_step. If crop_sliding,
        the sliding KV groups get a window-cropped mask; else full prefix. The
        full KV group always gets the full-prefix mask."""
        T = sample["input_ids"].shape[0]
        last_hidden = sample["last_hidden"].to(dev, dt)               # (T,H)
        # shared kv: each (Hkv, T, D) -> add batch dim (1,Hkv,T,D)
        kv = {}
        for name in ("kv_full_k", "kv_full_v", "kv_slide_k", "kv_slide_v"):
            kv[name] = sample[name].to(dev, dt).unsqueeze(0)
        shared = {
            "full_attention": (kv["kv_full_k"], kv["kv_full_v"]),
            "sliding_attention": (kv["kv_slide_k"], kv["kv_slide_v"]),
        }
        kv_pos = torch.arange(T, device=dev)
        full_mask = (kv_pos <= anchor)                                # (T,)
        if crop_sliding:
            slide_mask = (kv_pos <= anchor) & (kv_pos > anchor - sw)  # window
        else:
            slide_mask = full_mask
        # per-type 2D masks (1,T)
        masks = {
            "full_attention": full_mask.to(dt).unsqueeze(0),
            "sliding_attention": slide_mask.to(dt).unsqueeze(0),
        }

        with torch.no_grad():
            anchor_hidden = last_hidden[anchor]                       # (H,)
            t0 = target_lm_head(anchor_hidden.unsqueeze(0)).argmax(-1)  # (1,)
            emb = target_embed(t0.unsqueeze(0))                       # (1,1,H)
            hid = last_hidden[anchor].view(1, 1, -1)                  # (1,1,H)
            combined = torch.cat([emb, hid], dim=-1)                  # (1,1,2H)
            pos = torch.tensor([[min(anchor + 1, T - 1)]], device=dev)
            # Try per-type dict mask first (what a window-crop fix would need);
            # fall back to a single 2D mask (current trainer behavior) if the
            # model doesn't accept a dict.
            try:
                out = assistant(inputs_embeds=combined, shared_kv_states=shared,
                                position_ids=pos, attention_mask=masks)
                mask_mode = "dict(per-type)"
            except Exception:
                out = assistant(inputs_embeds=combined, shared_kv_states=shared,
                                position_ids=pos,
                                attention_mask=masks["sliding_attention"])
                mask_mode = "single-2D(sliding)"
        return out.logits.float().view(-1), mask_mode

    tested = 0
    diffs = []
    for i in range(n):
        s = ds[i]
        lm = s["loss_mask"].to(torch.bool)
        valid = lm[:-1] & lm[1:]
        pos = torch.nonzero(valid).flatten()
        pos = pos[pos > sw]
        if pos.numel() == 0:
            continue
        a = int(pos[0].item())
        T = s["input_ids"].shape[0]

        lg_full, mm = draft_logits(s, a, crop_sliding=False)   # trainer behavior
        lg_crop, _ = draft_logits(s, a, crop_sliding=True)     # vLLM semantics

        cos = torch.nn.functional.cosine_similarity(
            lg_full.unsqueeze(0), lg_crop.unsqueeze(0)).item()
        maxabs = (lg_full - lg_crop).abs().max().item()
        arg_full = int(lg_full.argmax()); arg_crop = int(lg_crop.argmax())
        agree = (arg_full == arg_crop)
        diffs.append((cos, maxabs, agree))
        print(f"\n[{tested}] row={i} seq_len={T} anchor={a} (>{sw}) mask_mode={mm}")
        print(f"    cosine(full,crop) = {cos:.6f}   max|Δlogit| = {maxabs:.4f}")
        print(f"    argmax full={arg_full}  crop={arg_crop}  "
              f"{'AGREE' if agree else 'DISAGREE <-- greedy token flips'}")
        tested += 1
        if tested >= args.n_examples:
            break

    print("\n==================== VERDICT ====================")
    if not diffs:
        print("  No beyond-window anchors found in the scan window — increase --num-scan.")
        return
    import statistics as st
    coss = [d[0] for d in diffs]
    disagreements = sum(1 for d in diffs if not d[2])
    print(f"  tested anchors        : {len(diffs)}")
    print(f"  mean cosine(full,crop): {st.mean(coss):.6f}  min: {min(coss):.6f}")
    print(f"  greedy-token flips    : {disagreements}/{len(diffs)}")
    if min(coss) < 0.999 or disagreements > 0:
        print("  >>> The un-cropped sliding mask CHANGES the draft output. The "
              "assistant uses the passed mask verbatim (no internal re-crop), so "
              "training's sliding layers attend context vLLM never sees. "
              "ROOT CAUSE CONFIRMED: crop sliding mask to the window. <<<")
    else:
        print("  >>> logits ~identical: the model self-crops sliding layers; the "
              "trainer mask is NOT the cause. Look elsewhere (position anchor+1 vs "
              "anchor, or the target-hidden/token pairing). <<<")


if __name__ == "__main__":
    main()

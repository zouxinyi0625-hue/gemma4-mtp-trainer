#!/usr/bin/env python3
"""Two clean step0 causal probes, judged by the SAME rule vLLM uses for greedy
accept: does draft_argmax flip? cosine/max|Δ| are diagnostics; the token flip is
the verdict. Read-only w.r.t. training; loads models to probe.

WHY: probe_sliding_rootcause.py showed the un-cropped sliding mask barely moves
draft logits (cosine ~0.996–0.99998) and NEVER flips the greedy token (0/5). But
that probe was contaminated: the assistant only accepts a SINGLE 2D attention
mask (no per-type dict), so cropping "sliding" also cropped the full layer — not
apples-to-apples. So we still have two open suspects, tested cleanly here:

  PROBE A — position anchor+1 vs anchor (RoPE phase).
    training_step.py:559 uses pos = anchor+1. The vLLM audit said the deployed
    forward uses pos = anchor (constant, does not advance). A one-slot RoPE shift
    changes every layer's attention. Run step0 twice, pos=anchor+1 vs pos=anchor,
    everything else fixed; does draft_argmax flip?

  PROBE B — sliding mask, CLEANLY isolated.
    Instead of relying on a per-type mask the model won't accept, we crop the
    SLIDING KV tensors themselves: zero out (mask) the sliding-KV positions that
    fall outside the window (kv_pos <= anchor-sw), leave the FULL KV untouched,
    then run with the full-prefix 2D mask. This makes the sliding layers unable
    to attend beyond-window context while the full layer is byte-identical to the
    trainer path. Compare draft_argmax: full sliding-KV vs window-cropped
    sliding-KV. This is the isolation the first probe lacked.

USAGE (server, training venv, stock assistant):
  python probe_step0_causes.py \
      --target    /tmp/models/gemma4/text_only \
      --assistant /tmp/models/gemma4/assistant \
      --cache-dir "$AZURE_ML_INPUT_ukwdata/maiprofile/mtp_26b/cache" \
      --num-scan 500 --n-examples 8
Paste the whole stdout back.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-scan", type=int, default=500)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--sliding-window", type=int, default=None)
    args = ap.parse_args()

    import json
    import os
    import statistics as st
    import torch
    from transformers import AutoModelForCausalLM
    from gemma4_mtp.target_cache import CacheDataset
    from gemma4_mtp.training_step import locate_target_parts

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16

    sw = args.sliding_window
    if sw is None:
        cfg = json.load(open(os.path.join(args.assistant, "config.json")))
        tc = cfg.get("text_config", cfg)
        sw = int(tc["sliding_window"])
    print(f"sliding_window = {sw}  device={dev}")

    print("loading target ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dt, trust_remote_code=True).to(dev).eval()
    _, target_lm_head, _, _ = locate_target_parts(target)
    target_embed = target.get_input_embeddings()

    print("loading STOCK assistant ...", flush=True)
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=dt, trust_remote_code=True).to(dev).eval()

    ds = CacheDataset(args.cache_dir)
    n = min(args.num_scan, len(ds))

    def build_inputs(sample, anchor):
        """Shared step0 tensors (t0, combined embed, full-prefix 2D mask)."""
        T = sample["input_ids"].shape[0]
        last_hidden = sample["last_hidden"].to(dev, dt)
        kv = {name: sample[name].to(dev, dt).unsqueeze(0)
              for name in ("kv_full_k", "kv_full_v", "kv_slide_k", "kv_slide_v")}
        kv_pos = torch.arange(T, device=dev)
        full_mask = (kv_pos <= anchor).to(dt).unsqueeze(0)          # (1,T)
        with torch.no_grad():
            ah = last_hidden[anchor]
            t0 = target_lm_head(ah.unsqueeze(0)).argmax(-1)
            emb = target_embed(t0.unsqueeze(0))                     # (1,1,H)
            hid = last_hidden[anchor].view(1, 1, -1)
            combined = torch.cat([emb, hid], dim=-1)                # (1,1,2H)
        return T, kv, kv_pos, full_mask, combined

    def run(combined, kv, full_mask, pos_val, crop_slide, anchor, kv_pos):
        """One step0 forward. If crop_slide, zero the sliding-KV positions beyond
        the window (<= anchor-sw); full KV untouched. Full-prefix 2D mask always."""
        kfk, kfv = kv["kv_full_k"], kv["kv_full_v"]
        ksk, ksv = kv["kv_slide_k"], kv["kv_slide_v"]
        if crop_slide:
            # sliding attends only (anchor-sw, anchor]; kill earlier positions.
            keep = (kv_pos > anchor - sw)                           # (T,)
            m = keep.view(1, 1, -1, 1).to(dt)                       # (1,1,T,1)
            ksk = ksk * m
            ksv = ksv * m
        shared = {"full_attention": (kfk, kfv),
                  "sliding_attention": (ksk, ksv)}
        pos = torch.tensor([[pos_val]], device=dev)
        with torch.no_grad():
            out = assistant(inputs_embeds=combined, shared_kv_states=shared,
                            position_ids=pos, attention_mask=full_mask)
        return out.logits.float().view(-1)

    def compare(lg_a, lg_b):
        cos = torch.nn.functional.cosine_similarity(
            lg_a.unsqueeze(0), lg_b.unsqueeze(0)).item()
        mx = (lg_a - lg_b).abs().max().item()
        return cos, mx, int(lg_a.argmax()), int(lg_b.argmax())

    A_res, B_res = [], []
    tested = 0
    for i in range(n):
        s = ds[i]
        lm = s["loss_mask"].to(torch.bool)
        valid = lm[:-1] & lm[1:]
        pos = torch.nonzero(valid).flatten()
        pos = pos[pos > sw]           # beyond-window anchors (where both suspects bite)
        if pos.numel() == 0:
            continue
        a = int(pos[0].item())
        T, kv, kv_pos, full_mask, combined = build_inputs(s, a)

        # PROBE A: pos anchor+1 vs anchor (no sliding crop, identical KV)
        lg_p1 = run(combined, kv, full_mask, min(a + 1, T - 1), False, a, kv_pos)
        lg_p0 = run(combined, kv, full_mask, a,                  False, a, kv_pos)
        cosA, mxA, argA1, argA0 = compare(lg_p1, lg_p0)
        A_res.append((cosA, mxA, argA1 == argA0))

        # PROBE B: sliding-KV cropped vs not, position held at trainer's anchor+1
        lg_uncrop = run(combined, kv, full_mask, min(a + 1, T - 1), False, a, kv_pos)
        lg_crop   = run(combined, kv, full_mask, min(a + 1, T - 1), True,  a, kv_pos)
        cosB, mxB, argBu, argBc = compare(lg_uncrop, lg_crop)
        B_res.append((cosB, mxB, argBu == argBc))

        print(f"\n[{tested}] row={i} seq_len={T} anchor={a} (>{sw})")
        print(f"  A pos {a+1} vs {a}: cos={cosA:.6f} max|Δ|={mxA:.3f} "
              f"argmax {argA1}/{argA0} "
              f"{'AGREE' if argA1==argA0 else 'FLIP <-- position changes greedy token'}")
        print(f"  B slide-KV uncrop vs crop: cos={cosB:.6f} max|Δ|={mxB:.3f} "
              f"argmax {argBu}/{argBc} "
              f"{'AGREE' if argBu==argBc else 'FLIP <-- sliding crop changes greedy token'}")
        tested += 1
        if tested >= args.n_examples:
            break

    def verdict(name, res, why):
        if not res:
            print(f"\n{name}: no anchors tested."); return
        coss = [r[0] for r in res]
        flips = sum(1 for r in res if not r[2])
        print(f"\n{name}: tested={len(res)}  mean_cos={st.mean(coss):.6f} "
              f"min_cos={min(coss):.6f}  greedy_flips={flips}/{len(res)}")
        if flips > 0:
            print(f"  >>> {flips}/{len(res)} greedy tokens FLIP -> {why} IS a real "
                  f"step0 cause of the accept drop. <<<")
        else:
            print(f"  >>> 0 flips -> {why} does NOT change the greedy accept token "
                  f"at step0; not the (main) cause.")

    print("\n==================== VERDICT ====================")
    verdict("PROBE A (position anchor+1 vs anchor)", A_res,
            "the position-off-by-one")
    verdict("PROBE B (sliding-mask crop, isolated)", B_res,
            "the sliding-mask crop")
    print("\nNote: cosine/max|Δ| are diagnostics; the FLIP count is the verdict, "
          "because vLLM greedy accept == (draft_argmax == target_argmax). A cause "
          "that never flips the argmax at step0 cannot explain the pos0 drop.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split the root cause into exactly two possibilities, using the SAME greedy
accept rule vLLM uses (draft_argmax == target_argmax). Read-only; loads models.

Position and sliding-mask are both ruled out (probe_step0_causes: 0 flips, and
the sliding crop changed logits by exactly 0.0 -> the model self-crops sliding
layers). So the remaining question is upstream: is the TRAINING FORWARD itself
mis-aligned with vLLM, or does the TRAINING OBJECTIVE damage the weights?

This probe runs the trainer's OWN step0 forward (t0 = target argmax at anchor,
combined=[embed(t0), hidden@anchor], full-prefix mask, pos anchor+1 — exactly
training_step_from_cache) and computes, per anchor, the greedy-accept hit
(draft_argmax == target_argmax_at_anchor+1) for BOTH the stock assistant and the
trained checkpoint. target_argmax uses the target lm_head WITH final_logit_
softcapping (=30) applied, to match how vLLM samples the token being matched.

INTERPRETATION
  stock_hit  ~= bench stock pos0 (~0.85-0.90 on the good layers), trained_hit
               lower  ->  the training forward is ALIGNED with vLLM; the drop is
               the OBJECTIVE damaging weights. Fix = training objective/lr/steps.
  stock_hit  MUCH LOWER than bench stock pos0  ->  the training forward itself
               disagrees with vLLM even on stock weights; there is still a
               forward-implementation bug (embed scale, token/hidden pairing, KV
               semantics, target-argmax reference) to hunt. The specific per-part
               dumps below (embed norm, |combined|, target-vs-independent argmax)
               tell you which.

USAGE (server, training venv):
  python probe_weight_vs_forward.py \
      --target    /tmp/models/gemma4/text_only \
      --stock     /tmp/models/gemma4/assistant \
      --trained   "$(ls -td $AZURE_ML_INPUT_ukwdata/maiprofile/mtp_26b/checkpoints/*/ | head -1)" \
      --cache-dir "$AZURE_ML_INPUT_ukwdata/maiprofile/mtp_26b/cache" \
      --num-scan 1000 --n-examples 300
Paste the whole stdout back. If --trained resolves wrong, pass it explicitly.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-scan", type=int, default=1000)
    ap.add_argument("--n-examples", type=int, default=300,
                    help="total anchors to score (across rows)")
    ap.add_argument("--anchors-per-row", type=int, default=4,
                    help="random answer anchors sampled per row")
    ap.add_argument("--skip-head", type=int, default=8,
                    help="skip the first N valid anchors per row — they land on the "
                         "chat-template header (<start_of_turn>model\\n), a fixed "
                         "token transition every sample shares (t0=3723, tgt_next=107). "
                         "Scoring those gives a meaningless 100%% hit. Real answer "
                         "content starts after the header.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--softcap", type=float, default=30.0,
                    help="target final_logit_softcapping (matches vLLM sampling)")
    args = ap.parse_args()

    import os
    import random
    import torch
    from transformers import AutoModelForCausalLM
    from gemma4_mtp.target_cache import CacheDataset
    from gemma4_mtp.training_step import locate_target_parts

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16
    print(f"device={dev}")
    print(f"trained checkpoint = {args.trained}")

    def softcap(logits, cap):
        if not cap or cap <= 0:
            return logits
        return torch.tanh(logits / cap) * cap

    print("loading target ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dt, trust_remote_code=True).to(dev).eval()
    _, target_lm_head, _, _ = locate_target_parts(target)
    target_embed = target.get_input_embeddings()

    print("loading STOCK assistant ...", flush=True)
    stock = AutoModelForCausalLM.from_pretrained(
        args.stock, dtype=dt, trust_remote_code=True).to(dev).eval()
    print("loading TRAINED assistant ...", flush=True)
    trained = AutoModelForCausalLM.from_pretrained(
        args.trained, dtype=dt, trust_remote_code=True).to(dev).eval()

    ds = CacheDataset(args.cache_dir)
    n = min(args.num_scan, len(ds))
    rng = random.Random(args.seed)

    def step0(assistant, combined, shared, pos, mask):
        with torch.no_grad():
            out = assistant(inputs_embeds=combined, shared_kv_states=shared,
                            position_ids=pos, attention_mask=mask)
        return out.logits.float().view(-1)

    # counters
    stock_hit = trained_hit = 0
    both = 0
    stock_eq_trained = 0     # draft argmax agreement stock vs trained
    embed_norm_sum = 0.0
    combined_norm_sum = 0.0
    tested = 0

    for i in range(n):
        if tested >= args.n_examples:
            break
        s = ds[i]
        T = s["input_ids"].shape[0]
        lm = s["loss_mask"].to(torch.bool)
        valid = lm[:-1] & lm[1:]
        pos_ids = torch.nonzero(valid).flatten()
        # Skip the header anchors (first --skip-head valid positions land on the
        # fixed chat-template transition); sample real answer-content anchors.
        pos_ids = pos_ids[args.skip_head:]
        pos_ids = pos_ids[pos_ids < T - 1]
        if pos_ids.numel() == 0:
            continue
        k = min(args.anchors_per_row, pos_ids.numel())
        chosen = rng.sample(pos_ids.tolist(), k)

        last_hidden = s["last_hidden"].to(dev, dt)
        kv = {name: s[name].to(dev, dt).unsqueeze(0)
              for name in ("kv_full_k", "kv_full_v", "kv_slide_k", "kv_slide_v")}
        shared = {"full_attention": (kv["kv_full_k"], kv["kv_full_v"]),
                  "sliding_attention": (kv["kv_slide_k"], kv["kv_slide_v"])}
        kv_pos = torch.arange(T, device=dev)

        for a in chosen:
            if tested >= args.n_examples:
                break
            mask = (kv_pos <= a).to(dt).unsqueeze(0)
            pos = torch.tensor([[min(a + 1, T - 1)]], device=dev)

            with torch.no_grad():
                # t0 = target argmax at anchor (what vLLM feeds); softcapped.
                ah = last_hidden[a]
                t0 = softcap(target_lm_head(ah.unsqueeze(0)), args.softcap).argmax(-1)
                emb = target_embed(t0.unsqueeze(0))          # (1,1,H) scaled inside
                hid = last_hidden[a].view(1, 1, -1)
                combined = torch.cat([emb, hid], dim=-1)
                embed_norm_sum += emb.float().norm().item()
                combined_norm_sum += combined.float().norm().item()

                # accept reference = target argmax at position anchor+1.
                tgt_next = softcap(
                    target_lm_head(last_hidden[a + 1].unsqueeze(0)),
                    args.softcap).argmax(-1).item()

            ls = step0(stock, combined, shared, pos, mask)
            lt = step0(trained, combined, shared, pos, mask)
            as_ = int(ls.argmax()); at_ = int(lt.argmax())

            sh = (as_ == tgt_next); th = (at_ == tgt_next)
            stock_hit += sh; trained_hit += th
            both += (sh and th)
            stock_eq_trained += (as_ == at_)
            tested += 1

            if tested <= 12:
                print(f"[{tested}] row={i} anchor={a} t0={int(t0.item())} "
                      f"tgt_next={tgt_next}  stock={as_}({'hit' if sh else 'miss'})  "
                      f"trained={at_}({'hit' if th else 'miss'})")

    print("\n==================== RESULTS ====================")
    print(f"  anchors scored: {tested}")
    if tested == 0:
        print("  none scored — increase --num-scan"); return
    sh_r = stock_hit / tested
    th_r = trained_hit / tested
    print(f"  STOCK   greedy-accept hit @ step0 (via TRAINING forward): {sh_r:.4f}")
    print(f"  TRAINED greedy-accept hit @ step0 (via TRAINING forward): {th_r:.4f}")
    print(f"  stock==trained draft argmax agreement: {stock_eq_trained/tested:.4f}")
    print(f"  mean |embed(t0)|={embed_norm_sum/tested:.2f}  "
          f"mean |combined|={combined_norm_sum/tested:.2f}")
    print("\n----- interpretation -----")
    print("  Compare STOCK hit above against the bench stock pos0 (~0.85-0.90 on")
    print("  the strong layers; the eval here is mixed-layer so expect a blend).")
    print("  * STOCK hit ~matches bench stock pos0, TRAINED lower  -> training")
    print("    forward is ALIGNED; the OBJECTIVE is damaging weights. Fix training")
    print("    (lr/steps/objective), not the forward.")
    print("  * STOCK hit MUCH BELOW bench stock pos0  -> the training forward")
    print("    disagrees with vLLM even on stock weights: a forward-impl bug")
    print("    remains (embed scale, token/hidden pairing, KV semantics, or the")
    print("    target-argmax reference). Use the norms + agreement above to localize.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Confirm (or refute) that AUTOREGRESSIVE self-feeding at draft steps k>=1 is
what damages training — by measuring per-step greedy-accept hit under two feeding
regimes, for stock vs trained. Read-only; loads models.

Established so far:
  - training forward is aligned with vLLM (stock step0 hit 0.84 == bench stock pos0)
  - training does NOT damage step0 (trained step0 hit 0.8367 == stock)
  - position off-by-one and sliding-mask are ruled out (0 argmax flips)
The remaining structural difference from the WORKING DSpark trainer:
  - DSpark trains TEACHER-FORCED / parallel: draft step k consumes a mask token +
    the target's ground-truth context; never its own step k-1 output.
  - OUR trainer FREE-RUNS: training_step.py:566-567 feeds the draft's OWN argmax
    (tok_k) and OWN backbone_hidden (hid) into step k+1.
Hypothesis: free-running makes step>=1 fit garbage inputs, and the resulting
gradients corrupt the shared draft weights, dragging down every position.

THIS PROBE runs the draft autoregressively for K steps two ways, on the SAME
answer anchors, for BOTH stock and trained, and reports per-step hit rate:
  regime TF  (teacher-forced, DSpark-style): at step k feed GT token @ anchor+k
             (= input_ids[anchor+k]) and GT hidden = last_hidden[anchor+k].
  regime AR  (autoregressive, our training): at step k feed the draft's own
             argmax from step k-1 and its own backbone_hidden (exactly
             training_step.py:566-567). step0 identical in both.
Accept ref at step k = target argmax @ position anchor+k+1 (softcapped like vLLM).

READS (per step): does trained's AR hit collapse at k>=1 while TF holds? Then
free-running is the culprit and the DSpark-style teacher-forced fix is justified.
Also: stock AR per-step ~ bench per-position (bench is autoregressive) — a sanity
tie to the real numbers.

USAGE:
  python probe_autoregressive.py \
      --target /tmp/models/gemma4/text_only \
      --stock  /tmp/models/gemma4/assistant \
      --trained "/scratch/.../checkpoints/20260721_022119/epoch0" \
      --cache-dir "$AZURE_ML_INPUT_ukwdata/maiprofile/mtp_26b/cache" \
      --num-scan 1500 --anchors-per-row 4 --n-examples 400 --steps 5
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-scan", type=int, default=1500)
    ap.add_argument("--anchors-per-row", type=int, default=4)
    ap.add_argument("--n-examples", type=int, default=400)
    ap.add_argument("--skip-head", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--softcap", type=float, default=30.0)
    args = ap.parse_args()

    import random
    import torch
    from transformers import AutoModelForCausalLM
    from gemma4_mtp.target_cache import CacheDataset
    from gemma4_mtp.training_step import locate_target_parts

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16
    K = args.steps

    def softcap(x, c):
        return x if not c or c <= 0 else torch.tanh(x / c) * c

    print("loading target ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dt, trust_remote_code=True).to(dev).eval()
    _, target_lm_head, _, _ = locate_target_parts(target)
    target_embed = target.get_input_embeddings()
    print("loading stock ...", flush=True)
    stock = AutoModelForCausalLM.from_pretrained(
        args.stock, dtype=dt, trust_remote_code=True).to(dev).eval()
    print("loading trained ...", flush=True)
    trained = AutoModelForCausalLM.from_pretrained(
        args.trained, dtype=dt, trust_remote_code=True).to(dev).eval()

    ds = CacheDataset(args.cache_dir)
    n = min(args.num_scan, len(ds))
    rng = random.Random(args.seed)

    def draft_step(assistant, tok, hid, shared, pos, mask):
        with torch.no_grad():
            emb = target_embed(tok)                       # (1,1,H) scaled inside
            combined = torch.cat([emb, hid], dim=-1)      # (1,1,2H)
            out = assistant(inputs_embeds=combined, shared_kv_states=shared,
                            position_ids=pos, attention_mask=mask)
        return out.logits.float().view(-1), out.last_hidden_state  # (V,), (1,1,H)

    def run_chain(assistant, s, a, T, last_hidden, shared, kv_pos, regime):
        """Autoregressively run K draft steps from anchor a. Returns list of
        (hit_k) booleans, k=0..K-1. regime in {TF, AR}."""
        mask = (kv_pos <= a).to(dt).unsqueeze(0)
        pos = torch.tensor([[min(a + 1, T - 1)]], device=dev)
        # step0 input = t0 = target argmax @ anchor (both regimes identical)
        with torch.no_grad():
            t0 = softcap(target_lm_head(last_hidden[a].unsqueeze(0)),
                         args.softcap).argmax(-1)
        tok = t0.unsqueeze(0)                             # (1,1)
        hid = last_hidden[a].view(1, 1, -1)              # (1,1,H)
        hits = []
        for k in range(K):
            idx = a + k + 1
            if idx > T - 1:
                break
            lg, bh = draft_step(assistant, tok, hid, shared, pos, mask)
            with torch.no_grad():
                tgt = softcap(target_lm_head(last_hidden[idx].unsqueeze(0)),
                              args.softcap).argmax(-1).item()
            hits.append(int(lg.argmax()) == tgt)
            # prepare next step's input
            if regime == "AR":
                tok = lg.argmax().view(1, 1)             # draft's own argmax
                hid = bh                                  # draft's own hidden
            else:  # TF (DSpark-style): GT token + GT hidden at anchor+k+1
                if idx + 1 > T - 1:
                    break
                tok = s["input_ids"][idx].view(1, 1).to(dev)
                hid = last_hidden[idx].view(1, 1, -1)
        return hits

    # counters: [model][regime][k] -> (hit, total)
    import collections
    cnt = collections.defaultdict(lambda: [0, 0])
    tested = 0
    for i in range(n):
        if tested >= args.n_examples:
            break
        s = ds[i]
        T = s["input_ids"].shape[0]
        lm = s["loss_mask"].to(torch.bool)
        valid = lm[:-1] & lm[1:]
        pv = torch.nonzero(valid).flatten()[args.skip_head:]
        pv = pv[pv < T - 1 - K]
        if pv.numel() == 0:
            continue
        chosen = rng.sample(pv.tolist(), min(args.anchors_per_row, pv.numel()))
        last_hidden = s["last_hidden"].to(dev, dt)
        kv = {name: s[name].to(dev, dt).unsqueeze(0)
              for name in ("kv_full_k", "kv_full_v", "kv_slide_k", "kv_slide_v")}
        shared = {"full_attention": (kv["kv_full_k"], kv["kv_full_v"]),
                  "sliding_attention": (kv["kv_slide_k"], kv["kv_slide_v"])}
        kv_pos = torch.arange(T, device=dev)
        for a in chosen:
            if tested >= args.n_examples:
                break
            for mname, model in (("stock", stock), ("trained", trained)):
                for regime in ("TF", "AR"):
                    hits = run_chain(model, s, a, T, last_hidden, shared, kv_pos, regime)
                    for k, h in enumerate(hits):
                        c = cnt[(mname, regime, k)]
                        c[0] += int(h); c[1] += 1
            tested += 1

    print(f"\n==================== RESULTS (anchors={tested}) ====================")
    print(f"{'step':>4} | {'stock TF':>9} {'stock AR':>9} | "
          f"{'train TF':>9} {'train AR':>9}")
    print("-" * 54)
    for k in range(K):
        def rate(m, r):
            c = cnt[(m, r, k)]
            return f"{c[0]/c[1]:.3f}" if c[1] else "  -  "
        print(f"{k:>4} | {rate('stock','TF'):>9} {rate('stock','AR'):>9} | "
              f"{rate('trained','TF'):>9} {rate('trained','AR'):>9}")
    print("\n----- interpretation -----")
    print("  * stock AR ~ bench per-position (bench is autoregressive) — sanity tie.")
    print("  * If trained AR collapses at k>=1 vs trained TF (and vs stock AR),")
    print("    free-running IS the damage path -> switch training to teacher-forced")
    print("    (DSpark-style: feed GT token + GT hidden at step k, parallel).")
    print("  * If trained TF also drops vs stock TF, the objective itself hurts")
    print("    even teacher-forced — a deeper problem than the feeding regime.")


if __name__ == "__main__":
    main()

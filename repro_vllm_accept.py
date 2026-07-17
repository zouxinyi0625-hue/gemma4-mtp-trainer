#!/usr/bin/env python3
"""Standalone reproduction of vLLM's Gemma4 MTP spec-decoding acceptance rate.

Goal: reproduce the vLLM bench "Acceptance rate %" / per-position / accept-len
numbers WITHOUT running a vLLM server, using the same target + assistant +
eval prompts. If this matches the bench (e.g. ~76% for stock, ~23% for a
mistrained draft), we have provably understood vLLM's accept rule and can use
this for fast iteration.

Faithful to vLLM @ 6cbe448ee:
  DRAFT loop (single-anchor, SinglePositionMultiTokenCandidateGenerator):
    - at each target decode position (anchor), draft k tokens autoregressively
    - position_ids fixed = anchor; shared_kv_states constant (target's KV)
    - step0 token = target's sampled token; hidden = target last_hidden@anchor
    - step j>0 token = draft's own argmax; hidden = draft's backbone_hidden
    (this recipe is verify_parity-proven bit-exact to the official generator)
  ACCEPT rule:
    - GREEDY (rejection_greedy_sample_kernel:744): accepted <=> draft_token ==
      argmax(target_logits at that position). Chained: first mismatch stops.
    - RANDOM (rejection_random_sample_kernel:810): accepted <=>
      target_prob[draft_token]/draft_prob[draft_token] >= uniform. Chained.
  METRICS (metrics.py:88-98):
    - accept_rate = accepted / total_draft_tokens
    - accept_len  = 1 + accepted / num_drafts
    - per_pos[k]  = accepted_at_pos_k / num_drafts   (unconditional, prefix)

Run:
  python repro_vllm_accept.py \
      --target /tmp/models/gemma4/text_only \
      --assistant /tmp/models/gemma4/assistant \
      --data ./data/mtp_short/eval_maiprofile_short_26b.jsonl \
      --num-samples 100 --spec-tokens 5 --max-gen 128 --mode greedy --bf16
"""
from __future__ import annotations

import argparse
import json


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--data", required=True, help="eval JSONL (conversations)")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--spec-tokens", type=int, default=5, help="k draft tokens/step")
    ap.add_argument("--max-gen", type=int, default=128, help="max tokens generated per prompt")
    ap.add_argument("--mode", choices=["greedy", "random"], default="greedy",
                    help="accept rule: greedy=argmax-match (temperature 0), "
                         "random=rejection sampling (temperature>0)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="target sampling temperature for --mode random")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    import math
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = args.device
    k = args.spec_tokens

    print("=== Loading models ===", flush=True)
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, device_map=device, trust_remote_code=True).eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=dtype, device_map=device, trust_remote_code=True).eval()

    target_base = getattr(target, "model", target)
    target_lm_head = getattr(target, "lm_head", None)
    target_embed = target.get_input_embeddings()

    def target_step(ids):
        """Full-seq target forward -> (last_hidden (1,T,H), shared_kv, logits (1,T,V))."""
        out = target_base(input_ids=ids, return_shared_kv_states=True, use_cache=False)
        lh = out.last_hidden_state
        skv = out.shared_kv_states
        logits = target_lm_head(lh) if target_lm_head is not None else \
            target(input_ids=ids, use_cache=False).logits
        return lh, skv, logits

    def draft_propose(t0_id, hidden0, shared_kv, anchor_pos):
        """k draft tokens (single-anchor, verify_parity-proven recipe).

        Returns (draft_ids list, draft_probs list) — probs only needed for
        --mode random.
        """
        pos = torch.tensor([[anchor_pos]], device=device)     # constant
        drafts, dprobs = [], []
        hidden = hidden0                                       # (1,1,H)
        tok_id = t0_id                                         # (1,1)
        for _ in range(k):
            tok_embed = target_embed(tok_id)                  # (1,1,H) NO normalizer
            combined = torch.cat([tok_embed, hidden], dim=-1)  # (1,1,2H)
            out = assistant(inputs_embeds=combined, shared_kv_states=shared_kv,
                            position_ids=pos, attention_mask=None)
            hidden = out.last_hidden_state                     # (1,1,H) backbone
            dl = out.logits[:, -1, :]                          # (1,V)
            nxt = dl.argmax(dim=-1, keepdim=True)              # (1,1) draft argmax
            drafts.append(nxt)
            if args.mode == "random":
                dprobs.append(torch.softmax(dl.float(), dim=-1))  # (1,V)
            tok_id = nxt
        return drafts, dprobs

    # --- load prompts ---
    def iter_prompts(path, n):
        got = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("status") not in (None, "success"):
                    continue
                conv = obj.get("conversations")
                if not conv:
                    continue
                # build prompt up to first assistant turn (like the bench)
                msgs = []
                for m in conv:
                    if m.get("role") == "assistant":
                        break
                    msgs.append(m)
                if not msgs:
                    continue
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
                yield text
                got += 1
                if got >= n:
                    return

    print(f"=== Reproducing vLLM accept ({args.mode}), k={k} ===", flush=True)
    total_drafts = 0
    total_draft_tokens = 0
    total_accepted = 0
    pos_accept = [0] * k

    with torch.no_grad():
        for si, text in enumerate(iter_prompts(args.data, args.num_samples)):
            ids = tok(text, return_tensors="pt").input_ids.to(device)
            prompt_len = ids.size(1)
            gen = 0
            while gen < args.max_gen:
                last_hidden, shared_kv, logits = target_step(ids)
                # bonus/verified token = target greedy at last position
                t0 = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (1,1)
                anchor_pos = ids.size(1) - 1
                hidden0 = last_hidden[:, -1:, :]                      # (1,1,H)
                drafts, dprobs = draft_propose(t0, hidden0, shared_kv, anchor_pos)
                draft_ids = torch.cat(drafts, dim=1)                 # (1,k)

                # verify: run target over [ids, t0, drafts]
                cand = torch.cat([ids, t0, draft_ids], dim=1)
                _, _, vlogits = target_step(cand)
                base = ids.size(1)   # position of t0 in cand
                # target dist at position base+j predicts token base+j+1 = draft j
                total_drafts += 1
                accepted = []
                for j in range(k):
                    total_draft_tokens += 1
                    tgt_logits_j = vlogits[0, base + j, :]           # (V,)
                    if args.mode == "greedy":
                        # accepted <=> draft == argmax(target)  (kernel:744-745)
                        ok = int(draft_ids[0, j].item()) == int(tgt_logits_j.argmax().item())
                    else:
                        # rejection sampling: p(x)/q(x) >= u  (kernel:810)
                        tp = torch.softmax(tgt_logits_j.float() / args.temperature, dim=-1)
                        x = int(draft_ids[0, j].item())
                        q = float(dprobs[j][0, x].item())
                        p = float(tp[x].item())
                        u = float(torch.rand(1, device=device).item())
                        ok = (q > 0) and (p / q >= u)
                    if ok:
                        total_accepted += 1
                        pos_accept[j] += 1
                        accepted.append(int(draft_ids[0, j].item()))
                    else:
                        break   # chained: first rejection stops

                # commit t0 + accepted + (first corrected token if any rejected)
                new = [int(t0.item())] + accepted
                if len(accepted) < k:
                    corrected = vlogits[0, base + len(accepted), :].argmax().item()
                    new.append(int(corrected))
                ids = torch.cat([ids, torch.tensor([new], device=device)], dim=1)
                gen += len(new)
                if tok.eos_token_id in new:
                    break

            if (si + 1) % 10 == 0:
                ar = total_accepted / max(total_draft_tokens, 1) * 100
                al = total_accepted / max(total_drafts, 1) + 1
                print(f"  [{si+1}/{args.num_samples}] accept_rate={ar:.2f}% "
                      f"accept_len={al:.2f}", flush=True)

    ar = total_accepted / max(total_draft_tokens, 1) * 100
    al = total_accepted / max(total_drafts, 1) + 1
    print(f"\n{'='*56}")
    print(f"mode: {args.mode}")
    print(f"Acceptance rate (%):   {ar:.2f}")
    print(f"Acceptance length:     {al:.2f}")
    print(f"Drafts:                {total_drafts}")
    print(f"Draft tokens:          {total_draft_tokens}")
    print(f"Accepted tokens:       {total_accepted}")
    print(f"Per-position acceptance (%):")
    for j in range(k):
        print(f"  Position {j}:           {pos_accept[j] / max(total_drafts,1) * 100:.2f}")
    print(f"{'='*56}")
    print("Compare to your vLLM bench. If they match, this repro is faithful and")
    print("can be used for fast iteration without a vLLM server.")


if __name__ == "__main__":
    main()

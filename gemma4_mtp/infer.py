#!/usr/bin/env python3
"""Minimal MTP speculative decoding with the Gemma 4 assistant (pure transformers).

The simplest possible end-to-end speculative decoder: no vLLM, no batching, no
KV-cache reuse tricks — just the honest verify loop so we can confirm the MTP
recipe (inputs_embeds = concat(target_embed(token)*sqrt(H), hidden),
shared_kv_states, autoregressive multi-step) actually accepts tokens against the
target on a real prompt.

Loop (greedy, batch size 1):
    1. Target forward on the current sequence -> next token t0 (the "bonus"/
       verified token) + last_hidden + shared_kv_states.
    2. Assistant proposes k tokens autoregressively from that hidden:
         step 0: token=t0,           hidden=target_last_hidden
         step j: token=draft_{j-1},  hidden=prev backbone_hidden
       (shared_kv_states constant; positions do not advance — matches vLLM.)
    3. Verify: append [t0, d0..d_{k-1}] and run ONE target forward over them.
       Accept draft d_j while it equals the target's greedy argmax at that
       position; stop at first mismatch (standard spec-decoding acceptance).
    4. Append accepted tokens (+ the first corrected token) and repeat.

This mirrors vLLM's acceptance semantics closely enough to sanity-check the
recipe. Numbers here are a smoke test, not the official benchmark (that's the
vllm-msn scaffold).

Run:
    python -m gemma4_mtp.infer \
        --target /tmp/models/gemma4/text_only \
        --assistant /tmp/models/gemma4/assistant \
        --prompt "Explain speculative decoding in one sentence." \
        --max-new-tokens 64 --spec-tokens 5 --bf16
"""

from __future__ import annotations

import argparse
import math


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--prompt", default="Explain speculative decoding in one sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--spec-tokens", type=int, default=5, help="k draft tokens/step")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-spec", action="store_true",
                    help="disable speculation (pure target greedy) for A/B check")
    ap.add_argument("--official", action="store_true",
                    help="use transformers' built-in Gemma4 MTP spec decoding "
                         "(ground-truth reference) instead of our manual loop")
    return ap.parse_args()


def _find_embed(mod):
    import torch.nn as nn
    try:
        emb = mod.get_input_embeddings()
        if emb is not None:
            return emb
    except Exception:
        pass
    for n, m in mod.named_modules():
        if isinstance(m, nn.Embedding) and m.embedding_dim >= 2048:
            return m
    raise RuntimeError("could not locate target input embedding")


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    target_embed = _find_embed(target)
    backbone = target_embed.embedding_dim
    normalizer = math.sqrt(backbone)
    print(f"  backbone_hidden={backbone}, normalizer={normalizer:.2f}", flush=True)

    # Build initial input.
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.size(1)

    def target_forward(input_ids):
        """Return (last_hidden, shared_kv_states, logits) over the full sequence."""
        out = target_base(input_ids=input_ids, return_shared_kv_states=True,
                          use_cache=False)
        lh = out.last_hidden_state
        skv = out.shared_kv_states
        logits = target_lm_head(lh) if target_lm_head is not None else \
            target(input_ids=input_ids, use_cache=False).logits
        return lh, skv, logits

    def draft_propose(t0_id, last_hidden_vec, shared_kv, draft_pos, attn_mask):
        """Propose k tokens autoregressively. Returns list of token ids.

        Matches the OFFICIAL transformers implementation
        (SinglePositionMultiTokenCandidateGenerator.get_candidates, v5.13.0):
          - token embedding is the target's RAW input embedding (NO normalizer)
          - inputs_embeds = concat(token_embed, last_hidden)
          - position_ids constant = [[seq_len - 1]] across all draft steps
          - attention_mask passed through from the target's model_kwargs
        """
        drafts = []
        hidden = last_hidden_vec  # (1, 1, H)
        tok_id = t0_id            # (1, 1)
        pos = torch.tensor([[draft_pos]], device=device)  # (1,1), constant
        for _ in range(k):
            tok_embed = target_embed(tok_id)                       # (1,1,H) NO normalizer
            combined = torch.cat([tok_embed, hidden], dim=-1)      # (1,1,2H)
            out = assistant(inputs_embeds=combined, shared_kv_states=shared_kv,
                            position_ids=pos, attention_mask=attn_mask)
            hidden = out.last_hidden_state                          # (1,1,H)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)
            drafts.append(int(nxt.item()))
            tok_id = nxt
        return drafts

    print("=== Generating ===", flush=True)

    # --- Official ground-truth mode: use transformers' built-in Gemma4 MTP
    # speculative decoding (SinglePositionMultiTokenCandidateGenerator). This is
    # the reference; compare its throughput/quality to our manual loop.
    if args.official:
        import time
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
        # First: assisted (spec decoding with the assistant).
        t0 = time.time()
        out_assisted = target.generate(ids, assistant_model=assistant, **gen_kwargs)
        dt_assisted = time.time() - t0
        n_assisted = out_assisted.shape[1] - prompt_len
        # Second: plain target greedy for A/B.
        t0 = time.time()
        out_plain = target.generate(ids, **gen_kwargs)
        dt_plain = time.time() - t0
        text_out = tok.decode(out_assisted[0, prompt_len:], skip_special_tokens=True)
        print(f"\n--- Official assisted output ---\n{text_out}\n", flush=True)
        print(f"{'='*50}")
        print(f"Assisted: {n_assisted} tok in {dt_assisted:.2f}s "
              f"({n_assisted/dt_assisted:.1f} tok/s)")
        print(f"Plain:    {out_plain.shape[1]-prompt_len} tok in {dt_plain:.2f}s "
              f"({(out_plain.shape[1]-prompt_len)/dt_plain:.1f} tok/s)")
        print(f"Speedup:  {dt_plain/dt_assisted:.2f}x")
        print(f"{'='*50}")
        return

    total_new = 0
    total_drafts = 0
    total_accepted = 0
    steps = 0

    with torch.no_grad():
        while total_new < args.max_new_tokens:
            steps += 1
            last_hidden, shared_kv, logits = target_forward(ids)
            # t0 = verified bonus token = target greedy at last position.
            t0 = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)

            if args.no_spec:
                ids = torch.cat([ids, t0], dim=1)
                total_new += 1
                if t0.item() == tok.eos_token_id:
                    break
                continue

            # Draft proposes k tokens from the last position's hidden state.
            # Official recipe: position_ids = [[seq_len - 1]] (the last SEEN
            # token's position), held constant across draft steps.
            last_hidden_vec = last_hidden[:, -1:, :]  # (1,1,H)
            draft_pos = ids.size(1) - 1
            drafts = draft_propose(t0, last_hidden_vec, shared_kv, draft_pos, None)
            total_drafts += k

            # Verify: run target over [ids + t0 + drafts], compare greedy.
            cand = torch.cat(
                [ids, t0, torch.tensor([drafts], device=device)], dim=1)
            _, _, vlogits = target_forward(cand)
            # Position of t0 in cand is prompt-tail index = ids.size(1).
            base = ids.size(1)
            # target greedy at position base+j predicts token at base+j+1.
            accepted = []
            for j in range(k):
                expected = vlogits[0, base + j, :].argmax().item()
                if drafts[j] == expected:
                    accepted.append(drafts[j])
                else:
                    break
            n_acc = len(accepted)
            total_accepted += n_acc

            # Commit: t0 (always) + accepted drafts + the first corrected token.
            new_tokens = [int(t0.item())] + accepted
            if n_acc < k:
                corrected = vlogits[0, base + n_acc, :].argmax().item()
                new_tokens.append(corrected)
            new_tensor = torch.tensor([new_tokens], device=device)
            ids = torch.cat([ids, new_tensor], dim=1)
            total_new += len(new_tokens)

            if tok.eos_token_id in new_tokens:
                break

    gen_ids = ids[0, prompt_len:]
    text_out = tok.decode(gen_ids, skip_special_tokens=True)
    print(f"\n--- Output ---\n{text_out}\n", flush=True)
    print(f"{'='*50}")
    print(f"New tokens:        {total_new}")
    print(f"Target forwards:   {steps * (1 if args.no_spec else 2)}")
    if not args.no_spec and total_drafts:
        print(f"Draft tokens:      {total_drafts}")
        print(f"Accepted:          {total_accepted}")
        print(f"Acceptance rate:   {total_accepted/total_drafts*100:.1f}%")
        print(f"Accept length:     {total_accepted/max(steps,1)+1:.2f}")
    print(f"{'='*50}")
    print("Note: greedy smoke test. Official numbers come from vllm-msn bench.")


if __name__ == "__main__":
    main()

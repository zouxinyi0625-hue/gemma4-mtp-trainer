#!/usr/bin/env python3
"""Evaluate an MTP assistant's acceptance rate against the target model.

Simulates vLLM's speculative decoding verification loop (greedy):
for each prompt, the draft proposes k tokens autoregressively, then the target
verifies them. We report acceptance rate, acceptance length, and per-position
acceptance — these should closely match vLLM's online benchmark numbers when
using the same models.

This script validates that our forward implementation (inputs_embeds recipe,
multi-step recursion, shared_kv_states) is correct: if the pretrained stock
assistant achieves ~80% acceptance / ~5.0 accept length here (matching the
vLLM online baseline), then the training loop built on the same forward is
also correct.

## Verified recursive recipe (from vllm/model_executor/models/gemma4_mtp.py):

    Step 0:
        inputs_embeds = embed(next_token) * sqrt(backbone_hidden_size)
        combined = concat(inputs_embeds, target_last_hidden)  # (B, 5632)
        pre_projection(combined) -> decoder -> post_projection
        -> (draft_hidden, backbone_hidden)

    Step k > 0:
        inputs_embeds = embed(draft_token_{k-1}) * sqrt(backbone_hidden_size)
        combined = concat(inputs_embeds, backbone_hidden_{k-1})
        pre_projection(combined) -> decoder -> post_projection
        -> (draft_hidden, backbone_hidden)

    Key: shared_kv_states is CONSTANT across steps (target's KV, not draft's).
         position_ids do NOT advance (constant_draft_positions in vLLM).

Run:
    python -m gemma4_mtp.eval \
        --target /tmp/models/gemma4/text_only \
        --assistant /tmp/models/gemma4/assistant \
        --data /path/to/maiprofile_regenerated.jsonl \
        --num-samples 100 --spec-tokens 5
"""

from __future__ import annotations

import argparse
import math
import sys


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--data", required=True, help="conversations JSONL (same as train)")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--spec-tokens", type=int, default=5, help="k: draft tokens per step")
    ap.add_argument("--max-gen-tokens", type=int, default=128,
                    help="max tokens to generate per sample for eval")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma4_mtp.data import DataConfig, build_dataset

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = args.device

    print("=== Loading models ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, device_map=device, trust_remote_code=True,
    ).eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=dtype, device_map=device, trust_remote_code=True,
    ).eval()

    # Locate sub-modules we need for the recursive recipe.
    target_base = getattr(target, "model", target)
    target_lm_head = getattr(target, "lm_head", None)

    # Assistant embed_tokens: search a couple of nesting levels robustly.
    def _find_embed(mod):
        for path in ("model.embed_tokens", "embed_tokens",
                     "model.model.embed_tokens"):
            obj = mod
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                return obj
        return None

    asst_embed = _find_embed(assistant)
    if asst_embed is None:
        raise RuntimeError(
            "could not locate assistant embed_tokens; inspect assistant module tree")
    asst_pre_proj = getattr(assistant, "pre_projection", None)
    # backbone_hidden_size for the normalizer. Gemma4Config is multimodal:
    # hidden_size lives in text_config, not at the top level. backbone_hidden_size
    # is a top-level field on the assistant config.
    def _hidden_size(cfg):
        if hasattr(cfg, "get_text_config"):
            tc = cfg.get_text_config()
            if getattr(tc, "hidden_size", None):
                return tc.hidden_size
        return getattr(cfg, "hidden_size", None)

    backbone_hidden_size = (
        getattr(assistant.config, "backbone_hidden_size", None)
        or _hidden_size(target.config)
        or 2816
    )
    normalizer = math.sqrt(backbone_hidden_size)
    print(f"  backbone_hidden_size={backbone_hidden_size}, normalizer={normalizer:.2f}",
          flush=True)

    print("=== Loading data ===", flush=True)
    data_cfg = DataConfig(max_length=2048)
    dataset = build_dataset(args.data, tokenizer, data_cfg)
    num_samples = min(args.num_samples, len(dataset))
    print(f"  evaluating {num_samples} samples, spec_tokens={args.spec_tokens}", flush=True)

    # --- Metrics accumulators ---
    total_drafts = 0
    total_draft_tokens = 0
    total_accepted = 0
    per_pos_accepted = [0] * args.spec_tokens
    per_pos_total = [0] * args.spec_tokens

    k = args.spec_tokens

    print("=== Running eval ===", flush=True)
    with torch.no_grad():
        for sample_idx in range(num_samples):
            sample = dataset[sample_idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)  # (1, T)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

            seq_len = input_ids.size(1)
            if seq_len < 10:
                continue

            # Use first half as "prompt", generate into second half for eval.
            # We compare draft predictions vs target's greedy at each position.
            prompt_len = seq_len // 2
            # Get target hidden states for the full sequence (teacher forcing).
            tgt_out = target_base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_shared_kv_states=True,
                output_hidden_states=False,
                use_cache=False,
            )
            target_last_hidden = tgt_out.last_hidden_state  # (1, T, 2816)
            shared_kv_states = tgt_out.shared_kv_states

            # Target logits for verification (greedy).
            if target_lm_head is not None:
                target_logits = target_lm_head(target_last_hidden)  # (1, T, V)
            else:
                target_logits = target(input_ids=input_ids, use_cache=False).logits
            target_greedy = target_logits.argmax(dim=-1)  # (1, T)
            # target_greedy[:, t] = argmax P_target(.|x_0..x_t) = token at t+1.

            # Simulate speculative decoding from each position in the eval range.
            eval_start = prompt_len
            eval_end = min(seq_len - k - 1, eval_start + args.max_gen_tokens)
            if eval_end <= eval_start:
                continue

            for t in range(eval_start, eval_end):
                # --- Draft proposes k tokens from position t ---
                draft_tokens = []
                backbone_h = target_last_hidden[:, t:t+1, :]  # (1, 1, 2816)

                for step in range(k):
                    # Token to embed: step 0 = ground truth token at t+1
                    # (teacher-forced context); step k>0 = draft's own prediction.
                    if step == 0:
                        tok = input_ids[:, t+1:t+2]  # (1, 1)
                    else:
                        tok = draft_tokens[-1].unsqueeze(0).unsqueeze(0)  # (1, 1)

                    # Recursive recipe (vLLM gemma4_mtp.py):
                    # inputs_embeds = embed(tok) * sqrt(backbone_dim)
                    # combined = cat(inputs_embeds, hidden_states)
                    tok_embed = asst_embed(tok) * normalizer  # (1, 1, 2816)
                    combined = torch.cat([tok_embed, backbone_h], dim=-1)  # (1, 1, 5632)

                    # Run the assistant forward
                    asst_out = assistant(
                        inputs_embeds=combined,
                        shared_kv_states=shared_kv_states,
                        position_ids=None,
                        attention_mask=None,
                    )
                    # assistant returns logits + last_hidden_state
                    draft_logits = asst_out.logits  # (1, 1, V)
                    # last_hidden_state = post_projection output = backbone_hidden
                    backbone_h = asst_out.last_hidden_state  # (1, 1, 2816)

                    draft_token = draft_logits.argmax(dim=-1).squeeze()  # scalar
                    draft_tokens.append(draft_token)

                # --- Target verifies the k draft tokens ---
                total_drafts += 1
                for pos in range(k):
                    total_draft_tokens += 1
                    per_pos_total[pos] += 1
                    # Draft predicted token at position t+1+pos
                    # Target's greedy at position t+pos is target_greedy[:, t+pos]
                    # (which predicts token at t+pos+1)
                    expected = target_greedy[0, t + pos].item()
                    predicted = draft_tokens[pos].item()
                    if predicted == expected:
                        total_accepted += 1
                        per_pos_accepted[pos] += 1
                    else:
                        break  # first rejection stops the chain

            if (sample_idx + 1) % 10 == 0:
                acc_rate = total_accepted / max(total_draft_tokens, 1) * 100
                acc_len = total_accepted / max(total_drafts, 1) + 1
                print(f"  [{sample_idx+1}/{num_samples}] accept_rate={acc_rate:.2f}% "
                      f"accept_len={acc_len:.2f}", flush=True)

    # --- Final report ---
    acc_rate = total_accepted / max(total_draft_tokens, 1) * 100
    acc_len = total_accepted / max(total_drafts, 1) + 1
    print(f"\n{'='*60}")
    print(f"Acceptance rate (%):    {acc_rate:.2f}")
    print(f"Acceptance length:      {acc_len:.2f}")
    print(f"Drafts:                 {total_drafts}")
    print(f"Draft tokens:           {total_draft_tokens}")
    print(f"Accepted tokens:        {total_accepted}")
    print(f"Per-position acceptance (%):")
    for pos in range(k):
        if per_pos_total[pos] > 0:
            print(f"  Position {pos}:            "
                  f"{per_pos_accepted[pos]/per_pos_total[pos]*100:.2f}")
    print(f"{'='*60}")
    print(f"\nExpected (vLLM online baseline): accept_rate=80.58%, accept_len=5.03")
    print(f"If close -> our forward implementation is correct for training.")
    print(f"If far off -> check inputs_embeds recipe / shared_kv_states / position handling.")


if __name__ == "__main__":
    main()

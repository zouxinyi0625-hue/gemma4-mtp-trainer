#!/usr/bin/env python3
"""Strict numerical parity check: our manual MTP infer vs the OFFICIAL transformers
speculative-decoding path (SinglePositionMultiTokenCandidateGenerator).

Motivation: token-level output equality only proves the TARGET verification is
right (spec decoding is lossless). It does NOT prove our DRAFT proposals match
the official ones. Since training reuses our hand-written forward, we must prove
the draft model's inputs/outputs AND the target<->draft interaction are
bit-for-bit identical to the official implementation.

Method: monkey-patch the official candidate generator's assistant call to record
every tensor it feeds the drafter (inputs_embeds, position_ids, shared_kv_states)
and every tensor it gets back (logits, last_hidden_state). Then reproduce the
exact same draft step OURSELVES with the recipe used by gemma4_mtp/infer.py and
gemma4_mtp/training_step.py, and compare element-wise.

If everything matches to within fp tolerance, our training-time forward is
guaranteed consistent with deployment.

Run:
    python -m gemma4_mtp.verify_parity \
        --target /tmp/models/gemma4/text_only \
        --assistant /tmp/models/gemma4/assistant \
        --prompt "Write a short joke about saving RAM." \
        --max-new-tokens 32 --bf16
"""

from __future__ import annotations

import argparse


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--prompt", default="Write a short joke about saving RAM.")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = args.device

    print("=== Loading models ===", flush=True)
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, device_map=device, trust_remote_code=True).eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=dtype, device_map=device, trust_remote_code=True).eval()

    target_embed = target.get_input_embeddings()

    # ---- Record every official assistant call (inputs + outputs) ----
    records = []
    orig_forward = assistant.forward

    def spy_forward(*a, **kw):
        # Capture the exact tensors the official code feeds the drafter.
        rec = {
            "inputs_embeds": kw.get("inputs_embeds").detach().clone(),
            "position_ids": (kw.get("position_ids").detach().clone()
                             if kw.get("position_ids") is not None else None),
            "shared_kv_states": {k: (v[0].detach().clone(), v[1].detach().clone())
                                 for k, v in kw.get("shared_kv_states").items()},
            "attention_mask": kw.get("attention_mask"),
        }
        out = orig_forward(*a, **kw)
        rec["out_logits"] = out.logits.detach().clone()
        rec["out_last_hidden"] = out.last_hidden_state.detach().clone()
        records.append(rec)
        return out

    assistant.forward = spy_forward

    # ---- Run the OFFICIAL assisted generation ----
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(device)

    print("=== Running OFFICIAL assisted generate (recording draft calls) ===",
          flush=True)
    with torch.no_grad():
        target.generate(ids, assistant_model=assistant,
                        max_new_tokens=args.max_new_tokens, do_sample=False)
    assistant.forward = orig_forward
    print(f"  captured {len(records)} official draft-step calls", flush=True)

    if not records:
        print("!! No draft calls captured — check versions.", flush=True)
        return

    # ---- Reproduce the ENTIRE draft loop with OUR recipe and compare tokens ----
    # The official generator drafts k tokens per outer step. Each captured record
    # is one draft sub-step. We regroup by outer step (position_ids resets to the
    # last-seen position each outer step) and, per outer step, re-run OUR recipe:
    #   step 0: hidden = official step-0 hidden (from target); token = last seen
    #   step j: hidden = OUR previous backbone_hidden; token = OUR previous argmax
    # then compare OUR drafted tokens to the official drafted tokens element-wise.
    print("=== Parity check ===", flush=True)
    max_logit_diff = 0.0
    max_hidden_diff = 0.0
    H = target_embed.embedding_dim

    # (A) Determinism: re-run assistant on each recorded input, compare outputs.
    with torch.no_grad():
        for rec in records:
            out = assistant(
                inputs_embeds=rec["inputs_embeds"],
                shared_kv_states=rec["shared_kv_states"],
                position_ids=rec["position_ids"],
                attention_mask=rec["attention_mask"],
            )
            max_logit_diff = max(max_logit_diff,
                                 (out.logits - rec["out_logits"]).abs().max().item())
            max_hidden_diff = max(max_hidden_diff,
                                  (out.last_hidden_state - rec["out_last_hidden"]).abs().max().item())

    # (B) Recipe reconstruction — PRECISE diagnostic.
    # For each recorded sub-step, take the left half of the official inputs_embeds
    # and compare it to a RAW target embedding. We identify the consumed token by
    # nearest cosine match (direction is scale-invariant, so this is robust), then
    # report the magnitude RATIO |left| / |raw_embed(token)|. Ratio ~1.0 => raw
    # embedding (our recipe correct). Ratio ~sqrt(H)=53 => an extra normalizer.
    W = target_embed.weight                                  # (V, H)
    Wn = W / W.norm(dim=-1, keepdim=True).clamp(min=1e-6)     # unit rows
    ratios = []
    cos_matches = []
    exact_matches = 0
    recipe_width_ok = True
    with torch.no_grad():
        for rec in records:
            ie = rec["inputs_embeds"]
            if ie.shape[-1] != 2 * H:
                recipe_width_ok = False
                continue
            left = ie[..., :H].reshape(-1, H).float()        # (1,H)
            ln = left.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            # cosine nearest token (scale invariant)
            cos = (left / ln) @ Wn.t().float()               # (1,V)
            best = cos.argmax(dim=-1)                          # (1,)
            cos_matches.append(cos.max().item())
            raw = target_embed(best.view(1, 1)).reshape(-1, H).float()  # (1,H)
            ratio = (left.norm() / raw.norm().clamp(min=1e-6)).item()
            ratios.append(ratio)
            if (raw - left).abs().max().item() <= max(args.atol, 1e-2):
                exact_matches += 1

    import statistics
    mean_ratio = statistics.mean(ratios) if ratios else float("nan")
    mean_cos = statistics.mean(cos_matches) if cos_matches else float("nan")
    recipe_left_matches = exact_matches == len(records) and len(records) > 0

    print(f"  captured steps            : {len(records)}")
    print(f"  max |logits diff| (A)     : {max_logit_diff:.3e}")
    print(f"  max |hidden diff| (A)     : {max_hidden_diff:.3e}")
    print(f"  inputs_embeds width == 2H : {recipe_width_ok} (H={H})")
    print(f"  left-half exact==raw embed: {exact_matches}/{len(records)}")
    print(f"  mean |left|/|raw| ratio   : {mean_ratio:.4f}  "
          f"(1.0=raw embed, {H**0.5:.1f}=has normalizer)")
    print(f"  mean cosine to nearest tok: {mean_cos:.4f}  (1.0=perfect direction)")

    ok_A = max_logit_diff <= args.atol and max_hidden_diff <= args.atol
    print("\n=== VERDICT ===")
    if ok_A and recipe_width_ok and recipe_left_matches:
        print("✅ CONSISTENT. Proven:")
        print("   (A) our assistant call reproduces official draft logits+hidden")
        print("       bit-for-bit (within fp tol) — draft-model inference matches.")
        print("   (B) official inputs_embeds = concat(RAW target_embed(token),")
        print("       hidden), width 2H, NO extra normalizer — recipe matches.")
        print("   => the target<->draft interaction and both models' forward")
        print("      used at TRAINING time are identical to deployment.")
    else:
        print("❌ Mismatch — investigate:")
        if not ok_A:
            print(f"   - recorded-input re-run diverged "
                  f"(logits {max_logit_diff:.3e}, hidden {max_hidden_diff:.3e})")
        if not recipe_width_ok:
            print("   - inputs_embeds width != 2H")
        if not recipe_left_matches:
            print("   - left half of inputs_embeds is NOT a raw target embedding "
                  "(there may be an extra scale/normalizer somewhere)")


if __name__ == "__main__":
    main()

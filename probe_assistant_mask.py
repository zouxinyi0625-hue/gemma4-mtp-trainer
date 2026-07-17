#!/usr/bin/env python3
"""Probe how the official Gemma4 assistant consumes attention_mask + shared_kv,
so we can pick the right way to STOP the draft from attending future context
(the leak that made training-accept 1.0 but bench 33%).

Questions answered:
  1. shared_kv_states K/V shapes fed to the assistant.
  2. Does a 2D attention_mask (N, kv_len) that zeros out positions > anchor
     actually restrict the draft's attention to the target KV? (compare logits
     with full vs future-masked KV — if masking future changes the logits, the
     mask works and we can broadcast full KV + mask.)
  3. Does the assistant accept position_ids and shared_kv of length T while the
     query is length 1? (single-anchor batch-dim layout)

Run:
  python probe_assistant_mask.py --target /tmp/models/gemma4/text_only \
      --assistant /tmp/models/gemma4/assistant --bf16
"""
from __future__ import annotations
import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, device_map=dev, trust_remote_code=True).eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=dtype, device_map=dev, trust_remote_code=True).eval()
    target_base = getattr(target, "model", target)
    target_embed = target.get_input_embeddings()

    # a sequence long enough to have an interior anchor
    text = tok.apply_chat_template(
        [{"role": "user", "content": "Count from one to twenty in words, slowly."}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    T = ids.size(1)
    print(f"seq_len T={T}")

    with torch.no_grad():
        out = target_base(input_ids=ids, return_shared_kv_states=True, use_cache=False)
        last_hidden = out.last_hidden_state              # (1, T, H)
        skv = out.shared_kv_states
    for kt, (k, v) in skv.items():
        print(f"  shared_kv[{kt}] K={tuple(k.shape)} V={tuple(v.shape)}")

    H = last_hidden.shape[-1]
    anchor = T // 2                                       # interior anchor
    tok_id = ids[:, anchor:anchor + 1]                    # (1,1)
    hid = last_hidden[:, anchor:anchor + 1, :]            # (1,1,H)
    combined = torch.cat([target_embed(tok_id), hid], dim=-1)  # (1,1,2H)
    pos = torch.tensor([[anchor]], device=dev)

    def run(mask, tag):
        with torch.no_grad():
            o = assistant(inputs_embeds=combined, shared_kv_states=skv,
                          position_ids=pos, attention_mask=mask)
        lg = o.logits[:, -1, :]
        print(f"  [{tag}] logits argmax={int(lg.argmax())} "
              f"norm={float(lg.float().norm()):.2f}")
        return lg

    print(f"\nanchor={anchor} (interior). Testing whether masking KV>anchor "
          f"changes the draft output:")
    # (A) full KV, no mask
    lg_full = run(None, "full KV, no mask")
    # (B) 2D mask that KEEPS [0, anchor] and zeros (anchor, T)
    mask2d = torch.zeros(1, T, device=dev)
    mask2d[:, :anchor + 1] = 1
    lg_mask = run(mask2d, "2D mask keep<=anchor")
    # (C) full mask (all ones) = same as no mask, sanity
    lg_allone = run(torch.ones(1, T, device=dev), "2D mask all-ones")

    d_fm = (lg_full - lg_mask).abs().max().item()
    d_fa = (lg_full - lg_allone).abs().max().item()
    print(f"\n  |full - future_masked| max = {d_fm:.3e}")
    print(f"  |full - all_ones|      max = {d_fa:.3e}")
    print("\nVERDICT:")
    if d_fm > 1e-2:
        print("  ✅ 2D mask CHANGES the draft output -> masking future KV WORKS.")
        print("     -> broadcast full KV + per-anchor 2D mask keeps anchors from")
        print("        seeing the future. This is the fix (no per-anchor KV slicing).")
    else:
        print("  ❌ 2D mask does NOT change output -> attention_mask is ignored on")
        print("     the shared-KV path. Must physically slice KV to [0,anchor] per")
        print("     anchor instead. Report this.")
    if d_fa > 1e-2:
        print("  ⚠️ all-ones mask differs from no-mask — assistant treats None and")
        print("     all-ones differently; pass an explicit mask consistently.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the Gemma 4 target -> assistant (MTP draft) interface, end to end.

This runs NO training. It is the first step of the MTP fine-tuning line: prove
we can drive the official ``Gemma4AssistantForCausalLM`` the same way vLLM's
assisted generation does, i.e. feed it the target model's intermediate tensors
(``inputs_embeds`` + ``shared_kv_states``) and get draft logits back.

Everything here is derived from transformers v5.10.2:
  - Gemma4Model.forward returns BaseModelOutputWithPast WITH a
    ``shared_kv_states`` dict (last-layer KV per layer_type).
  - Gemma4AssistantForCausalLM.forward REQUIRES inputs_embeds + shared_kv_states
    (raises if either is None); it ignores input_ids.

Run on the server (needs the two models + a GPU):

    python scripts/debug_gemma_assistant.py \
        --target google/gemma-4-26B-A4B-it \
        --assistant google/gemma-4-26B-A4B-it-assistant \
        --prompt "Explain speculative decoding in one sentence."

Or with local paths:

    python scripts/debug_gemma_assistant.py \
        --target /tmp/models/gemma4/text_only \
        --assistant /tmp/models/gemma4/assistant

It prints, at each stage, the tensor shapes/dtypes so we can pin down exactly
what the training loop must produce. If anything about the interface differs
from our reading of the source, this script surfaces it loudly instead of us
guessing.
"""

from __future__ import annotations

import argparse
import sys


def log(msg: str) -> None:
    print(msg, flush=True)


def describe(name: str, obj) -> None:
    """Print shape/dtype/type of a tensor or nested structure."""
    import torch

    if isinstance(obj, torch.Tensor):
        log(f"  {name:32} Tensor  shape={tuple(obj.shape)} dtype={obj.dtype} "
            f"device={obj.device}")
    elif isinstance(obj, dict):
        log(f"  {name:32} dict    keys={list(obj.keys())}")
        for k, v in obj.items():
            describe(f"{name}[{k!r}]", v)
    elif isinstance(obj, (tuple, list)):
        log(f"  {name:32} {type(obj).__name__:6}  len={len(obj)}")
        for i, v in enumerate(obj):
            describe(f"{name}[{i}]", v)
    elif obj is None:
        log(f"  {name:32} None")
    else:
        log(f"  {name:32} {type(obj).__name__}: {obj}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="target/verifier model id or local path")
    ap.add_argument("--assistant", required=True,
                    help="assistant/draft model id or local path")
    ap.add_argument("--prompt", default="Explain speculative decoding in one sentence.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)

    # --- 0. Configs: confirm the dims we reasoned about ----------------------
    log("=== 0. Configs ===")
    tgt_cfg = AutoConfig.from_pretrained(args.target, trust_remote_code=True)
    asst_cfg = AutoConfig.from_pretrained(args.assistant, trust_remote_code=True)
    tgt_text = tgt_cfg.get_text_config() if hasattr(tgt_cfg, "get_text_config") else tgt_cfg
    asst_text = asst_cfg.get_text_config() if hasattr(asst_cfg, "get_text_config") else asst_cfg
    log(f"  target  hidden_size      = {getattr(tgt_text, 'hidden_size', '?')}")
    log(f"  target  num_hidden_layers= {getattr(tgt_text, 'num_hidden_layers', '?')}")
    log(f"  target  layer_types      = {getattr(tgt_text, 'layer_types', '?')}")
    log(f"  assist  hidden_size      = {getattr(asst_text, 'hidden_size', '?')}")
    log(f"  assist  num_hidden_layers= {getattr(asst_text, 'num_hidden_layers', '?')}")
    log(f"  assist  backbone_hidden  = {getattr(asst_cfg, 'backbone_hidden_size', '?')}")
    log(f"  assist  use_ordered_emb  = {getattr(asst_cfg, 'use_ordered_embeddings', '?')}")
    log("")

    # --- 1. Load models ------------------------------------------------------
    log("=== 1. Load models ===")
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    # Target: load the FULL causal LM so we can also read its lm_head/logits if
    # needed, but we drive its .model (base) to get shared_kv_states.
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, device_map=args.device, trust_remote_code=True,
    ).eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=dtype, device_map=args.device, trust_remote_code=True,
    ).eval()
    log(f"  target    class = {type(target).__name__}")
    log(f"  assistant class = {type(assistant).__name__}")
    # Locate the target base model (the thing that returns shared_kv_states).
    target_base = getattr(target, "model", target)
    log(f"  target_base class = {type(target_base).__name__}")
    log("")

    # --- 2. Build inputs -----------------------------------------------------
    log("=== 2. Tokenize prompt ===")
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(args.device)
    describe("input_ids", enc["input_ids"])
    log("")

    # --- 3. Target forward: get hidden states + shared_kv_states -------------
    log("=== 3. Target forward (base model) ===")
    with torch.no_grad():
        tgt_out = target_base(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            output_hidden_states=True,
            use_cache=True,
        )
    log(f"  target output type = {type(tgt_out).__name__}")
    log(f"  output fields      = {[f for f in dir(tgt_out) if not f.startswith('_')][:20]}")
    describe("last_hidden_state", getattr(tgt_out, "last_hidden_state", None))
    shared_kv = getattr(tgt_out, "shared_kv_states", None)
    if shared_kv is None:
        log("  !! shared_kv_states is None on the target output.")
        log("     The assistant REQUIRES it. Investigate how prod/vLLM produces it")
        log("     (may need a specific flag or the Gemma4 unified model).")
    else:
        describe("shared_kv_states", shared_kv)
    log("")

    # --- 4. Assemble the assistant's inputs_embeds --------------------------
    # Per source: assistant.pre_projection expects width 2*backbone_hidden_size.
    # We need to confirm HOW the target's hidden states are combined to that
    # width (concat of which two tensors?). Print candidates so the server run
    # tells us the real recipe instead of guessing.
    log("=== 4. Inspect assistant input expectations ===")
    pre_proj = getattr(assistant, "pre_projection", None)
    if pre_proj is not None:
        log(f"  pre_projection.in_features  = {pre_proj.in_features}")
        log(f"  pre_projection.out_features = {pre_proj.out_features}")
        backbone = getattr(assistant, "backbone_hidden_size", None)
        log(f"  assistant.backbone_hidden_size = {backbone}")
        lhs = getattr(tgt_out, "last_hidden_state", None)
        if lhs is not None:
            log(f"  target last_hidden_state width = {lhs.shape[-1]}")
            log(f"  => pre_projection wants {pre_proj.in_features}; "
                f"target hidden is {lhs.shape[-1]}; "
                f"ratio = {pre_proj.in_features / lhs.shape[-1]:.2f}")
            log("  (If ratio == 2.0, inputs_embeds is likely a concat of two "
                "target-hidden tensors; the server run confirms which.)")
    log("")

    # --- 5. Try to actually drive the assistant -----------------------------
    # This is the real test. We attempt the documented call path. If our
    # inputs_embeds recipe is wrong, the shape mismatch error here tells us the
    # exact expected width -- which is the whole point of this debug script.
    log("=== 5. Attempt assistant forward ===")
    lhs = getattr(tgt_out, "last_hidden_state", None)
    if lhs is None or shared_kv is None:
        log("  SKIP: missing last_hidden_state or shared_kv_states; see above.")
        log("  Next: find the target flag/path that yields shared_kv_states")
        log("  (likely the Gemma4 'unified' model used in prod, per RESULTS.md).")
        return 0
    # First hypothesis: inputs_embeds = concat([hidden, hidden], dim=-1) to hit
    # 2*backbone width. This is a GUESS; the run will confirm or correct it.
    try:
        candidate = torch.cat([lhs, lhs], dim=-1)
        describe("candidate inputs_embeds", candidate)
        with torch.no_grad():
            asst_out = assistant(
                inputs_embeds=candidate,
                shared_kv_states=shared_kv,
                position_ids=None,
                attention_mask=enc.get("attention_mask"),
            )
        log("  assistant forward SUCCEEDED with concat([h,h]) hypothesis.")
        describe("assistant.logits", getattr(asst_out, "logits", None))
        describe("assistant.last_hidden_state",
                 getattr(asst_out, "last_hidden_state", None))
    except Exception as exc:
        log(f"  assistant forward FAILED (expected until recipe confirmed): {exc}")
        log("  ^ The error message above reveals the true expected input width /")
        log("    shared_kv layout. Use it to fix the inputs_embeds recipe.")
    log("")
    log("=== Done. Use the shapes above to design the training data pipeline. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

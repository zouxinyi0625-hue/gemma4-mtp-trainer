#!/usr/bin/env python3
"""Diagnose why the fine-tuned MTP draft degrades in vLLM (76% -> 3%).

Checks two prime suspects (no training, read-only, a few seconds):
  #4 checkpoint weight-key prefix vs vLLM's hf_to_vllm_mapper expectation
  #1 token-embedding scale: sqrt(embed_dim) vs sqrt(backbone_hidden_size)

Usage:
  python diag_ckpt.py <CKPT_DIR> [TARGET_DIR]
    CKPT_DIR   = your exported checkpoint (e.g. .../mtp_maiprofile/<tag>/step600)
    TARGET_DIR = target model (default /tmp/models/gemma4/text_only)
"""
import glob
import math
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

CKPT = sys.argv[1]
TARGET = sys.argv[2] if len(sys.argv) > 2 else "/tmp/models/gemma4/text_only"

print("=" * 60)
print("#4  checkpoint weight-key structure")
print("=" * 60)
files = sorted(glob.glob(os.path.join(CKPT, "*.safetensors")))
if not files:
    # fall back to pytorch_model.bin
    bins = sorted(glob.glob(os.path.join(CKPT, "*.bin")))
    print(f"no safetensors; found {bins}")
    keys = []
    for b in bins:
        keys += list(torch.load(b, map_location="cpu", weights_only=True).keys())
else:
    keys = []
    for f in files:
        with safe_open(f, framework="pt") as sf:
            keys += list(sf.keys())

proj = [k for k in keys if "projection" in k]
head = [k for k in keys if "lm_head" in k]
emb = [k for k in keys if "embed_tokens" in k]
print(f"total keys: {len(keys)}")
print("projection keys:")
for k in proj:
    print("   ", k)
print("lm_head keys:", head)
print("embed_tokens keys:", emb)
print()
print("vLLM hf_to_vllm_mapper prepends 'model.' to pre_projection./post_projection.")
print("  -> vLLM EXPECTS top-level  pre_projection.weight  (becomes model.pre_projection.weight)")
print("  -> if your keys are  model.pre_projection.weight  they become model.model.* and FAIL to load")
bad = [k for k in proj if k.startswith("model.pre_projection") or k.startswith("model.post_projection")]
top = [k for k in proj if k.startswith("pre_projection") or k.startswith("post_projection")]
if bad:
    print(f"  VERDICT #4: LIKELY BUG — projections nested under model.* : {bad}")
elif top:
    print(f"  VERDICT #4: keys are top-level {top} — vLLM mapper should match. OK.")
else:
    print(f"  VERDICT #4: projection keys have an unexpected prefix — inspect: {proj}")

print()
print("=" * 60)
print("#1  token-embedding scale")
print("=" * 60)
tgt = AutoModelForCausalLM.from_pretrained(
    TARGET, dtype=torch.bfloat16, trust_remote_code=True)
te = tgt.get_input_embeddings()
print("target embed type      :", type(te).__name__)
print("target embed.embedding_dim:", te.embedding_dim)

acfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
backbone = getattr(acfg, "backbone_hidden_size", None)
tc = acfg.get_text_config() if hasattr(acfg, "get_text_config") else acfg
draft_hidden = getattr(tc, "hidden_size", None)
print("assistant backbone_hidden_size:", backbone)
print("assistant hidden_size (draft) :", draft_hidden)

hf_scale = math.sqrt(te.embedding_dim)
vllm_scale = math.sqrt(backbone) if backbone else float("nan")
print(f"HF   scale = sqrt(embed_dim)  = {hf_scale:.3f}")
print(f"vLLM scale = sqrt(backbone)   = {vllm_scale:.3f}")
if backbone and abs(hf_scale - vllm_scale) < 1e-3:
    print("  VERDICT #1: scales MATCH. OK.")
else:
    print("  VERDICT #1: scales DIFFER -> token half off by this ratio -> BUG.")

print()
print("=" * 60)
print("#6  lm_head / embed_tokens TIE and dims (prime new suspect)")
print("=" * 60)
print("tie_word_embeddings:", getattr(acfg, "tie_word_embeddings",
                                      getattr(tc, "tie_word_embeddings", "?")))
# actual shape of the checkpoint's embed_tokens (which lm_head is tied to)
emb_shape = None
if files:
    for f in files:
        with safe_open(f, framework="pt") as sf:
            if "model.embed_tokens.weight" in sf.keys():
                emb_shape = sf.get_slice("model.embed_tokens.weight").get_shape()
                break
print("checkpoint model.embed_tokens.weight shape:", emb_shape)
print("draft hidden_size (vLLM lm_head expects vocab x this):", draft_hidden)
print()
print("vLLM: draft lm_head is (vocab, draft_hidden). When tie_word_embeddings=True,")
print("      lm_head.weight = draft embed_tokens.weight, which in vLLM is (vocab, draft_hidden).")
print("      vLLM computes draft_logits = lm_head(norm_hidden) where norm_hidden is DRAFT-dim.")
if emb_shape is not None and draft_hidden is not None:
    if emb_shape[1] == draft_hidden:
        print(f"  VERDICT #6: embed_tokens is (vocab, {emb_shape[1]}) == draft_hidden. OK — "
              "tied lm_head is draft-dim as vLLM expects.")
    elif emb_shape[1] == te.embedding_dim:
        print(f"  VERDICT #6: BUG — checkpoint embed_tokens is (vocab, {emb_shape[1]}) = "
              f"BACKBONE dim ({te.embedding_dim}), NOT draft_hidden ({draft_hidden}). "
              "The training OVERWROTE the draft embed with the target backbone embed and saved it. "
              "vLLM ties its draft-dim lm_head expecting draft_hidden -> dim conflict / garbage logits -> 3%.")
    else:
        print(f"  VERDICT #6: embed_tokens dim {emb_shape[1]} matches neither "
              f"draft {draft_hidden} nor backbone {te.embedding_dim} — inspect.")

print()
print("=" * 60)
print("#7  DIFF your checkpoint keys vs the STOCK assistant (the 76% model)")
print("=" * 60)
STOCK = os.environ.get("STOCK_ASSISTANT", "/tmp/models/gemma4/assistant")
print(f"stock assistant dir: {STOCK}")
stock_files = sorted(glob.glob(os.path.join(STOCK, "*.safetensors")))
stock_keys = set()
stock_shapes = {}
if stock_files:
    for f in stock_files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                stock_keys.add(k)
                stock_shapes[k] = tuple(sf.get_slice(k).get_shape())
else:
    print("  (no safetensors in stock dir; skipping diff)")

your_keys = set(keys)
if stock_keys:
    only_stock = sorted(stock_keys - your_keys)
    only_yours = sorted(your_keys - stock_keys)
    print(f"stock keys: {len(stock_keys)}   your keys: {len(your_keys)}")
    print("keys in STOCK but MISSING from yours (vLLM may expect these!):")
    for k in only_stock:
        print("   -", k, stock_shapes.get(k))
    print("keys in YOURS but not in stock:")
    for k in only_yours:
        print("   +", k)
    # shape mismatches on shared keys
    shared = your_keys & stock_keys
    your_shapes = {}
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                your_shapes[k] = tuple(sf.get_slice(k).get_shape())
    mism = [(k, stock_shapes[k], your_shapes.get(k))
            for k in sorted(shared) if stock_shapes[k] != your_shapes.get(k)]
    print("shape mismatches on shared keys:")
    for k, s, y in mism:
        print(f"   ! {k}: stock={s} yours={y}")
    if not only_stock and not only_yours and not mism:
        print("  VERDICT #7: checkpoint structure IDENTICAL to stock. Export is fine;")
        print("             the degradation is from the TRAINED VALUES, not the format.")
    else:
        print("  VERDICT #7: STRUCTURE DIFFERS from the 76% stock model -> likely the bug.")



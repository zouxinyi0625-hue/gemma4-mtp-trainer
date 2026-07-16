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

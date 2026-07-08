# MTP Training Design (Gemma 4 assistant)

Status: design confirmed against verified interface (debug run 2026-07-08).
This documents HOW we train, before writing the loop, so the approach is
reviewable.

## What we're training

Fine-tune `Gemma4AssistantForCausalLM` (the 4-layer draft) so its next-token
distribution better matches the target (`gemma-4-26B-A4B-it`) on MAI Profile
data, raising acceptance rate. We do **not** change its architecture.

## Verified interface (from debug_gemma_assistant.py)

Per training sample (a token sequence of length T):

```
# 1. Target base forward (frozen, no grad)
tgt = target_base(input_ids, attention_mask,
                  return_shared_kv_states=True, use_cache=False)
last_hidden      = tgt.last_hidden_state          # (B, T, 2816)
shared_kv_states = tgt.shared_kv_states            # {sliding:(K,V), full:(K,V)}

# 2. Build assistant input and run the draft (trainable)
inputs_embeds = torch.cat([last_hidden, last_hidden], dim=-1)   # (B, T, 5632)
out = assistant(inputs_embeds=inputs_embeds,
                shared_kv_states=shared_kv_states,
                position_ids=..., attention_mask=...)
draft_logits = out.logits                          # (B, T, 262144)
```

## The key architectural difference vs speculators MTP

speculators' MTP head (Qwen) is a **single layer** that takes
`(hidden, token_embedding)` and is unrolled K steps, each step feeding its own
output back in, with per-step weighted CE (`beta^k`). See
`speculators/src/speculators/models/mtp/core.py` forward.

The Gemma 4 assistant is **different**: a full 4-layer decoder whose forward
signature has no `token_embeddings` arg and is not designed to be fed its own
output recursively. At inference vLLM/transformers calls it autoregressively
(generation_config `num_assistant_tokens=6`) — the multi-token behaviour comes
from repeated forward calls, not from an internal K-step unroll.

**Therefore the training objective is standard next-token prediction on the
draft's single-step output**, distilled against the target — not the speculators
K-step recurrence. This matches how the assistant is actually used.

## Loss: distillation (soft CE) — the acceptance-rate lever

We want the draft to match the *target's* distribution (that's exactly what
speculative decoding accepts on), so:

```
# target next-token distribution (frozen) — reuse the SAME target forward,
# take its lm_head logits (or the model's logits output) as soft labels
with no_grad:
    target_logits = target_lm_head(last_hidden)        # (B, T, 262144)
    target_probs  = softmax(target_logits / temperature, dim=-1)

# draft predicts the next token; align positions (predict t+1 from pos t)
loss = - sum_v target_probs[:, :-1] * log_softmax(draft_logits[:, :-1])
loss = (loss * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum()
```

- **soft CE (KL to target)** is the primary loss — proven to lift acceptance
  more than hard CE (DeepSpec eagle3/loss.py uses the same idea).
- Optionally add a small hard-CE term against the ground-truth next token
  (the target-regenerated answer) for stability.
- `loss_mask` excludes prompt tokens; train only on assistant-response tokens.

## What is frozen / trained

- **Frozen:** the entire target model; the assistant's `lm_head` and
  `model.embed_tokens` (they are tied and shared with vocab — keep them stock so
  the checkpoint stays a drop-in replacement).
- **Trained:** the assistant's 4 decoder layers + `pre_projection` +
  `post_projection`.

> Open question to validate on server: whether unfreezing `lm_head` helps or
> breaks vLLM compatibility. Default = freeze.

## Multi-token (optional, later)

If single-step distillation under-delivers, we can emulate inference-time
multi-step by teacher-forcing: feed the draft its own predicted token embedding
for step 2..k. But start simple (single-step) — it's the honest match to the
architecture and the fastest path to a first acceptance-rate number.

## Data

- MAI Profile prompts, **target-regenerated** answers (same as DSpark pipeline,
  `DeepSpec@dev/maiprofile`). MTP learns the target's own outputs.
- Offline cache of `(input_ids, loss_mask)` per sample; hidden states + KV are
  produced on-the-fly from the frozen target during training (they're large;
  caching them for 26B is expensive — start online).

## Export

Save the fine-tuned assistant with the **same config/architecture** as the stock
`google/gemma-4-26B-A4B-it-assistant` so vLLM loads it identically. Then bench on
the vllm-msn scaffold (MTP config, swap assistant path) vs the stock assistant.

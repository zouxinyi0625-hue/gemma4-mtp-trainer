# MTP Training Design (Gemma 4 assistant)

Status: design confirmed against verified interface (debug run 2026-07-08) and
against the official transformers speculative-decoding path (see
`gemma4_mtp/verify_parity.py`, which proves our training-time forward is
bit-for-bit identical to deployment).

## What we're training

Fine-tune `Gemma4AssistantForCausalLM` (the 4-layer draft) so its next-token
distribution better matches the target (`gemma-4-26B-A4B-it`) on MAI Profile
data, raising acceptance rate. We do **not** change its architecture.

## Verified interface (from debug_gemma_assistant.py + verify_parity.py)

Per training sample (a token sequence of length T):

```
# 1. Target base forward (frozen, no grad)
tgt = target_base(input_ids, attention_mask,
                  return_shared_kv_states=True, use_cache=False)
last_hidden      = tgt.last_hidden_state          # (B, T, 2816)
shared_kv_states = tgt.shared_kv_states            # {sliding:(K,V), full:(K,V)}

# 2. Build the assistant input and run the draft (trainable)
#    inputs_embeds = concat(target_embed(token), hidden), width 2*2816=5632.
#    target_embed is the TARGET's backbone-dim embedding; it is a
#    Gemma4TextScaledWordEmbedding that ALREADY multiplies by sqrt(hidden_size)
#    internally, so we do NOT apply an extra normalizer (verified in
#    verify_parity.py: |left|/|raw_embed| ratio ~= 1.0, not sqrt(H)).
tok_embed     = target_embed(token_ids)                        # (B, T, 2816)
inputs_embeds = torch.cat([tok_embed, hidden], dim=-1)          # (B, T, 5632)
out = assistant(inputs_embeds=inputs_embeds,
                shared_kv_states=shared_kv_states,
                position_ids=..., attention_mask=...)
draft_logits    = out.logits                       # (B, T, 262144)
backbone_hidden = out.last_hidden_state             # (B, T, 2816), fed back in TTT
```

## The objective: multi-step TTT distillation (matches deployment)

At inference, vLLM/transformers drives the assistant **autoregressively** for
`k` draft tokens per outer step (`SinglePositionMultiTokenCandidateGenerator`):

```
step 0: token = input_ids[t],   hidden = target_last_hidden[t]
step j: token = draft_{j-1},    hidden = backbone_hidden from step j-1
```

with `shared_kv_states` held constant (the target's KV, not the draft's) and
`position_ids` NOT advancing across draft steps. Training must match this, so
the objective is the **Training-Time Test (TTT)** unroll described in the
Gemma 4 Technical Report (§2.6): unroll K draft steps, feed each step its own
`backbone_hidden` back in, and supervise each step with per-step weights
(`beta^k`, normalized). Single-step training makes "pos0 good, tail collapses";
TTT trains the whole accept-length curve. See `gemma4_mtp/training_step.py`.

> Note: this supersedes the earlier single-step
> `concat(last_hidden, last_hidden)` sketch. `verify_parity.py` proved the real
> recipe is `concat(target_embed(token), hidden)` unrolled K steps.

## Loss: distillation (soft CE) — the acceptance-rate lever

We want the draft to match the *target's* distribution (that's exactly what
speculative decoding accepts on), so at each aligned step:

```
# target next-token distribution (frozen), recomputed from the target's OWN
# cached/last hidden via the frozen tied lm_head — the FULL distribution, not a
# top-K approximation. Only supervised (loss_mask==1) positions are gathered
# first, so the 262144-wide softmax is computed only where it matters (avoids
# the OOM of a full-sequence vocab softmax).
with no_grad:
    target_logits = target_lm_head(target_hidden_at_step)   # (n_sup, 262144)
    target_probs  = softmax(target_logits / temperature, dim=-1)
soft_ce = -(target_probs * log_softmax(draft_logits / temperature)).sum(-1).mean()
```

- **soft CE (KL to target)** is the primary loss — proven to lift acceptance
  more than hard CE (DeepSpec eagle3/loss.py uses the same idea).
- Optionally add a small hard-CE term against the ground-truth next token
  (the target-regenerated answer) for stability (`--hard-ce-weight`).
- `loss_mask` excludes prompt tokens; train only on assistant-response tokens.

## What is frozen / trained

- **Frozen:** the entire target model; the assistant's `lm_head` and
  `model.embed_tokens` (they are tied and shared with vocab — keep them stock so
  the checkpoint stays a drop-in replacement).
- **Trained:** the assistant's 4 decoder layers + `pre_projection` +
  `post_projection`.

> Open question to validate on server: whether unfreezing `lm_head` helps or
> breaks vLLM compatibility. Default = freeze.

## Multi-token TTT (the default, not optional)

TTT (feed the draft its own `backbone_hidden` for steps 1..K-1) is the default
objective, controlled by `--ttt-steps` (match deployment `spec_tokens`, ~5) and
`--step-weight-beta`. Set `--ttt-steps 1` to fall back to single-step training
for an A/B. One known simplification to validate on the server: training runs
the K steps as full-sequence teacher forcing with `position_ids=None` /
`attention_mask=None`, whereas inference holds `position_ids` constant at
`seq_len-1`; confirm this does not hurt convergence vs the accept-length curve.

## Data

- MAI Profile prompts, **target-regenerated** answers (same as DSpark pipeline,
  `DeepSpec@dev/maiprofile`). MTP learns the target's own outputs.
- Two paths:
  - **Online** (`train.py --data ...`): the frozen target forward runs every
    step to produce `last_hidden` + `shared_kv_states`; target soft labels come
    from `lm_head(last_hidden)`. Simple, but re-runs the 26B target every epoch.
  - **Offline cache** (`prepare_cache.py` -> `train.py --cache-dir ...`): the
    target is FROZEN, so its per-sample signals are identical every epoch. We
    precompute them ONCE and stream to a sharded on-mount cache
    (`gemma4_mtp/target_cache.py`: async writer + large binary shards +
    fixed-width index + mmap reader). Training then runs only the small
    assistant — no 26B target, no OOM. Per sample we cache `input_ids`,
    `loss_mask`, `last_hidden`, and the `{full,sliding}` shared KV; we do NOT
    cache logits (~1GB/sample) — soft labels are recomputed from `last_hidden`
    via the frozen tied `lm_head` at train time.

## Export

Save the fine-tuned assistant with the **same config/architecture** as the stock
`google/gemma-4-26B-A4B-it-assistant` so vLLM loads it identically. Then bench on
the vllm-msn scaffold (MTP config, swap assistant path) vs the stock assistant.

## Cross-check against speculators PR #768 (official Gemma4 MTP support)

speculators is adding official Gemma4 MTP *training* support via a 3-PR stack
(Issue #586): #758 (extract verifier KV), #767 (multi-level LM head), #768
(`QueryOnlyGemma2Attention`). As of research all three are OPEN / unmerged.

That work **rebuilds** the drafter from `Gemma2DecoderLayer` inside speculators,
so it must manually handle three things. We reuse Google's native
`Gemma4AssistantForCausalLM`, so the official transformers forward handles all
three for us — verified against `modeling_gemma4.py` v5.10.2:

1. **KV is pre-rotated.** #768 notes "verifier's KV cache already has RoPE
   applied". Confirmed in official Gemma4: a non-shared layer stores KV into
   `shared_kv_states` *after* `apply_rotary_pos_emb`
   (`store_full_length_kv` path), so the shared KV we pass to the assistant is
   already rotated.
2. **RoPE on query only.** #768 applies RoPE to queries only. Confirmed: on a
   kv-shared layer the official code rotates `query_states` but reuses
   `shared_kv_states[layer_type]` directly (no re-rotation of KV).
3. **sliding vs full routing.** #768 picks local/global KV by `sliding_window`.
   Confirmed: official code indexes `shared_kv_states[self.layer_type]`
   ({sliding_attention, full_attention}) automatically.

**Conclusion:** our approach needs no low-level attention/RoPE/KV handling — the
native assistant forward is correct by construction. This validates the decision
to use the official assistant instead of porting into speculators, and avoids
depending on three unmerged PRs. #768 remains a useful reference for the
centroid-masked (multi-level) LM head if we later unfreeze/adopt it.

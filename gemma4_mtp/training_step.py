"""Core MTP fine-tuning step for the Gemma 4 assistant (TTT multi-step).

Implements the autoregressive Training-Time Test (TTT) objective described in
the Gemma 4 Technical Report (arXiv 2607.02770 §2.6) and matched to vLLM's
inference recipe (vllm/model_executor/models/gemma4_mtp.py):

    Per draft step, the MTP head consumes:
      - the PREVIOUS step's last-layer activations (hidden), and
      - a token embedding (from the target's backbone-dim embedder),
    concatenated and projected, cross-attending the target's KV cache.

Recipe (verified):
    combined = concat(target_embed(token) * sqrt(backbone), hidden)   # (B,T,5632)
    draft_logits, backbone_hidden = assistant(inputs_embeds=combined,
                                              shared_kv_states=<target KV, const>)

    step 0: token = input_ids[t],   hidden = target_last_hidden[t]
    step k: token = input_ids[t+k], hidden = backbone_hidden[t] from step k-1

At step k, position t predicts token t+k+1. We supervise each step against the
target's own next-token distribution (soft CE / KL) and/or the ground-truth
token (hard CE), with per-step weights. Training the head across K steps (TTT)
prevents the "pos0 good, tail collapses" failure of single-step training.

torch is imported lazily so the module is import-safe without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MTPLossConfig:
    """Hyperparameters for the TTT distillation loss."""

    # Number of TTT draft steps to unroll (match deployment spec_tokens).
    ttt_steps: int = 5
    # Per-step loss weights; if None, use decaying beta**k (normalized).
    step_weights: list[float] | None = None
    step_weight_beta: float = 0.8
    temperature: float = 1.0
    # Distillation (soft-CE / KL to target) weight.
    soft_ce_weight: float = 1.0
    # Hard-CE against the ground-truth next token weight.
    hard_ce_weight: float = 0.0
    ignore_index: int = -100


def compute_step_weights(cfg: MTPLossConfig) -> list[float]:
    if cfg.step_weights is not None:
        assert len(cfg.step_weights) == cfg.ttt_steps
        return list(cfg.step_weights)
    w = [cfg.step_weight_beta ** k for k in range(cfg.ttt_steps)]
    s = sum(w)
    return [x / s for x in w]


def locate_target_parts(target):
    """Return (target_base, lm_head, embed, normalizer) for the recipe.

    embed is the TARGET's backbone-dim embedding (not the assistant's draft-dim
    embedding). normalizer = sqrt(backbone_hidden_size) (vLLM applies this).
    """
    import torch.nn as nn

    target_base = getattr(target, "model", target)
    lm_head = getattr(target, "lm_head", None)

    embed = None
    try:
        embed = target.get_input_embeddings()
    except Exception:
        embed = None
    if embed is None:
        for n, m in target.named_modules():
            if isinstance(m, nn.Embedding) and m.embedding_dim >= 2048:
                embed = m
                break
    if embed is None:
        raise RuntimeError("could not locate target input embedding")

    backbone = embed.embedding_dim
    normalizer = math.sqrt(backbone)
    return target_base, lm_head, embed, normalizer


def build_target_signals(target, input_ids, attention_mask):
    """Frozen target forward -> last_hidden + shared_kv_states + soft-label logits.

    Caller wraps this in torch.no_grad().
    """
    target_base, lm_head, _, _ = locate_target_parts(target)

    base_out = target_base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_shared_kv_states=True,
        use_cache=False,
    )
    last_hidden = base_out.last_hidden_state
    shared_kv_states = base_out.shared_kv_states
    if shared_kv_states is None:
        raise RuntimeError(
            "target returned shared_kv_states=None; assistant requires it "
            "(ensure return_shared_kv_states=True is honored)."
        )
    if lm_head is not None:
        target_logits = lm_head(last_hidden)
    else:
        target_logits = target(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False).logits
    return {
        "last_hidden": last_hidden,
        "shared_kv_states": shared_kv_states,
        "target_logits": target_logits,
    }


def _assistant_step(assistant, target_embed, normalizer, token_ids, hidden,
                    shared_kv_states, attention_mask):
    """One draft forward. Returns (draft_logits, backbone_hidden).

    combined = concat(target_embed(token_ids) * normalizer, hidden)  # (B,T,2H)
    """
    import torch

    # NO manual normalizer: Gemma4's get_input_embeddings() is a
    # Gemma4TextScaledWordEmbedding that already multiplies by sqrt(hidden_size)
    # internally (embed_scale). Multiplying again would over-scale by ~53x and
    # destroy the draft (confirmed vs transformers v5.13.0 modeling_gemma4
    # line 1608 + official SinglePositionMultiTokenCandidateGenerator).
    tok_embed = target_embed(token_ids)                       # (B, T, H)
    combined = torch.cat([tok_embed, hidden], dim=-1)          # (B, T, 2H)
    out = assistant(
        inputs_embeds=combined,
        shared_kv_states=shared_kv_states,
        position_ids=None,
        attention_mask=attention_mask,
    )
    return out.logits, out.last_hidden_state


def _step_loss(draft_logits, target_logits, hard_targets, mask, cfg):
    """Soft-CE (KL to target) + optional hard-CE at one aligned step.

    Memory-critical: the vocab is 262144, so materializing softmax over every
    (B, L) position blows up GPU memory (the OOM at log_softmax). Only mask==1
    positions (assistant-response tokens) are supervised, so we GATHER those
    rows first and compute the 262144-wide softmax ONLY on them. For long
    prompts with short responses this cuts the softmax activation by 10x+.
    """
    import torch
    import torch.nn.functional as F

    temp = cfg.temperature
    B, L, V = draft_logits.shape
    flat_mask = mask.reshape(-1).bool()                        # (B*L,)
    n_sup = int(flat_mask.sum())
    if n_sup == 0:
        # No supervised tokens this step; return a differentiable zero.
        z = draft_logits.sum() * 0.0
        return z, {"soft_ce": z.detach()}

    d_flat = draft_logits.reshape(-1, V)[flat_mask]            # (n_sup, V)
    t_flat = target_logits.reshape(-1, V)[flat_mask]           # (n_sup, V)

    with torch.no_grad():
        soft_t = F.softmax(t_flat / temp, dim=-1)
    log_p = F.log_softmax(d_flat / temp, dim=-1)
    soft_ce = -(soft_t * log_p).sum(dim=-1).mean() * (temp * temp)

    total = cfg.soft_ce_weight * soft_ce
    out = {"soft_ce": soft_ce.detach()}
    if cfg.hard_ce_weight > 0:
        ht = hard_targets.reshape(-1)[flat_mask]               # (n_sup,)
        hard_ce = F.cross_entropy(d_flat, ht)
        total = total + cfg.hard_ce_weight * hard_ce
        out["hard_ce"] = hard_ce.detach()
    return total, out


def training_step(target, assistant, batch, cfg: MTPLossConfig):
    """One TTT training step. Returns (loss, metrics).

    batch: input_ids, attention_mask, loss_mask (all (B, T)).
    Target frozen (no grad); assistant trained.
    """
    import torch

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    loss_mask = batch["loss_mask"]
    B, T = input_ids.shape

    _, _, target_embed, normalizer = locate_target_parts(target)

    with torch.no_grad():
        signals = build_target_signals(target, input_ids, attention_mask)
    target_last_hidden = signals["last_hidden"]        # (B, T, H)
    shared_kv_states = signals["shared_kv_states"]
    target_logits = signals["target_logits"]           # (B, T, V)

    weights = compute_step_weights(cfg)
    K = cfg.ttt_steps

    total_loss = torch.zeros((), device=input_ids.device)
    metrics: dict[str, object] = {}

    # Recurrent hidden fed to the draft; starts as the target's last hidden.
    hidden = target_last_hidden                        # (B, T, H) indexed by t

    for k in range(K):
        # At step k, position t consumes token[t+k] and predicts token[t+k+1].
        # Valid positions: t in [0, T-k-2]  (need t+k+1 <= T-1).
        L = T - k - 1
        if L <= 0:
            break
        # Token consumed this step: input_ids[:, k : k+L]  (position t -> t+k).
        token_ids_k = input_ids[:, k:k + L]                       # (B, L)
        hidden_k = hidden[:, :L, :]                               # (B, L, H)

        draft_logits, backbone_hidden = _assistant_step(
            assistant, target_embed, normalizer,
            token_ids_k, hidden_k, shared_kv_states, None,
        )                                                        # (B, L, V), (B, L, H)

        # Supervision: draft_logits[:, t] should predict token[t+k+1] with the
        # target's distribution at position t+k.
        tgt_logits_k = target_logits[:, k:k + L, :]               # (B, L, V)
        hard_targets_k = input_ids[:, k + 1:k + 1 + L]            # (B, L)
        mask_k = loss_mask[:, k + 1:k + 1 + L]                    # (B, L)

        step_loss, step_metrics = _step_loss(
            draft_logits, tgt_logits_k, hard_targets_k, mask_k, cfg,
        )
        total_loss = total_loss + weights[k] * step_loss
        metrics[f"step{k}_soft_ce"] = step_metrics["soft_ce"]
        if "hard_ce" in step_metrics:
            metrics[f"step{k}_hard_ce"] = step_metrics["hard_ce"]

        # Recurrent feedback: next step consumes this step's backbone hidden.
        # backbone_hidden is indexed by t (query position); pad back to full T
        # so slicing lines up next iteration.
        if k + 1 < K:
            pad = target_last_hidden[:, L:, :]                    # tail filler
            hidden = torch.cat([backbone_hidden, pad], dim=1)     # (B, T, H)

    metrics["loss"] = total_loss.detach()
    return total_loss, metrics


def training_step_from_cache(assistant, target_embed, target_lm_head, batch,
                             cfg: MTPLossConfig):
    """TTT training step using PRECOMPUTED target signals (no 26B forward).

    batch (from the cache collate) provides:
      input_ids   (B, T)          loss_mask   (B, T)
      last_hidden (B, T, H)        shared_kv_states dict of (K,V)
    Target soft labels are recomputed on the fly as target_lm_head(last_hidden)
    ONLY on supervised positions (inside _step_loss, after masking). lm_head is
    frozen (tied to embed), so this is a cheap matmul and gives the FULL target
    distribution. Only the assistant is trained.
    """
    import torch

    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    last_hidden = batch["last_hidden"]
    shared_kv_states = batch["shared_kv_states"]
    B, T = input_ids.shape

    weights = compute_step_weights(cfg)
    K = cfg.ttt_steps
    total_loss = torch.zeros((), device=input_ids.device)
    metrics: dict[str, object] = {}
    hidden = last_hidden

    for k in range(K):
        L = T - k - 1
        if L <= 0:
            break
        token_ids_k = input_ids[:, k:k + L]
        hidden_k = hidden[:, :L, :]
        draft_logits, backbone_hidden = _assistant_step(
            assistant, target_embed, None, token_ids_k, hidden_k,
            shared_kv_states, None)
        hard_targets_k = input_ids[:, k + 1:k + 1 + L]
        mask_k = loss_mask[:, k + 1:k + 1 + L]

        # Target soft labels: lm_head over the TARGET's cached hidden at the
        # supervised positions [k, k+L) — NOT the recurrent draft hidden. This
        # mirrors the online path (target_logits[:, k:k+L]). Cheap frozen matmul;
        # _step_loss then gathers mask==1 rows.
        with torch.no_grad():
            tgt_hidden_k = last_hidden[:, k:k + L, :]              # (B, L, H)
            tgt_logits_k = target_lm_head(tgt_hidden_k)           # (B, L, V)

        step_loss, sm = _step_loss(
            draft_logits, tgt_logits_k, hard_targets_k, mask_k, cfg)
        total_loss = total_loss + weights[k] * step_loss
        metrics[f"step{k}_soft_ce"] = sm["soft_ce"]
        if "hard_ce" in sm:
            metrics[f"step{k}_hard_ce"] = sm["hard_ce"]

        if k + 1 < K:
            pad = last_hidden[:, L:, :]
            hidden = torch.cat([backbone_hidden, pad], dim=1)

    metrics["loss"] = total_loss.detach()
    return total_loss, metrics

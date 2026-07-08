"""Core MTP fine-tuning step for the Gemma 4 assistant.

Implements the forward + loss described in docs/TRAINING_DESIGN.md, using the
interface VERIFIED by scripts/debug_gemma_assistant.py:

    target base (frozen, return_shared_kv_states=True)
      -> last_hidden (B,T,2816), shared_kv_states {sliding:(K,V), full:(K,V)}
    inputs_embeds = concat(last_hidden, last_hidden)  (B,T,5632)
    assistant(inputs_embeds, shared_kv_states) -> draft_logits (B,T,262144)

Training objective: single-step distillation. The draft, at position t, should
predict token t+1 with the same distribution the target would. We therefore
distill the draft's next-token logits against the target's next-token logits
(soft cross-entropy / KL), masked to assistant-response tokens only.

This module is import-safe without a GPU (torch is imported lazily inside the
functions that need it) so it can be syntax-checked locally.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MTPLossConfig:
    """Hyperparameters for the distillation loss."""

    temperature: float = 1.0
    # Weight on the soft-CE (distillation) term.
    soft_ce_weight: float = 1.0
    # Optional weight on hard-CE against the ground-truth next token (0 = off).
    hard_ce_weight: float = 0.0
    ignore_index: int = -100


def build_target_signals(target, input_ids, attention_mask):
    """Run the frozen target base to get last_hidden + shared_kv_states + logits.

    Returns a dict with:
      - last_hidden:      (B, T, H)      target final hidden state
      - shared_kv_states: dict           {layer_type: (K, V)}  (opaque, passed on)
      - target_logits:    (B, T, V)      target next-token logits (soft labels)

    The target model is assumed frozen; caller wraps this in torch.no_grad().
    ``target`` is the full causal LM (e.g. Gemma4ForConditionalGeneration); we
    use its base model for hidden/KV and its lm_head for the soft labels.
    """
    # Locate the base model (returns shared_kv_states) and the lm_head.
    target_base = getattr(target, "model", target)
    lm_head = getattr(target, "lm_head", None)

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
            "target returned shared_kv_states=None; the assistant requires it. "
            "Ensure return_shared_kv_states=True is honored by this model."
        )

    if lm_head is not None:
        target_logits = lm_head(last_hidden)
    else:
        # Some wrappers expose logits directly; fall back to a full forward.
        full = target(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False)
        target_logits = full.logits

    return {
        "last_hidden": last_hidden,
        "shared_kv_states": shared_kv_states,
        "target_logits": target_logits,
    }


def assistant_forward(assistant, last_hidden, shared_kv_states,
                      position_ids=None, attention_mask=None):
    """Run the (trainable) assistant on target signals -> draft logits.

    inputs_embeds is the verified concat(last_hidden, last_hidden).
    """
    import torch

    inputs_embeds = torch.cat([last_hidden, last_hidden], dim=-1)
    out = assistant(
        inputs_embeds=inputs_embeds,
        shared_kv_states=shared_kv_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    return out.logits


def distillation_loss(draft_logits, target_logits, input_ids, loss_mask,
                      cfg: MTPLossConfig):
    """Single-step next-token distillation loss.

    Alignment: at position t the draft predicts token t+1. So we compare
    draft_logits[:, :-1] against target_logits[:, :-1] (both predicting the
    token at t+1), and optionally against the ground-truth input_ids[:, 1:].

    loss_mask is 1 on supervised (assistant-response) positions. We shift it to
    the "prediction" positions (t where t+1 is a response token).
    """
    import torch
    import torch.nn.functional as F

    # Predict positions 0..T-2 (each predicts the next token).
    d_logits = draft_logits[:, :-1, :]          # (B, T-1, V)
    t_logits = target_logits[:, :-1, :]         # (B, T-1, V)
    # Supervision mask: position t is supervised iff token t+1 is a response tok.
    mask = loss_mask[:, 1:].to(d_logits.dtype)  # (B, T-1)

    temp = cfg.temperature
    with torch.no_grad():
        soft_targets = F.softmax(t_logits / temp, dim=-1)
    log_probs = F.log_softmax(d_logits / temp, dim=-1)
    # Soft cross-entropy per position, summed over vocab.
    per_pos_soft = -(soft_targets * log_probs).sum(dim=-1)  # (B, T-1)
    denom = mask.sum().clamp(min=1.0)
    soft_ce = (per_pos_soft * mask).sum() / denom
    # Temperature scaling keeps gradients comparable to hard CE.
    soft_ce = soft_ce * (temp * temp)

    total = cfg.soft_ce_weight * soft_ce
    metrics = {"soft_ce": soft_ce.detach()}

    if cfg.hard_ce_weight > 0:
        # Ground-truth next token = input_ids shifted left.
        hard_targets = input_ids[:, 1:].clone()               # (B, T-1)
        hard_targets[mask == 0] = cfg.ignore_index
        hard_ce = F.cross_entropy(
            d_logits.reshape(-1, d_logits.size(-1)),
            hard_targets.reshape(-1),
            ignore_index=cfg.ignore_index,
        )
        total = total + cfg.hard_ce_weight * hard_ce
        metrics["hard_ce"] = hard_ce.detach()

    metrics["loss"] = total.detach()
    return total, metrics


def training_step(target, assistant, batch, cfg: MTPLossConfig):
    """One training step. Returns (loss, metrics).

    ``batch`` provides input_ids, attention_mask, loss_mask (all (B, T)).
    Target is frozen (no grad); assistant is trained.
    """
    import torch

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    loss_mask = batch["loss_mask"]

    with torch.no_grad():
        signals = build_target_signals(target, input_ids, attention_mask)

    draft_logits = assistant_forward(
        assistant,
        signals["last_hidden"],
        signals["shared_kv_states"],
        position_ids=None,
        attention_mask=attention_mask,
    )
    loss, metrics = distillation_loss(
        draft_logits, signals["target_logits"], input_ids, loss_mask, cfg,
    )
    return loss, metrics

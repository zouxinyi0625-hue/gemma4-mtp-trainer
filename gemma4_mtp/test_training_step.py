"""Local unit tests for the MTP training step (no real models, no GPU).

Uses tiny mock modules that mimic the VERIFIED interface:
  - target.model(input_ids, ..., return_shared_kv_states=True) ->
      object with .last_hidden_state and .shared_kv_states
  - target.lm_head(hidden) -> logits
  - assistant(inputs_embeds, shared_kv_states, ...) -> object with .logits

These tests check the LOGIC (shapes, masking, alignment, backprop), which is
exactly what we can validate without the 26B model or a GPU. Numerical values
come from tiny random tensors, not from any real model.

Run:  python -m pytest gemma4_mtp/test_training_step.py -q
  or: python gemma4_mtp/test_training_step.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gemma4_mtp.training_step import (
    MTPLossConfig,
    build_target_signals,
    distillation_loss,
    training_step,
)

VOCAB = 37
HID = 8          # target hidden (stands in for 2816)
T = 6
B = 2


class _BaseOut:
    def __init__(self, last_hidden_state, shared_kv_states):
        self.last_hidden_state = last_hidden_state
        self.shared_kv_states = shared_kv_states


class MockTargetBase(nn.Module):
    """Mimics Gemma4Model: returns last_hidden + shared_kv_states."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HID)

    def forward(self, input_ids=None, attention_mask=None,
                return_shared_kv_states=False, use_cache=False):
        h = self.embed(input_ids)
        skv = None
        if return_shared_kv_states:
            k = torch.randn(input_ids.size(0), 1, input_ids.size(1), HID)
            v = torch.randn(input_ids.size(0), 1, input_ids.size(1), HID)
            skv = {"sliding_attention": (k, v), "full_attention": (k, v)}
        return _BaseOut(h, skv)


class MockTarget(nn.Module):
    """Mimics Gemma4ForConditionalGeneration: .model + .lm_head."""

    def __init__(self):
        super().__init__()
        self.model = MockTargetBase()
        self.lm_head = nn.Linear(HID, VOCAB, bias=False)


class _AsstOut:
    def __init__(self, logits):
        self.logits = logits


class MockAssistant(nn.Module):
    """Mimics Gemma4AssistantForCausalLM: pre_projection(2*HID->HID) + head."""

    def __init__(self):
        super().__init__()
        self.pre_projection = nn.Linear(2 * HID, HID, bias=False)
        self.decoder = nn.Linear(HID, HID)      # stand-in for 4 layers
        self.lm_head = nn.Linear(HID, VOCAB, bias=False)

    def forward(self, inputs_embeds=None, shared_kv_states=None,
                position_ids=None, attention_mask=None):
        assert inputs_embeds.size(-1) == 2 * HID, "inputs_embeds must be 2*HID"
        assert shared_kv_states is not None, "assistant requires shared_kv_states"
        h = self.pre_projection(inputs_embeds)
        h = torch.tanh(self.decoder(h))
        return _AsstOut(self.lm_head(h))


def _batch():
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    loss_mask = torch.ones(B, T, dtype=torch.long)
    loss_mask[:, :2] = 0        # first 2 tokens are "prompt": not supervised
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "loss_mask": loss_mask}


def test_build_target_signals_shapes():
    target = MockTarget()
    b = _batch()
    with torch.no_grad():
        sig = build_target_signals(target, b["input_ids"], b["attention_mask"])
    assert sig["last_hidden"].shape == (B, T, HID)
    assert sig["target_logits"].shape == (B, T, VOCAB)
    assert set(sig["shared_kv_states"].keys()) == {"sliding_attention", "full_attention"}
    print("✅ build_target_signals shapes OK")


def test_distillation_loss_masks_and_backprops():
    target = MockTarget()
    assistant = MockAssistant()
    b = _batch()
    cfg = MTPLossConfig(temperature=1.0, soft_ce_weight=1.0, hard_ce_weight=0.5)
    loss, metrics = training_step(target, assistant, b, cfg)
    assert loss.requires_grad, "loss must be differentiable"
    assert loss.ndim == 0, "loss must be scalar"
    loss.backward()
    # Assistant params must receive gradients; target must NOT.
    asst_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in assistant.parameters())
    tgt_grad = any(p.grad is not None for p in target.parameters())
    assert asst_grad, "assistant should get gradients"
    assert not tgt_grad, "target should be frozen (no grad in step)"
    assert "soft_ce" in metrics and "hard_ce" in metrics
    print(f"✅ loss={loss.item():.4f} soft_ce={metrics['soft_ce'].item():.4f} "
          f"hard_ce={metrics['hard_ce'].item():.4f}; grads routed correctly")


def test_fully_masked_batch_is_safe():
    """If no positions are supervised, loss must be finite (no div-by-zero)."""
    target = MockTarget()
    assistant = MockAssistant()
    b = _batch()
    b["loss_mask"] = torch.zeros(B, T, dtype=torch.long)
    cfg = MTPLossConfig(hard_ce_weight=0.0)
    loss, _ = training_step(target, assistant, b, cfg)
    assert torch.isfinite(loss), "loss must be finite even with empty mask"
    print("✅ fully-masked batch produces finite loss (no div-by-zero)")


def test_perfect_match_gives_low_soft_ce():
    """If draft logits == target logits, soft-CE should equal the target's entropy
    (finite, and lower than a mismatched case)."""
    torch.manual_seed(1)
    t_logits = torch.randn(B, T, VOCAB)
    d_logits_match = t_logits.clone().requires_grad_(True)
    d_logits_bad = torch.randn(B, T, VOCAB, requires_grad=True)
    ids = torch.randint(0, VOCAB, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    cfg = MTPLossConfig(hard_ce_weight=0.0)
    good, _ = distillation_loss(d_logits_match, t_logits, ids, mask, cfg)
    bad, _ = distillation_loss(d_logits_bad, t_logits, ids, mask, cfg)
    assert good < bad, f"matched logits should give lower soft-CE ({good} vs {bad})"
    print(f"✅ matched soft_ce={good.item():.4f} < mismatched={bad.item():.4f}")


if __name__ == "__main__":
    test_build_target_signals_shapes()
    test_distillation_loss_masks_and_backprops()
    test_fully_masked_batch_is_safe()
    test_perfect_match_gives_low_soft_ce()
    print("\nAll local logic tests passed.")

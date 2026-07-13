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

import json

import torch
import torch.nn as nn

from gemma4_mtp.training_step import (
    MTPLossConfig,
    build_target_signals,
    compute_step_weights,
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
    """Mimics Gemma4ForConditionalGeneration: .model + .lm_head + embed."""

    def __init__(self):
        super().__init__()
        self.model = MockTargetBase()
        self.lm_head = nn.Linear(HID, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.model.embed


class _AsstOut:
    def __init__(self, logits, last_hidden_state):
        self.logits = logits
        self.last_hidden_state = last_hidden_state


class MockAssistant(nn.Module):
    """Mimics Gemma4AssistantForCausalLM: pre_projection(2*HID->HID) + head.

    Returns logits + last_hidden_state (the backbone-dim hidden fed back in TTT).
    """

    def __init__(self):
        super().__init__()
        self.pre_projection = nn.Linear(2 * HID, HID, bias=False)
        self.decoder = nn.Linear(HID, HID)      # stand-in for 4 layers
        self.lm_head = nn.Linear(HID, VOCAB, bias=False)
        self.post_projection = nn.Linear(HID, HID, bias=False)

    def forward(self, inputs_embeds=None, shared_kv_states=None,
                position_ids=None, attention_mask=None):
        assert inputs_embeds.size(-1) == 2 * HID, "inputs_embeds must be 2*HID"
        assert shared_kv_states is not None, "assistant requires shared_kv_states"
        h = self.pre_projection(inputs_embeds)
        h = torch.tanh(self.decoder(h))
        backbone = self.post_projection(h)      # (B, L, HID)
        return _AsstOut(self.lm_head(h), backbone)


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


def test_ttt_masks_and_backprops():
    target = MockTarget()
    assistant = MockAssistant()
    # Freeze the target exactly like train.set_trainable does in real training.
    for p in target.parameters():
        p.requires_grad_(False)
    b = _batch()
    cfg = MTPLossConfig(ttt_steps=3, temperature=1.0,
                        soft_ce_weight=1.0, hard_ce_weight=0.5)
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
    # TTT produces per-step metrics for each of the 3 steps.
    assert "step0_soft_ce" in metrics and "step2_soft_ce" in metrics
    assert "step0_hard_ce" in metrics
    print(f"✅ TTT loss={loss.item():.4f} "
          f"step0_soft={metrics['step0_soft_ce'].item():.4f} "
          f"step2_soft={metrics['step2_soft_ce'].item():.4f}; grads routed correctly")


def test_fully_masked_batch_is_safe():
    """If no positions are supervised, loss must be finite (no div-by-zero)."""
    target = MockTarget()
    assistant = MockAssistant()
    b = _batch()
    b["loss_mask"] = torch.zeros(B, T, dtype=torch.long)
    cfg = MTPLossConfig(ttt_steps=3, hard_ce_weight=0.0)
    loss, _ = training_step(target, assistant, b, cfg)
    assert torch.isfinite(loss), "loss must be finite even with empty mask"
    print("✅ fully-masked batch produces finite loss (no div-by-zero)")


def test_step_weights_normalized():
    """Default decaying step weights sum to 1 and decrease."""
    cfg = MTPLossConfig(ttt_steps=5, step_weight_beta=0.8)
    w = compute_step_weights(cfg)
    assert len(w) == 5
    assert abs(sum(w) - 1.0) < 1e-6, f"weights should sum to 1, got {sum(w)}"
    assert all(w[i] > w[i+1] for i in range(4)), "weights should decay"
    # Explicit weights are passed through.
    cfg2 = MTPLossConfig(ttt_steps=3, step_weights=[1.0, 0.5, 0.25])
    assert compute_step_weights(cfg2) == [1.0, 0.5, 0.25]
    print(f"✅ step weights normalized+decaying: "
          f"{[round(x,3) for x in w]}")


def test_collate_pads_and_aligns():
    """collate right-pads variable-length samples and keeps masks aligned."""
    import torch
    from gemma4_mtp.data import collate

    pad = 999
    a = {"input_ids": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1]),
         "loss_mask": torch.tensor([0, 1, 1])}
    b = {"input_ids": torch.tensor([4, 5]),
         "attention_mask": torch.tensor([1, 1]),
         "loss_mask": torch.tensor([0, 1])}
    out = collate([a, b], pad_token_id=pad)
    assert out["input_ids"].shape == (2, 3)
    # shorter sample padded with pad id, and its pad positions unmasked.
    assert out["input_ids"][1, 2].item() == pad
    assert out["attention_mask"][1, 2].item() == 0
    assert out["loss_mask"][1, 2].item() == 0
    # real content preserved
    assert out["input_ids"][0].tolist() == [1, 2, 3]
    assert out["loss_mask"][0].tolist() == [0, 1, 1]
    print("✅ collate pads + aligns masks correctly")


def test_iter_jsonl_roundtrip():
    """iter_jsonl reads one object per line, skipping blanks."""
    import os
    import tempfile
    from gemma4_mtp.data import iter_jsonl

    rows = [
        {"conversations": [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello"}],
         "status": "success"},
        {"conversations": [{"role": "user", "content": "x"}], "status": "error"},
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write("\n")  # blank line should be skipped
        got = list(iter_jsonl(path))
        assert len(got) == 2
        assert got[0]["status"] == "success"
        assert got[1]["status"] == "error"
        print("✅ iter_jsonl round-trips + skips blank lines")
    finally:
        os.remove(path)


def test_freeze_policy():
    """set_trainable freezes target + assistant lm_head/embed, trains the rest."""
    from gemma4_mtp.train import set_trainable

    target = MockTarget()
    assistant = MockAssistant()
    # MockAssistant has no .model.embed_tokens; add a matching structure so the
    # freeze logic can find the tied embed like the real assistant.
    assistant.model = nn.Module()
    assistant.model.embed_tokens = nn.Embedding(VOCAB, HID)

    trainable = set_trainable(target, assistant)

    # target fully frozen
    assert all(not p.requires_grad for p in target.parameters())
    # assistant lm_head frozen
    assert all(not p.requires_grad for p in assistant.lm_head.parameters())
    # assistant embed frozen
    assert all(not p.requires_grad for p in assistant.model.embed_tokens.parameters())
    # decoder + projections trained
    assert all(p.requires_grad for p in assistant.pre_projection.parameters())
    assert all(p.requires_grad for p in assistant.decoder.parameters())
    # returned list is non-empty and only trainable params
    assert len(trainable) > 0 and all(p.requires_grad for p in trainable)
    print(f"✅ freeze policy correct ({len(trainable)} trainable tensors)")


if __name__ == "__main__":
    test_build_target_signals_shapes()
    test_ttt_masks_and_backprops()
    test_fully_masked_batch_is_safe()
    test_step_weights_normalized()
    test_collate_pads_and_aligns()
    test_iter_jsonl_roundtrip()
    test_freeze_policy()
    print("\nAll local logic tests passed.")

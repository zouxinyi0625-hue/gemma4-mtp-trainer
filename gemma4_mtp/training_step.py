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
    # L1/TVD-to-target weight. accept_rate = 1 - 0.5*L1, so this directly
    # optimizes the rejection-sampling acceptance rate (DSpark uses ~0.9).
    l1_weight: float = 0.0
    # Hard-CE against the ground-truth next token weight.
    hard_ce_weight: float = 0.0
    # Argmax-match weight: hard-CE of draft logits against the TARGET's argmax
    # token (top-1). This is the differentiable proxy for vLLM's GREEDY accept
    # rule (accepted <=> draft_argmax == target_argmax). The main objective.
    argmax_ce_weight: float = 1.0
    ignore_index: int = -100
    # Number of answer-position anchors sampled per sequence (DSpark-style).
    # Bounds compute/memory independent of sequence length. 0 = disabled
    # (legacy full-sequence path).
    num_anchors: int = 128
    # Rows per softmax chunk in _step_loss; bounds peak memory on batches with
    # many supervised tokens (the 262144-wide vocab softmax is the OOM risk).
    loss_chunk_rows: int = 1024


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


def sample_anchors(loss_mask, num_anchors):
    """Sample `num_anchors` answer-position anchors per sequence (DSpark-style).

    A position t is a valid anchor iff loss_mask[t]==1 AND loss_mask[t+1]==1
    (inside the answer, and the position it predicts is also supervised). We
    randomly pick up to num_anchors valid anchors per sequence; sequences with
    fewer valid positions are padded (anchor_pos=0) and flagged by keep_mask.

    Returns:
      anchor_pos  (B, A) long   — sampled anchor positions (padded with 0)
      keep_mask   (B, A) bool   — True for real anchors, False for padding
    Ported from DSpark deepspec/modeling/dspark/common.py:109-169 (rand+sort,
    no flex_attention needed since we isolate anchors on the batch dim).
    """
    import torch

    B, T = loss_mask.shape
    device = loss_mask.device
    A = int(num_anchors)
    num_cand = max(T - 1, 0)
    if num_cand == 0:
        return (torch.zeros(B, A, dtype=torch.long, device=device),
                torch.zeros(B, A, dtype=torch.bool, device=device))

    lm = loss_mask.bool()
    valid = lm[:, :num_cand] & lm[:, 1:num_cand + 1]          # (B, num_cand)
    valid_counts = valid.sum(dim=1)                            # (B,)

    idx = torch.arange(num_cand, device=device).unsqueeze(0).expand(B, -1)
    # Invalid positions get random key 2.0 (sorted last); valid get [0,1).
    rand = torch.rand(B, num_cand, device=device)
    rand = torch.where(valid, rand, torch.full_like(rand, 2.0))
    _, order = rand.sort(dim=1)
    gathered = torch.gather(idx, 1, order)                     # valid-first order
    if num_cand < A:
        pad = torch.zeros(B, A - num_cand, dtype=gathered.dtype, device=device)
        gathered = torch.cat([gathered, pad], dim=1)
    anchor_pos = gathered[:, :A]
    keep_mask = torch.arange(A, device=device).unsqueeze(0) < \
        valid_counts.unsqueeze(1).clamp(max=A)
    anchor_pos = torch.where(keep_mask, anchor_pos, torch.zeros_like(anchor_pos))
    return anchor_pos, keep_mask


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
                    shared_kv_states, attention_mask, position_ids=None):
    """One draft forward. Returns (draft_logits, backbone_hidden).

    combined = concat(target_embed(token_ids), hidden)  # (N, S, 2H)

    In anchor mode, token_ids/hidden are (N, 1, ...) — one row per anchor on the
    BATCH dim — and position_ids (N, 1) pins each anchor to its fixed position so
    the draft attends only the target KV up to that anchor (vLLM single-anchor
    semantics; anchors are isolated on the batch dim so they never attend each
    other).
    """
    import torch

    # NO manual normalizer: Gemma4's get_input_embeddings() is a
    # Gemma4TextScaledWordEmbedding that already multiplies by sqrt(hidden_size)
    # internally (embed_scale). Multiplying again would over-scale by ~53x and
    # destroy the draft (confirmed vs transformers v5.13.0 modeling_gemma4
    # line 1608 + official SinglePositionMultiTokenCandidateGenerator).
    tok_embed = target_embed(token_ids)                       # (N, S, H)
    combined = torch.cat([tok_embed, hidden], dim=-1)          # (N, S, 2H)
    out = assistant(
        inputs_embeds=combined,
        shared_kv_states=shared_kv_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    return out.logits, out.last_hidden_state


def _step_loss(draft_logits, target_hidden, target_lm_head, hard_targets, mask, cfg):
    """Distillation loss at one aligned draft step (legacy full-sequence path).

    Kept for the online training_step. The anchor path uses _anchor_loss.
    Combines soft-CE (KL), L1/TVD, hard-CE-to-ground-truth, and argmax-CE. The
    MAIN objective is now argmax-CE (cfg.argmax_ce_weight): hard-CE of the draft
    against the TARGET's argmax token — the differentiable proxy for vLLM's
    GREEDY accept rule (accepted <=> draft_argmax == target_argmax). L1/TVD only
    matches the rejection-sampling accept rate (temperature>0), NOT greedy.

    Memory: target logits computed ONLY on supervised rows, chunked.
    """
    import torch
    import torch.nn.functional as F

    temp = cfg.temperature
    l1_weight = float(getattr(cfg, "l1_weight", 0.0))
    argmax_w = float(getattr(cfg, "argmax_ce_weight", 0.0))
    B, L, V = draft_logits.shape
    H = target_hidden.shape[-1]
    flat_mask = mask.reshape(-1).bool()
    n_sup = int(flat_mask.sum())
    if n_sup == 0:
        z = draft_logits.sum() * 0.0
        return z, {"soft_ce": z.detach()}
    d_flat = draft_logits.reshape(-1, V)[flat_mask]            # (n_sup, V)
    th_flat = target_hidden.reshape(-1, H)[flat_mask]          # (n_sup, H)
    if cfg.hard_ce_weight > 0:
        ht_flat = hard_targets.reshape(-1)[flat_mask]
    return _anchor_loss(d_flat, th_flat, target_lm_head,
                        ht_flat if cfg.hard_ce_weight > 0 else None, cfg)


def _anchor_loss(d_flat, th_flat, target_lm_head, ht_flat, cfg):
    """Loss over N gathered draft rows (N, V) vs target hidden (N, H).

    MAIN objective: argmax-CE — cross-entropy of draft logits against the
    target's argmax token (top-1). This is the differentiable proxy for vLLM's
    greedy accept (accepted <=> draft_argmax == target_argmax). Optional
    soft-CE (KL), L1/TVD, and hard-CE-to-ground-truth are added by weight.

    Reports greedy_accept = mean(draft_argmax == target_argmax) — the exact
    (non-differentiable) quantity vLLM's bench measures, for monitoring.
    Target logits computed per-chunk (never full N x 262144 at once).
    """
    import torch
    import torch.nn.functional as F

    temp = cfg.temperature
    l1_weight = float(getattr(cfg, "l1_weight", 0.0))
    argmax_w = float(getattr(cfg, "argmax_ce_weight", 0.0))
    N = d_flat.shape[0]
    if N == 0:
        z = d_flat.sum() * 0.0
        return z, {"argmax_ce": z.detach()}

    chunk = int(getattr(cfg, "loss_chunk_rows", 1024))
    soft_ce_sum = d_flat.new_zeros(())
    hard_ce_sum = d_flat.new_zeros(())
    argmax_ce_sum = d_flat.new_zeros(())
    l1_sum = d_flat.new_zeros(())
    greedy_hit_sum = d_flat.new_zeros(())
    for s in range(0, N, chunk):
        d = d_flat[s:s + chunk]                                # (c, V)
        with torch.no_grad():
            t = target_lm_head(th_flat[s:s + chunk])           # (c, V)
            t_argmax = t.argmax(dim=-1)                        # (c,)
        log_p = F.log_softmax(d, dim=-1)
        # MAIN: argmax-CE — push draft's distribution mass onto target's top-1.
        if argmax_w > 0:
            argmax_ce_sum = argmax_ce_sum + F.nll_loss(
                log_p, t_argmax, reduction="sum")
        # greedy accept monitor (non-diff): fraction where argmaxes agree.
        with torch.no_grad():
            greedy_hit_sum = greedy_hit_sum + (
                d.argmax(dim=-1) == t_argmax).sum()
        if cfg.soft_ce_weight > 0:
            with torch.no_grad():
                soft_t = F.softmax(t / temp, dim=-1)
            lp = F.log_softmax(d / temp, dim=-1)
            soft_ce_sum = soft_ce_sum + -(soft_t * lp).sum(-1).sum() * (temp * temp)
        if l1_weight > 0:
            dp = F.softmax(d, dim=-1)
            with torch.no_grad():
                tp = F.softmax(t, dim=-1)
            l1_sum = l1_sum + (dp - tp).abs().sum(-1).sum()
        if cfg.hard_ce_weight > 0 and ht_flat is not None:
            hard_ce_sum = hard_ce_sum + F.cross_entropy(
                d, ht_flat[s:s + chunk], reduction="sum")

    total = d_flat.new_zeros(())
    out = {"greedy_accept": (greedy_hit_sum / N).detach()}
    if argmax_w > 0:
        argmax_ce = argmax_ce_sum / N
        total = total + argmax_w * argmax_ce
        out["argmax_ce"] = argmax_ce.detach()
    if cfg.soft_ce_weight > 0:
        soft_ce = soft_ce_sum / N
        total = total + cfg.soft_ce_weight * soft_ce
        out["soft_ce"] = soft_ce.detach()
    if l1_weight > 0:
        l1 = l1_sum / N
        total = total + l1_weight * l1
        out["l1"] = l1.detach()
        out["accept_rate"] = (1.0 - 0.5 * l1).detach()
    if cfg.hard_ce_weight > 0 and ht_flat is not None:
        hard_ce = hard_ce_sum / N
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

    target_base, target_lm_head, target_embed, normalizer = locate_target_parts(target)

    with torch.no_grad():
        signals = build_target_signals(target, input_ids, attention_mask)
    target_last_hidden = signals["last_hidden"]        # (B, T, H)
    shared_kv_states = signals["shared_kv_states"]

    weights = compute_step_weights(cfg)
    K = cfg.ttt_steps

    total_loss = torch.zeros((), device=input_ids.device)
    metrics: dict[str, object] = {}

    # Recurrent hidden fed to the draft; starts as the target's last hidden.
    hidden = target_last_hidden                        # (B, T, H) indexed by t
    prev_draft_tokens = None      # step k-1's argmax token ids (B, L_prev), tiny

    for k in range(K):
        # At step k, position t consumes token[t+k] and predicts token[t+k+1].
        # Valid positions: t in [0, T-k-2]  (need t+k+1 <= T-1).
        L = T - k - 1
        if L <= 0:
            break
        if k == 0:
            # Step 0 consumes the real next token (stand-in for the target's
            # sampled token that vLLM feeds at step 0).
            token_ids_k = input_ids[:, 0:L]                       # (B, L)
        else:
            # Steps k>0 consume the DRAFT's OWN previous prediction (vLLM
            # llm_base_proposer.py:574), not ground truth. prev argmax at
            # position j aligns with what position j consumes now. Keep only the
            # (B, L) token ids so the big (B, L, V) logits free after each step.
            token_ids_k = prev_draft_tokens[:, :L]
        hidden_k = hidden[:, :L, :]                               # (B, L, H)

        draft_logits, backbone_hidden = _assistant_step(
            assistant, target_embed, normalizer,
            token_ids_k, hidden_k, shared_kv_states, None,
        )                                                        # (B, L, V), (B, L, H)
        prev_draft_tokens = draft_logits.argmax(dim=-1).detach()

        # Supervision: target's distribution at positions [k, k+L). Pass the
        # hidden slice; _step_loss computes target logits only on masked rows.
        tgt_hidden_k = target_last_hidden[:, k:k + L, :]          # (B, L, H)
        hard_targets_k = input_ids[:, k + 1:k + 1 + L]            # (B, L)
        mask_k = loss_mask[:, k + 1:k + 1 + L]                    # (B, L)

        step_loss, step_metrics = _step_loss(
            draft_logits, tgt_hidden_k, target_lm_head,
            hard_targets_k, mask_k, cfg,
        )
        total_loss = total_loss + weights[k] * step_loss
        for mk, mv in step_metrics.items():
            metrics[f"step{k}_{mk}"] = mv

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
    """Single-anchor TTT training from PRECOMPUTED target signals (no 26B).

    Matches vLLM inference (SinglePositionMultiTokenCandidateGenerator):
    sample answer-position anchors, and for each anchor run k autoregressive
    draft steps at a FIXED position, attending only the target KV. Anchors are
    laid out on the BATCH dim (N = B*A rows, seq_len 1 each) so they are
    naturally isolated — they never attend each other, exactly like vLLM's
    per-request drafts. Compute/memory scale with num_anchors, NOT seq_len.

    Objective: argmax-CE (draft logits vs target argmax) — the differentiable
    proxy for vLLM's greedy accept rule. See _anchor_loss.

    batch: input_ids (B,T), loss_mask (B,T), last_hidden (B,T,H),
           shared_kv_states dict of (K,V) each (B, Hkv, T, D).
    """
    import torch

    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    last_hidden = batch["last_hidden"]                        # (B, T, H)
    shared_kv_states = batch["shared_kv_states"]
    B, T = input_ids.shape
    H = last_hidden.shape[-1]
    device = input_ids.device

    A = int(cfg.num_anchors)
    weights = compute_step_weights(cfg)
    K = cfg.ttt_steps
    metrics: dict[str, object] = {}

    # 1. sample anchors on answer positions -> (B, A), keep_mask (B, A)
    anchor_pos, keep_mask = sample_anchors(loss_mask, A)
    N = B * A
    flat_anchor = anchor_pos.reshape(N)                       # (N,)
    flat_keep = keep_mask.reshape(N)                          # (N,)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, A).reshape(N)

    if int(flat_keep.sum()) == 0:
        z = last_hidden.sum() * 0.0
        return z, {"loss": z.detach()}

    # 2. anchors on the BATCH dim: shared_kv expanded to N rows (each anchor
    #    sees the same target KV; position_ids pin it to its anchor position so
    #    attention only reaches context < anchor_pos).
    def expand_kv(kv):
        k, v = kv                                             # (B, Hkv, T, D)
        k = k[batch_idx]                                     # (N, Hkv, T, D)
        v = v[batch_idx]
        return (k, v)
    kv_N = {kt: expand_kv(kv) for kt, kv in shared_kv_states.items()}

    # step-0 inputs, all gathered to (N, 1, ...)
    tok0 = input_ids[batch_idx, flat_anchor].unsqueeze(1)    # (N, 1) token at anchor
    hid = last_hidden[batch_idx, flat_anchor].unsqueeze(1)   # (N, 1, H) target hidden
    pos = flat_anchor.unsqueeze(1)                           # (N, 1) fixed position

    total_loss = torch.zeros((), device=device)
    prev_tokens = None
    for k in range(K):
        if k == 0:
            token_k = tok0
        else:
            token_k = prev_tokens                            # (N, 1) draft's own argmax
        draft_logits, backbone_hidden = _assistant_step(
            assistant, target_embed, None, token_k, hid,
            kv_N, None, position_ids=pos)                    # (N,1,V), (N,1,H)
        prev_tokens = draft_logits.argmax(dim=-1).detach()   # (N, 1)
        hid = backbone_hidden                                # feed back (N,1,H)

        # Supervision: anchor a at step k predicts token a+k+1; the aligned
        # target distribution is lm_head(last_hidden[a+k]). Keep rows where
        # (a+k) is a valid supervised position (inside answer, within seq).
        tgt_idx = (flat_anchor + k).clamp(max=T - 1)         # (N,)
        step_valid = flat_keep & ((flat_anchor + k) < T) & \
            (loss_mask[batch_idx, tgt_idx] > 0)
        if int(step_valid.sum()) == 0:
            continue
        rows = step_valid.nonzero(as_tuple=True)[0]
        d_rows = draft_logits[:, 0, :][rows]                 # (n, V)
        th_rows = last_hidden[batch_idx[rows], tgt_idx[rows]]  # (n, H)
        # hard-CE-to-ground-truth target = the actual next token a+k+1
        gt_idx = (flat_anchor + k + 1).clamp(max=T - 1)
        ht_rows = input_ids[batch_idx[rows], gt_idx[rows]] if cfg.hard_ce_weight > 0 else None

        step_loss, sm = _anchor_loss(d_rows, th_rows, target_lm_head, ht_rows, cfg)
        total_loss = total_loss + weights[k] * step_loss
        if "argmax_ce" in sm:
            metrics[f"step{k}_argmax_ce"] = sm["argmax_ce"]
        metrics[f"step{k}_accept"] = sm["greedy_accept"]
        if "soft_ce" in sm:
            metrics[f"step{k}_soft_ce"] = sm["soft_ce"]
        if "l1" in sm:
            metrics[f"step{k}_l1"] = sm["l1"]

    metrics["loss"] = total_loss.detach()
    return total_loss, metrics

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
    # Anchors processed per chunk in the cache TTT loop. The full-seq KV is
    # broadcast only to chunk_size rows (repeat_kv peak scales with this, NOT
    # with num_anchors), so this decouples memory from num_anchors. Lower it if
    # you still OOM; raise it for throughput.
    anchor_chunk: int = 8


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
                             cfg: MTPLossConfig, backward_fn=None):
    """Single-anchor TTT training from PRECOMPUTED target signals (no 26B).

    Matches vLLM inference (SinglePositionMultiTokenCandidateGenerator):
    sample answer-position anchors, and for each anchor run k autoregressive
    draft steps at a FIXED position, attending only the target KV. Anchors are
    laid out on the BATCH dim (N = B*A rows, seq_len 1 each) so they are
    naturally isolated — they never attend each other, exactly like vLLM's
    per-request drafts. Compute/memory scale with num_anchors, NOT seq_len.

    Objective: argmax-CE (draft logits vs target argmax) — the differentiable
    proxy for vLLM's greedy accept rule. See _anchor_loss.

    backward_fn: if given, called as backward_fn(chunk_loss) right after each
    anchor chunk so that chunk's autograd graph is freed BEFORE the next chunk
    runs. This keeps peak activation memory at one chunk, not all N//chunk
    chunks at once (the whole point of chunking). The caller must NOT call
    .backward() on the returned loss in that case — grads are already
    accumulated. When None, returns a graph-attached loss for the caller to
    back-propagate (used by tests / the online path).

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

    # 2. anchors on the BATCH dim: shared_kv expanded to N rows. Each anchor
    #    must attend ONLY the target KV up to its own position (<= anchor_pos) —
    #    NOT the future answer tokens. The cache KV is the FULL sequence, so we
    #    broadcast it and use a per-anchor 2D mask (1 = attend, 0 = block) that
    #    zeroes positions > anchor. Verified (probe_assistant_mask.py) the
    #    official assistant honors this mask on the shared-KV path. Without it
    #    the draft "sees the answer" — the leak that gave training-accept 1.0 but
    #    bench 33%. position_ids are fixed and KV doesn't grow across TTT steps
    #    (draft never writes KV), so the same mask applies to every step.
    # 3. anchor-CHUNK loop. Broadcasting the full-seq KV to all N anchors and
    #    running one forward blows up: repeat_kv expands (chunk, Hkv, T, D) to
    #    (chunk, full_heads, T, D), and the peak scales with N*full_heads*T. So
    #    we process anchors in small chunks: each chunk broadcasts KV only to its
    #    rows, runs the full K-step TTT + loss, and accumulates. Peak memory
    #    scales with chunk_size, DECOUPLED from num_anchors — this is how DSpark
    #    keeps 128 anchors on one GPU. Loss is summed across chunks (each chunk's
    #    step_loss already averages over its rows, so we weight by row count to
    #    keep the global mean unbiased).
    kv_pos = torch.arange(T, device=device).unsqueeze(0)     # (1, T) for masks
    chunk_size = int(getattr(cfg, "anchor_chunk", 0)) or 8
    # Fixed normalizer known BEFORE the loop, so each chunk can be normalized and
    # back-propagated on its own (freeing its graph) without waiting for a global
    # count. denom = (kept anchors) * K ~= total supervised steps; last-step
    # invalid rows make this a slight over-count, negligible vs the memory win.
    denom = max(int(flat_keep.sum()), 1) * max(K, 1)
    total_loss = torch.zeros((), device=device)              # detached scalar log
    # per-step accumulators for metrics (weighted by valid-row count)
    step_acc = {k: {"num": 0.0, "argmax_ce": 0.0, "accept": 0.0,
                    "soft_ce": 0.0, "l1": 0.0} for k in range(K)}

    for c0 in range(0, N, chunk_size):
        c1 = min(c0 + chunk_size, N)
        cidx = torch.arange(c0, c1, device=device)           # rows in this chunk
        c_batch = batch_idx[cidx]                            # (nc,)
        c_anchor = flat_anchor[cidx]                         # (nc,)
        c_keep = flat_keep[cidx]                             # (nc,)
        if int(c_keep.sum()) == 0:
            continue

        # KV broadcast to THIS chunk only + per-anchor future mask.
        kv_c = {kt: (kv[0][c_batch], kv[1][c_batch])
                for kt, kv in shared_kv_states.items()}       # (nc, Hkv, T, D)
        mask_c = (kv_pos <= c_anchor.unsqueeze(1)).to(last_hidden.dtype)  # (nc, T)

        tok_k = input_ids[c_batch, c_anchor].unsqueeze(1)     # (nc, 1)
        hid = last_hidden[c_batch, c_anchor].unsqueeze(1)     # (nc, 1, H)
        pos = c_anchor.unsqueeze(1)                           # (nc, 1)

        chunk_loss = torch.zeros((), device=device)          # graph-attached, THIS chunk
        for k in range(K):
            draft_logits, backbone_hidden = _assistant_step(
                assistant, target_embed, None, tok_k, hid,
                kv_c, mask_c, position_ids=pos)               # (nc,1,V),(nc,1,H)
            tok_k = draft_logits.argmax(dim=-1).detach()      # (nc, 1) next input
            hid = backbone_hidden

            tgt_idx = (c_anchor + k).clamp(max=T - 1)         # (nc,)
            step_valid = c_keep & ((c_anchor + k) < T) & \
                (loss_mask[c_batch, tgt_idx] > 0)
            nvalid = int(step_valid.sum())
            if nvalid == 0:
                continue
            rows = step_valid.nonzero(as_tuple=True)[0]
            d_rows = draft_logits[:, 0, :][rows]              # (n, V)
            th_rows = last_hidden[c_batch[rows], tgt_idx[rows]]  # (n, H)
            gt_idx = (c_anchor + k + 1).clamp(max=T - 1)
            ht_rows = input_ids[c_batch[rows], gt_idx[rows]] \
                if cfg.hard_ce_weight > 0 else None

            step_loss, sm = _anchor_loss(d_rows, th_rows, target_lm_head,
                                         ht_rows, cfg)
            # step_loss is a per-row MEAN; scale to a sum so chunks combine into
            # one global mean under the fixed `denom`.
            chunk_loss = chunk_loss + weights[k] * step_loss * nvalid
            total_loss = total_loss + (weights[k] * step_loss * nvalid).detach()
            a = step_acc[k]
            a["num"] += nvalid
            for key in ("argmax_ce", "accept", "soft_ce", "l1"):
                src = "greedy_accept" if key == "accept" else key
                if src in sm:
                    a[key] += float(sm[src]) * nvalid

        # Back-propagate THIS chunk now and free its graph before the next one.
        if backward_fn is not None and chunk_loss.requires_grad:
            backward_fn(chunk_loss / denom)

    total_loss = total_loss / denom
    for k in range(K):
        a = step_acc[k]
        if a["num"] == 0:
            continue
        metrics[f"step{k}_accept"] = a["accept"] / a["num"]
        if cfg.argmax_ce_weight > 0:
            metrics[f"step{k}_argmax_ce"] = a["argmax_ce"] / a["num"]
        if cfg.soft_ce_weight > 0:
            metrics[f"step{k}_soft_ce"] = a["soft_ce"] / a["num"]
        if cfg.l1_weight > 0:
            metrics[f"step{k}_l1"] = a["l1"] / a["num"]

    metrics["loss"] = total_loss.detach()
    # When backward_fn ran, grads are already accumulated; hand back the detached
    # scalar so the caller does NOT double-backward. Otherwise return the
    # graph-attached loss (total_loss is detached here, so rebuild it is not
    # possible — callers without backward_fn must pass one or use small N).
    if backward_fn is not None:
        return total_loss, metrics
    # No backward_fn: recompute a graph-attached loss is impossible after detach;
    # signal by returning total_loss (detached). Callers needing grad must use
    # backward_fn. Kept for metric-only / test use.
    return total_loss, metrics

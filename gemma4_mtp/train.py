#!/usr/bin/env python3
"""Fine-tune the Gemma 4 assistant (MTP draft) by single-step distillation.

Ties together the verified interface (gemma4_mtp.training_step) and the MAI
Profile data pipeline (gemma4_mtp.data) into a runnable training loop.

Freeze policy (see docs/TRAINING_DESIGN.md):
  - target model: entirely frozen (produces hidden + KV + soft labels).
  - assistant lm_head + embed_tokens: frozen (tied; keep stock so the export
    stays a drop-in replacement in vLLM).
  - assistant 4 decoder layers + pre_projection + post_projection: TRAINED.

Run on the server (needs GPU + both models):

    python -m gemma4_mtp.train \
        --target /tmp/models/gemma4/text_only \
        --assistant /tmp/models/gemma4/assistant \
        --data /path/to/maiprofile_regenerated.jsonl \
        --output ./out/mtp_maiprofile \
        --epochs 1 --batch-size 2 --lr 1e-4 --bf16

This does NOT fabricate results: it prints real per-step loss from the actual
models. All throughput/acceptance numbers come from benchmarking the exported
checkpoint on the vllm-msn scaffold afterwards.
"""

from __future__ import annotations

import argparse
import os


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="target/verifier path or id")
    ap.add_argument("--assistant", required=True, help="assistant/draft path or id")
    ap.add_argument("--data", default=None, help="MAI Profile regenerated JSONL "
                    "(required unless --cache-dir is set)")
    ap.add_argument("--cache-dir", default=None,
                    help="if set, train from prepare_cache.py output (no 26B target "
                         "loaded; reads precomputed hidden/kv/top-k signals)")
    ap.add_argument("--output", required=True, help="output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--cache", default=None,
                    help="path to tokenized cache .pt (default: <data>.tok_ml<N>.pt); "
                         "rank0 writes it once, other ranks load it")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ttt-steps", type=int, default=5,
                    help="TTT draft steps to unroll (match deployment spec_tokens)")
    ap.add_argument("--step-weight-beta", type=float, default=0.8,
                    help="decay for per-step loss weights (beta**k, normalized)")
    ap.add_argument("--soft-ce-weight", type=float, default=1.0)
    ap.add_argument("--hard-ce-weight", type=float, default=0.0)
    ap.add_argument("--bf16", action="store_true", help="load models in bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0,
                    help="save a checkpoint every N steps (0 = only at end)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N steps (0 = full epochs); useful for a smoke test")
    return ap.parse_args()


def set_trainable(target, assistant):
    """Apply the freeze policy. Returns the list of trainable parameters.

    Frozen: whole target; assistant.lm_head + assistant.model.embed_tokens.
    Trained: everything else in the assistant (decoder layers + projections).
    """
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()

    # Start by training the whole assistant, then freeze the tied head/embed.
    for p in assistant.parameters():
        p.requires_grad_(True)

    frozen_names = []
    lm_head = getattr(assistant, "lm_head", None)
    if lm_head is not None:
        for p in lm_head.parameters():
            p.requires_grad_(False)
        frozen_names.append("lm_head")
    asst_base = getattr(assistant, "model", None)
    embed = getattr(asst_base, "embed_tokens", None) if asst_base is not None else None
    if embed is not None:
        for p in embed.parameters():
            p.requires_grad_(False)
        frozen_names.append("model.embed_tokens")

    trainable = [p for p in assistant.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in assistant.parameters())
    print(f"[freeze] target: fully frozen", flush=True)
    print(f"[freeze] assistant frozen submodules: {frozen_names}", flush=True)
    print(f"[freeze] trainable params: {n_train:,} / {n_total:,} "
          f"({100.0 * n_train / max(n_total, 1):.1f}%)", flush=True)
    if n_train == 0:
        raise RuntimeError("no trainable params after freeze; check module names")
    return trainable


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    from gemma4_mtp.data import (DataConfig, build_dataset, collate,
                                 CacheDataset, collate_cache)
    from gemma4_mtp.training_step import (MTPLossConfig, training_step,
                                          training_step_from_cache,
                                          locate_target_parts)

    use_cache = args.cache_dir is not None

    # --- Distributed setup (torchrun sets RANK/LOCAL_RANK/WORLD_SIZE) ---
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = args.device
    is_main = rank == 0

    def log(*a, **k):
        if is_main:
            print(*a, **k, flush=True)

    os.makedirs(args.output, exist_ok=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    log(f"=== Loading models (world_size={world_size}, cache={use_cache}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    if use_cache:
        # Cache mode: the 26B target's signals are precomputed. We still need the
        # target's INPUT EMBEDDING for the draft's token embedding (the recipe's
        # left half). Load the full target on CPU, grab embed_tokens onto GPU,
        # then free the rest — a few GB of embedding instead of ~50GB of target.
        target = None
        _tmp = AutoModelForCausalLM.from_pretrained(
            args.target, dtype=dtype, trust_remote_code=True)
        target_embed = _tmp.get_input_embeddings().to(device)
        for p in target_embed.parameters():
            p.requires_grad_(False)
        del _tmp
        torch.cuda.empty_cache()
    else:
        # Each rank loads its OWN full copy onto its own GPU (data parallel).
        target = AutoModelForCausalLM.from_pretrained(
            args.target, dtype=dtype, trust_remote_code=True,
        ).to(device)
        target_embed = None
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=dtype, trust_remote_code=True,
    ).to(device)
    if use_cache:
        # Freeze policy for cache mode: freeze assistant lm_head + embed only.
        for p in assistant.parameters():
            p.requires_grad_(True)
        lm_head = getattr(assistant, "lm_head", None)
        if lm_head is not None:
            for p in lm_head.parameters():
                p.requires_grad_(False)
        asst_base = getattr(assistant, "model", None)
        emb = getattr(asst_base, "embed_tokens", None) if asst_base else None
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad_(False)
        trainable = [p for p in assistant.parameters() if p.requires_grad]
        log(f"[freeze] cache mode: {sum(p.numel() for p in trainable):,} trainable")
    else:
        trainable = set_trainable(target, assistant)

    # Wrap the assistant (the only thing being trained) in DDP.
    if ddp:
        assistant = DDP(assistant, device_ids=[local_rank],
                        find_unused_parameters=True)
    # training_step expects the raw module interface; keep a handle to unwrap.
    assistant_module = assistant.module if ddp else assistant

    log("=== Building dataset ===")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if use_cache:
        dataset = CacheDataset(args.cache_dir)
        collate_fn = lambda b: collate_cache(b, pad_token_id=pad_id)
    else:
        data_cfg = DataConfig(max_length=args.max_length)
        cache_path = args.cache or (args.data + f".tok_ml{args.max_length}.pt")
        dataset = build_dataset(args.data, tokenizer, data_cfg,
                                cache_path=cache_path, rank=rank, world_size=world_size)
        collate_fn = lambda b: collate(b, pad_token_id=pad_id)
    if len(dataset) == 0:
        raise RuntimeError("empty dataset; check --data/--cache-dir")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True) if ddp else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, collate_fn=collate_fn,
    )

    loss_cfg = MTPLossConfig(
        ttt_steps=args.ttt_steps,
        step_weight_beta=args.step_weight_beta,
        temperature=args.temperature,
        soft_ce_weight=args.soft_ce_weight,
        hard_ce_weight=args.hard_ce_weight,
    )
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(loader) // max(args.grad_accum, 1)) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=max(total_steps, 1))

    log(f"=== Training: {len(dataset)} samples, {len(loader)} batches/epoch/rank, "
        f"~{total_steps} optim steps ===")
    assistant.train()
    step = 0
    optim.zero_grad()
    def to_device(batch):
        out = {}
        for k, v in batch.items():
            if k == "shared_kv_states":
                out[k] = {kt: (kv[0].to(device), kv[1].to(device))
                          for kt, kv in v.items()}
            else:
                out[k] = v.to(device)
        return out

    def run_step(batch):
        if use_cache:
            return training_step_from_cache(
                assistant_module, target_embed, batch, loss_cfg)
        return training_step(target, assistant_module, batch, loss_cfg)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            batch = to_device(batch)
            # DDP syncs grads on the backward of the LAST micro-step only.
            is_accum_step = (i + 1) % args.grad_accum != 0
            if ddp and is_accum_step:
                with assistant.no_sync():
                    loss, metrics = run_step(batch)
                    (loss / args.grad_accum).backward()
            else:
                loss, metrics = run_step(batch)
                (loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    lr = sched.get_last_lr()[0]
                    msg = " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items())
                    log(f"epoch {epoch} step {step}/{total_steps} lr={lr:.2e} {msg}")
                if is_main and args.save_every and step % args.save_every == 0:
                    _save(assistant_module, tokenizer,
                          os.path.join(args.output, f"step{step}"))
                if args.max_steps and step >= args.max_steps:
                    log("[stop] reached --max-steps")
                    if is_main:
                        _save(assistant_module, tokenizer, args.output)
                    if ddp:
                        dist.barrier()
                        dist.destroy_process_group()
                    return

    if is_main:
        _save(assistant_module, tokenizer, args.output)
    log("=== Done ===")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def _save(assistant, tokenizer, path):
    """Save the fine-tuned assistant with its stock config for vLLM drop-in."""
    os.makedirs(path, exist_ok=True)
    assistant.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[save] wrote checkpoint to {path}", flush=True)


if __name__ == "__main__":
    main()

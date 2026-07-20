#!/usr/bin/env python3
"""Regenerate assistant answers with a vLLM-served 26B target, producing the
regen JSONL that build_split.py consumes.

WHY vLLM serve (not local vllm.LLM / sglang): keeps the generation path identical
to the benchmark stack (same OpenAI-compatible endpoint, same sampling), so the
answers the draft is trained to predict are drawn from exactly the model+sampling
that vLLM will serve at deployment.

PIPELINE
    DSpark short-layers prompts (train_<layer>.jsonl, each row has "id" +
    "conversations" up to a user turn)
      -> THIS script: for each user turn, ask the served 26B for the assistant
         answer, splice it in; write {conversations, status, id[, source_layer]}
      -> build_split.py: route each regen row into train/eval by id.

The output row schema matches build_split.py's expectation exactly
(build_split.py:15,120-126): {"conversations":[...], "status":"success", "id":...}.

USAGE (on the server, in the vLLM venv)

  1. Start a vLLM server for the 26B target (OpenAI-compatible), e.g.:

       vllm serve google/gemma-4-26B-A4B-it-text-only \
         --served-model-name gemma4 --port 8100 \
         --tensor-parallel-size 2 --max-model-len 24576 \
         --quantization fp8 --trust-remote-code

     (Any serve config works; the tokenizer/template must match training.)

  2. Run this script against it:

       python -m gemma4_mtp.regen \
         --model gemma4 \
         --server http://localhost:8100/v1 \
         --input  /path/to/dspark/short_layers/train_layer1_actual.jsonl \
         --output /path/to/regen/train_layer1_actual_regen_26b.jsonl \
         --concurrency 64 --temperature 0.7 --top-p 0.95 --max-tokens 4096

  To regenerate ALL layers into one file, concatenate the per-layer inputs first
  (they all carry ids), or run per layer and `cat` the outputs; build_split reads
  a single --regen file. Pass --source-layer to tag rows for build_split stats.

Resume: with --resume, already-written rows in --output are skipped (counted by
line), so an interrupted run continues.

Defaults (temperature 0.7 / top_p 0.95 / max_tokens 4096) match the 26B bench
config (configs/26b_e011_mtp.json) and DSpark's generate_train_data.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="served model name (matches vllm serve --served-model-name)")
    ap.add_argument("--server", required=True, nargs="+",
                    help="one or more OpenAI-compatible base URLs, e.g. "
                         "http://localhost:8100/v1 http://localhost:8101/v1 ... "
                         "Requests are round-robined across them (one per GPU).")
    ap.add_argument("--input", required=True,
                    help="DSpark prompt JSONL (rows with id + conversations)")
    ap.add_argument("--output", required=True, help="regen JSONL to write")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="in-flight requests")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--num-samples", type=int, default=None,
                    help="cap number of input rows (debug)")
    ap.add_argument("--source-layer", default=None,
                    help="optional source_layer tag added to each row (build_split "
                         "uses it only for stats)")
    ap.add_argument("--resume", action="store_true",
                    help="skip rows already present in --output (by line count)")
    return ap.parse_args()


def _build_kwargs(args, messages):
    kw = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        kw["top_p"] = args.top_p
    return kw


def _error_row(sample, message):
    sample = dict(sample)
    sample["status"] = "error"
    sample["error"] = message
    return sample


def regen_one(client, args, sample):
    """Regenerate every assistant turn in one conversation via the served model.

    Ported from DSpark generate_train_data.call_sglang: walk the conversation,
    pass through system, re-ask on each user turn, drop the original assistant
    turns and splice in the model's answer. Returns the row with
    conversations replaced + status set (schema build_split.py expects).
    """
    # Input conversation field varies: raw MAI Profile uses "prompt_messages"
    # (system + user, prompt only); DSpark normalized uses "conversations".
    conversations = (sample.get("prompt_messages")
                     or sample.get("conversations")
                     or sample.get("messages"))
    if not conversations:
        return _error_row(sample, "missing conversations")
    if conversations[0].get("role") == "assistant":
        return _error_row(sample, "conversation starts with assistant")

    regenerated = []
    for message in conversations:
        role = message.get("role")
        if role == "system":
            regenerated.append(message)
            continue
        if role == "assistant":
            continue  # drop original; we regenerate below
        if role != "user":
            return _error_row(sample, f"invalid role: {role}")

        regenerated.append(message)
        try:
            resp = client.chat.completions.create(**_build_kwargs(args, regenerated))
        except Exception as exc:  # noqa: BLE001 — record and continue
            return _error_row(sample, str(exc))
        regenerated.append({
            "role": "assistant",
            "content": resp.choices[0].message.content,
        })

    out = dict(sample)
    out.pop("prompt_messages", None)   # replaced by "conversations" below
    out["conversations"] = regenerated
    out["status"] = "success"
    # build_split routes by id = "{layer}:{prompt_hash}" (matches DSpark's split
    # files). Construct it from source_layer + prompt_hash when not already set.
    if "id" not in out and out.get("prompt_hash"):
        if args.source_layer:
            out["id"] = f"{args.source_layer}:{out['prompt_hash']}"
        else:
            out["id"] = out["prompt_hash"]
    if args.source_layer is not None and "source_layer" not in out:
        out["source_layer"] = args.source_layer
    return out


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "openai package required: pip install openai") from exc

    # One client per served endpoint; requests round-robin across them so all
    # GPUs stay busy.
    clients = [OpenAI(base_url=url, api_key="none") for url in args.server]

    total = _count_lines(args.input)
    skip = _count_lines(args.output) if args.resume else 0
    error_path = args.output.replace(".jsonl", "_error.jsonl")
    if skip >= total and total > 0:
        print(f"all {total} rows already processed")
        return

    print(f"=== regen via vLLM serve ===")
    print(f"  model={args.model}  servers={len(clients)}: {args.server}")
    print(f"  input={args.input} ({total} rows)  output={args.output}")
    print(f"  sampling: temp={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens}  concurrency={args.concurrency}/server")
    if skip:
        print(f"  resume: skipping first {skip} rows")

    mode = "a" if (args.resume and skip > 0) else "w"
    n_ok = 0
    n_err = 0
    submitted = 0
    total_inflight = args.concurrency * len(clients)

    # progress bar target = rows we'll actually process this run
    if args.num_samples is not None:
        target = min(args.num_samples, max(total - skip, 0))
    else:
        target = max(total - skip, 0)
    try:
        from tqdm import tqdm
        pbar = tqdm(total=target, desc="regen", unit="row")
    except ImportError:
        pbar = None

    with (
        open(args.input, "r", encoding="utf-8") as fin,
        open(args.output, mode, encoding="utf-8") as fout,
        open(error_path, mode, encoding="utf-8") as ferr,
        ThreadPoolExecutor(max_workers=total_inflight) as pool,
    ):
        for _ in range(skip):
            next(fin, None)

        inflight = []

        def drain(block):
            nonlocal n_ok, n_err
            progressed = False
            for fut in list(inflight):
                if fut.done():
                    row = fut.result()
                    if row.get("status") == "success":
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_ok += 1
                    else:
                        ferr.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_err += 1
                    inflight.remove(fut)
                    progressed = True
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(ok=n_ok, err=n_err, refresh=False)
            if block and not progressed:
                time.sleep(0.05)

        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.num_samples is not None and submitted >= args.num_samples:
                break
            sample = json.loads(line)
            while len(inflight) >= total_inflight:
                drain(block=True)
            client = clients[submitted % len(clients)]   # round-robin GPUs
            inflight.append(pool.submit(regen_one, client, args, sample))
            submitted += 1

        while inflight:
            drain(block=True)

    if pbar is not None:
        pbar.close()
    print("=== done ===")
    print(f"  success={n_ok}  errors={n_err}  (errors -> {error_path})")


if __name__ == "__main__":
    main()

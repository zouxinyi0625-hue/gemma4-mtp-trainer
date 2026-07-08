"""Data pipeline for MTP fine-tuning: MAI Profile conversations -> tensors.

Reads the same target-regenerated JSONL the DSpark pipeline produces
(`DeepSpec/scripts/data/generate_train_data.py`): one JSON object per line with

    {"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}],
     "status": "success"}

and turns each conversation into (input_ids, attention_mask, loss_mask) where
loss_mask == 1 ONLY on assistant-response tokens (so training supervises the
draft only where the target actually generated text).

The Gemma4 chat-template constants and the response-masking approach are taken
verbatim from DeepSpec's verified parser
(`DeepSpec/deepspec/data/parser.py`, ChatTemplate "gemma4" + GeneralParser) so
this trainer masks tokens identically to the DSpark line. Kept self-contained
here so the repo doesn't import the DeepSpec package.

torch is imported lazily so the module can be syntax-checked without it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


# --- Gemma4 chat template (verbatim from DeepSpec parser.py TEMPLATE_REGISTRY) -
GEMMA4_ASSISTANT_HEADER = "<|turn>model\n"
GEMMA4_USER_HEADER = "<|turn>user\n"
GEMMA4_END_OF_TURN = "<turn|>\n"
GEMMA4_ASSISTANT_LOSS_PREFIX = "<|channel>thought\n<channel|>"


@dataclass
class DataConfig:
    max_length: int = 2048
    # Drop samples with no supervised tokens (all-prompt / parse failures).
    drop_empty_loss_mask: bool = True
    # Only keep rows with this status (generate_train_data.py sets "success").
    require_status_success: bool = True


class Gemma4ConversationParser:
    """Render a conversation and build a response-only loss mask.

    Mirrors DeepSpec GeneralParser for the gemma4 template: it locates each
    assistant span via a regex on the rendered text, then maps character spans
    to token indices by re-encoding prefixes (robust to tokenizer merges).
    """

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_loss_prefix = GEMMA4_ASSISTANT_LOSS_PREFIX
        self.assistant_header = GEMMA4_ASSISTANT_HEADER
        # Same pattern shape as DeepSpec: header + lazy content up to end-of-turn.
        self.assistant_pattern = (
            re.escape(self.assistant_header)
            + r"([\s\S]*?(?:"
            + re.escape(GEMMA4_END_OF_TURN)
            + "|$))"
        )

    def _prepare_render_messages(self, messages):
        # DeepSpec prepends the loss prefix to assistant content so the mask can
        # skip the "thought" scaffolding and supervise only the real answer.
        if not self.assistant_loss_prefix:
            return messages
        out = []
        for m in messages:
            if m.get("role") != "assistant":
                out.append(m)
                continue
            content = m["content"]
            if not isinstance(content, str):
                raise ValueError("assistant content must be text for gemma4 MTP")
            rm = dict(m)
            if not content.startswith(self.assistant_loss_prefix):
                rm["content"] = f"{self.assistant_loss_prefix}{content}"
            out.append(rm)
        return out

    def parse(self, conversation):
        import torch

        messages = list(conversation)
        if messages and messages[0]["role"] != "system":
            # gemma4 template registers no system prompt; leave as-is.
            pass
        render_messages = self._prepare_render_messages(messages)

        conversation_text = self.tokenizer.apply_chat_template(
            render_messages, tokenize=False, add_generation_prompt=False,
        )

        enc = self.tokenizer(
            conversation_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc.input_ids[0]
        attention_mask = enc.attention_mask[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        for match in re.finditer(self.assistant_pattern, conversation_text, re.DOTALL):
            content_start_char = match.start(1)
            if self.assistant_loss_prefix and conversation_text.startswith(
                self.assistant_loss_prefix, content_start_char,
            ):
                content_start_char += len(self.assistant_loss_prefix)
            content_end_char = match.end(1)
            prefix_ids = self.tokenizer.encode(
                conversation_text[:content_start_char],
                add_special_tokens=False, truncation=True, max_length=self.max_length,
            )
            full_ids = self.tokenizer.encode(
                conversation_text[:content_end_char],
                add_special_tokens=False, truncation=True, max_length=self.max_length,
            )
            start = min(len(prefix_ids), len(input_ids))
            end = min(len(full_ids), len(input_ids))
            if start < end:
                loss_mask[start:end] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }


def iter_jsonl(path):
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_dataset(jsonl_path, tokenizer, cfg: DataConfig | None = None):
    """Return a torch Dataset of {input_ids, attention_mask, loss_mask}.

    Rows are parsed lazily-then-materialized; large corpora should stream, but
    for a first training run we keep it simple and hold token tensors in memory
    (they're small vs the 26B model). Raw text is not retained.
    """
    import torch
    from torch.utils.data import Dataset

    cfg = cfg or DataConfig()
    parser = Gemma4ConversationParser(tokenizer, max_length=cfg.max_length)

    samples = []
    n_total = n_skipped = 0
    for obj in iter_jsonl(jsonl_path):
        n_total += 1
        if cfg.require_status_success and obj.get("status") not in (None, "success"):
            n_skipped += 1
            continue
        conv = obj.get("conversations")
        if not conv:
            n_skipped += 1
            continue
        try:
            parsed = parser.parse(conv)
        except Exception:
            n_skipped += 1
            continue
        if cfg.drop_empty_loss_mask and int(parsed["loss_mask"].sum()) == 0:
            n_skipped += 1
            continue
        samples.append(parsed)

    print(f"[data] kept {len(samples)}/{n_total} conversations "
          f"({n_skipped} skipped) from {jsonl_path}", flush=True)

    class _DS(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            return samples[i]

    return _DS()


def collate(batch, pad_token_id: int):
    """Right-pad a list of samples into batched tensors."""
    import torch

    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    loss_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
        loss_mask[i, :n] = b["loss_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "loss_mask": loss_mask}

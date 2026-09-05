"""Versioned chat template for FramerAI instruction and tool-calling format."""

import json
from typing import Any

import torch


class ChatTemplate:
    """Versioned Chat Template for SFT, DPO, and inference.

    Formats conversations into structured role segments:
    - system:     <system>...
    - user:       <user>...
    - assistant:  <assistant>...
    - tool_call:  <tool_call>...
    - tool:       <tool>...
    """

    VERSION = "v1"

    def __init__(self, version: str = "v1"):
        if version != "v1":
            raise ValueError(f"Unsupported ChatTemplate version: {version!r}. Supported: 'v1'")
        self.version = version

    def format_messages(self, messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
        """Format a list of message dicts into a single prompt string."""
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            content_str = str(content).strip() if content is not None else ""

            if role == "system":
                formatted_parts.append(f"<system>{content_str}")
            elif role == "user":
                formatted_parts.append(f"<user>{content_str}")
            elif role == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    tc = msg["tool_calls"]
                    if isinstance(tc, (dict, list)):
                        tc_str = json.dumps(tc)
                    elif isinstance(tc, str):
                        try:
                            parsed = json.loads(tc)
                            tc_str = json.dumps(parsed)
                        except Exception:
                            tc_str = tc
                    else:
                        tc_str = str(tc)
                    formatted_parts.append(f"<tool_call>{tc_str}</tool_call>")
                else:
                    formatted_parts.append(f"<assistant>{content_str}")
            elif role == "tool_call":
                if isinstance(content, (dict, list)):
                    c_str = json.dumps(content)
                elif isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        c_str = json.dumps(parsed)
                    except Exception:
                        c_str = content
                else:
                    c_str = str(content)
                formatted_parts.append(f"<tool_call>{c_str}</tool_call>")
            elif role in ("tool", "tool_result"):
                name = msg.get("name", "")
                prefix = f"[{name}] " if name else ""
                formatted_parts.append(f"<tool>{prefix}{content_str}")
            else:
                formatted_parts.append(f"<{role}>{content_str}")

        text = "".join(formatted_parts)
        if add_generation_prompt and not text.endswith("<assistant>"):
            text += "<assistant>"
        return text

    def encode_conversation(
        self,
        messages: list[dict[str, Any]],
        tokenizer,
        max_len: int = 512,
        pad_to_max: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Encode a multi-turn conversation into input_ids and masked labels for SFT.

        Labels are pre-shifted for next-token prediction:
        input_ids = full_sequence[:-1]
        labels = full_sequence[1:] (with non-assistant target positions set to -100)
        """
        full_tokens = [tokenizer.sos_id]
        full_is_target = [False]

        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            content_str = str(content).strip() if content is not None else ""

            if role == "system":
                text = f"<system>{content_str}"
                is_assistant = False
            elif role == "user":
                text = f"<user>{content_str}"
                is_assistant = False
            elif role == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    tc = msg["tool_calls"]
                    if isinstance(tc, (dict, list)):
                        tc_str = json.dumps(tc)
                    elif isinstance(tc, str):
                        try:
                            parsed = json.loads(tc)
                            tc_str = json.dumps(parsed)
                        except Exception:
                            tc_str = tc
                    else:
                        tc_str = str(tc)
                    text = f"<tool_call>{tc_str}</tool_call>"
                else:
                    text = f"<assistant>{content_str}"
                is_assistant = True
            elif role == "tool_call":
                if isinstance(content, (dict, list)):
                    c_str = json.dumps(content)
                elif isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        c_str = json.dumps(parsed)
                    except Exception:
                        c_str = content
                else:
                    c_str = str(content)
                text = f"<tool_call>{c_str}</tool_call>"
                is_assistant = True
            elif role in ("tool", "tool_result"):
                name = msg.get("name", "")
                prefix = f"[{name}] " if name else ""
                text = f"<tool>{prefix}{content_str}"
                is_assistant = False
            else:
                text = f"<{role}>{content_str}"
                is_assistant = False

            turn_ids = tokenizer.encode(text, add_special=False)
            full_tokens.extend(turn_ids)
            full_is_target.extend([is_assistant] * len(turn_ids))

        full_tokens.append(tokenizer.eos_id)
        full_is_target.append(full_is_target[-1] if full_is_target else False)

        input_ids = full_tokens[:-1]
        raw_labels = full_tokens[1:]
        target_mask = full_is_target[1:]

        labels = [tok if target else -100 for tok, target in zip(raw_labels, target_mask, strict=False)]

        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]
            attention_mask = [1] * max_len
        elif pad_to_max and len(input_ids) < max_len:
            seq_len = len(input_ids)
            pad_len = max_len - seq_len
            input_ids.extend([tokenizer.pad_id] * pad_len)
            labels.extend([-100] * pad_len)
            attention_mask = [1] * seq_len + [0] * pad_len
        else:
            attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

"""Versioned chat template for FramerAI instruction and tool-calling format."""

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
            content = str(msg.get("content", "")).strip()

            if role == "system":
                formatted_parts.append(f"<system>{content}")
            elif role == "user":
                formatted_parts.append(f"<user>{content}")
            elif role == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Tool call emission
                    tc = msg["tool_calls"]
                    tc_str = str(tc) if not isinstance(tc, str) else tc
                    formatted_parts.append(f"<tool_call>{tc_str}")
                else:
                    formatted_parts.append(f"<assistant>{content}")
            elif role == "tool_call":
                formatted_parts.append(f"<tool_call>{content}")
            elif role in ("tool", "tool_result"):
                name = msg.get("name", "")
                prefix = f"[{name}] " if name else ""
                formatted_parts.append(f"<tool>{prefix}{content}")
            else:
                formatted_parts.append(f"<{role}>{content}")

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

        Labels are set to -100 for system, user, and tool-result tokens so only
        assistant (and tool_call) response tokens contribute to the SFT loss.
        """
        input_ids = [tokenizer.sos_id]
        labels = [-100]

        for msg in messages:
            role = msg.get("role", "").lower()
            content = str(msg.get("content", "")).strip()

            if role == "system":
                text = f"<system>{content}"
                is_assistant = False
            elif role == "user":
                text = f"<user>{content}"
                is_assistant = False
            elif role == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    tc = msg["tool_calls"]
                    tc_str = str(tc) if not isinstance(tc, str) else tc
                    text = f"<tool_call>{tc_str}"
                else:
                    text = f"<assistant>{content}"
                is_assistant = True
            elif role == "tool_call":
                text = f"<tool_call>{content}"
                is_assistant = True
            elif role in ("tool", "tool_result"):
                name = msg.get("name", "")
                prefix = f"[{name}] " if name else ""
                text = f"<tool>{prefix}{content}"
                is_assistant = False
            else:
                text = f"<{role}>{content}"
                is_assistant = False

            turn_ids = tokenizer.encode(text, add_special=False)
            input_ids.extend(turn_ids)
            if is_assistant:
                labels.extend(turn_ids)
            else:
                labels.extend([-100] * len(turn_ids))

        input_ids.append(tokenizer.eos_id)
        labels.append(tokenizer.eos_id)

        # Truncate
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
        elif pad_to_max and len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids.extend([tokenizer.pad_id] * pad_len)
            labels.extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

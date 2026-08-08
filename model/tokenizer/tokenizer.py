"""BPE Tokenizer for FramerAI."""

import json
import os
import re


class FramerTokenizer:
    """Byte-Pair Encoding tokenizer with special tokens for multimodal tasks.

    Token id layout, which every saved tokenizer and checkpoint depends on::

        [0, num_special)                       special tokens
        [num_special, num_special + 256)       raw bytes
        [num_special + 256, first_merge_id)    reserved marker slots
        [first_merge_id, vocab_size)           learned BPE merges

    New markers go in the reserved block, which sits *after* the byte range and
    has a fixed capacity. Adding one therefore leaves every byte id untouched and
    consumes a slot that was already accounted for, instead of shifting the whole
    vocabulary and silently invalidating tokenizers saved earlier.
    """

    VERSION = 2

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "<img>": 4,
        "<img_end>": 5,
        "<vid>": 6,
        "<vid_end>": 7,
        "<code>": 8,
        "<code_end>": 9,
        "<user>": 10,
        "<assistant>": 11,
        "<system>": 12,
        "<audio>": 13,
        "<audio_end>": 14,
    }

    # Fixed-capacity block after the byte range. Offsets are stable forever;
    # unused slots stay empty so a later addition still shifts nothing.
    RESERVED_SLOTS = 16
    RESERVED_TOKENS = {
        "<img_patch>": 0,
        "<audio_frame>": 1,
    }

    def __init__(
        self,
        vocab_size: int = 50304,
        special_tokens: dict = None,
        reserved_slots: int = None,
        reserved_tokens: dict = None,
    ):
        self.vocab_size = vocab_size
        self.special_tokens = dict(special_tokens or self.SPECIAL_TOKENS)
        self.num_special = len(self.special_tokens)
        self.reserved_slots = self.RESERVED_SLOTS if reserved_slots is None else int(reserved_slots)

        # Initialize byte-level vocabulary
        self.byte_to_token = {}
        self.token_to_byte = {}
        for i in range(256):
            tok_id = self.num_special + i
            self.byte_to_token[i] = tok_id
            self.token_to_byte[tok_id] = bytes([i])

        reserved_base = self.num_special + 256
        offsets = self.RESERVED_TOKENS if reserved_tokens is None else reserved_tokens
        self.reserved_tokens = {
            name: reserved_base + int(offset)
            for name, offset in offsets.items()
            if int(offset) < self.reserved_slots
        }
        self.first_merge_id = reserved_base + self.reserved_slots

        # Every atomic marker the encoder must not split and the decoder must
        # render verbatim: the specials plus whatever the reserved block holds.
        self.marker_tokens = {**self.special_tokens, **self.reserved_tokens}

        self.merges = []
        self.merge_map = {}
        self.vocab = {v: k for k, v in self.marker_tokens.items()}
        for i in range(256):
            self.vocab[self.num_special + i] = bytes([i])

        self.pad_id = self.special_tokens["<pad>"]
        self.sos_id = self.special_tokens["<sos>"]
        self.eos_id = self.special_tokens["<eos>"]
        self.unk_id = self.special_tokens["<unk>"]

    def _get_pairs(self, tokens: list) -> dict:
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def train(self, texts: list, target_vocab_size: int = None):
        """Train BPE merges on a corpus of texts."""
        target = target_vocab_size or self.vocab_size
        num_merges = target - self.first_merge_id

        # Convert all text to byte tokens
        all_tokens = []
        for text in texts:
            text_bytes = text.encode("utf-8")
            tokens = [self.byte_to_token[b] for b in text_bytes]
            all_tokens.append(tokens)

        next_id = self.first_merge_id
        for _ in range(num_merges):
            pair_counts = {}
            for tokens in all_tokens:
                for pair, count in self._get_pairs(tokens).items():
                    pair_counts[pair] = pair_counts.get(pair, 0) + count

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(best_pair)
            self.merge_map[best_pair] = next_id
            self.vocab[next_id] = best_pair

            # Apply merge
            for j in range(len(all_tokens)):
                new_tokens = []
                i = 0
                while i < len(all_tokens[j]):
                    if i < len(all_tokens[j]) - 1 and (all_tokens[j][i], all_tokens[j][i + 1]) == best_pair:
                        new_tokens.append(next_id)
                        i += 2
                    else:
                        new_tokens.append(all_tokens[j][i])
                        i += 1
                all_tokens[j] = new_tokens

            next_id += 1

    def encode(self, text: str, add_special: bool = True) -> list:
        """Encode text to token IDs."""
        # Handle special token markers
        special_pattern = "|".join(re.escape(t) for t in self.marker_tokens)
        parts = re.split(f"({special_pattern})", text)

        tokens = []
        for part in parts:
            if part in self.marker_tokens:
                tokens.append(self.marker_tokens[part])
            elif part:
                # Byte-level encoding
                byte_tokens = [self.byte_to_token[b] for b in part.encode("utf-8")]
                # Apply merges
                for merge_pair in self.merges:
                    new_tokens = []
                    i = 0
                    merged_id = self.merge_map[merge_pair]
                    while i < len(byte_tokens):
                        if i < len(byte_tokens) - 1 and (byte_tokens[i], byte_tokens[i + 1]) == merge_pair:
                            new_tokens.append(merged_id)
                            i += 2
                        else:
                            new_tokens.append(byte_tokens[i])
                            i += 1
                    byte_tokens = new_tokens
                tokens.extend(byte_tokens)

        if add_special:
            tokens = [self.sos_id] + tokens + [self.eos_id]

        return tokens

    def decode(self, token_ids: list) -> str:
        """Decode token IDs back to text."""
        reverse_marker = {v: k for k, v in self.marker_tokens.items()}
        byte_list = []
        for tid in token_ids:
            if tid in reverse_marker:
                if reverse_marker[tid] not in ("<pad>", "<sos>", "<eos>"):
                    byte_list.extend(reverse_marker[tid].encode("utf-8"))
            elif tid in self.token_to_byte:
                byte_list.extend(self.token_to_byte[tid])
            else:
                # Decompose merged token
                byte_list.extend(self._decompose(tid))
        return bytes(byte_list).decode("utf-8", errors="replace")

    def _decompose(self, token_id: int) -> list:
        if token_id in self.token_to_byte:
            return list(self.token_to_byte[token_id])
        if token_id in self.vocab and isinstance(self.vocab[token_id], tuple):
            left, right = self.vocab[token_id]
            return self._decompose(left) + self._decompose(right)
        return []

    def save(self, path: str):
        """Save tokenizer to disk."""
        os.makedirs(path, exist_ok=True)
        reserved_base = self.num_special + 256
        data = {
            "version": self.VERSION,
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.special_tokens,
            "reserved_slots": self.reserved_slots,
            "reserved_tokens": {
                name: tid - reserved_base for name, tid in self.reserved_tokens.items()
            },
        }
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FramerTokenizer":
        """Load tokenizer from disk.

        Restores the merge vocabulary as well as the merge map. Without the
        ``vocab`` entries ``_decompose`` cannot expand a merged id back into its
        bytes, and ``decode`` drops every merged token silently.

        Version 1 files predate the reserved block, so they load with zero
        reserved slots and keep their original merge ids.
        """
        with open(os.path.join(path, "tokenizer.json")) as f:
            data = json.load(f)

        version = int(data.get("version", 1))
        specials = data.get("special_tokens")
        if specials:
            specials = {name: int(tid) for name, tid in specials.items()}
        reserved_slots = data.get("reserved_slots", cls.RESERVED_SLOTS if version >= 2 else 0)
        reserved_tokens = data.get("reserved_tokens", None if version >= 2 else {})

        tokenizer = cls(
            data["vocab_size"],
            special_tokens=specials,
            reserved_slots=reserved_slots,
            reserved_tokens=reserved_tokens,
        )

        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.merge_map = {}
        for i, pair in enumerate(tokenizer.merges):
            merged_id = tokenizer.first_merge_id + i
            tokenizer.merge_map[pair] = merged_id
            tokenizer.vocab[merged_id] = pair
        return tokenizer

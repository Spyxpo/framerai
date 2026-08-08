"""BPE Tokenizer for FramerAI."""

import json
import os
import re


class FramerTokenizer:
    """Byte-Pair Encoding tokenizer with special tokens for multimodal tasks."""

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

    def __init__(self, vocab_size: int = 50304):
        self.vocab_size = vocab_size
        self.special_tokens = dict(self.SPECIAL_TOKENS)
        self.num_special = len(self.special_tokens)

        # Initialize byte-level vocabulary
        self.byte_to_token = {}
        self.token_to_byte = {}
        for i in range(256):
            tok_id = self.num_special + i
            self.byte_to_token[i] = tok_id
            self.token_to_byte[tok_id] = bytes([i])

        self.merges = []
        self.merge_map = {}
        self.vocab = {**{v: k for k, v in self.special_tokens.items()}}
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
        num_merges = target - self.num_special - 256

        # Convert all text to byte tokens
        all_tokens = []
        for text in texts:
            text_bytes = text.encode("utf-8")
            tokens = [self.byte_to_token[b] for b in text_bytes]
            all_tokens.append(tokens)

        next_id = self.num_special + 256
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
        special_pattern = "|".join(re.escape(t) for t in self.special_tokens.keys())
        parts = re.split(f"({special_pattern})", text)

        tokens = []
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
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
        reverse_special = {v: k for k, v in self.special_tokens.items()}
        byte_list = []
        for tid in token_ids:
            if tid in reverse_special:
                if reverse_special[tid] not in ("<pad>", "<sos>", "<eos>"):
                    byte_list.extend(reverse_special[tid].encode("utf-8"))
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
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "FramerTokenizer":
        """Load tokenizer from disk."""
        with open(os.path.join(path, "tokenizer.json")) as f:
            data = json.load(f)
        tokenizer = cls(data["vocab_size"])
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.merge_map = {tuple(m): i + tokenizer.num_special + 256 for i, m in enumerate(tokenizer.merges)}
        return tokenizer

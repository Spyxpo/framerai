"""Block-paged KV cache, with optional 8-bit storage.

The cache was a pair of tensors concatenated on every decoded token, so each
step reallocated the whole history and copied it forward: memory traffic
quadratic in the sequence for a cache that only ever grows by one. It was also
stored densely in the activation dtype at whatever length was reached, and on
the trillion-scale presets that is the number that decides whether the declared
window is servable at all. Sixty-four layers of eight key-value heads at a
million tokens runs to hundreds of gigabytes in bf16, which is more than the
weights.

Two changes, both about storage rather than about what attention computes:

- **Paging.** One buffer grown in fixed blocks. Appending writes into the
  buffer, and reading is a slice of it rather than a fresh concatenation, so a
  decode step copies nothing and the amortised cost over a sequence is linear.
- **Quantisation.** Optionally hold keys and values as int8 with a scale per
  token and head, and dequantise on read. Against bf16 that halves storage once
  the scales are counted, and the compute dtype is unchanged, so this trades a
  little precision in what is remembered for a window that fits.

Whether that precision costs accuracy is a question for the long-context
retrieval suite, not an assumption to make here.
"""

import torch

DEFAULT_BLOCK_SIZE = 256
CACHE_DTYPES = ("auto", "int8")


class KVCache:
    """A growable key-value cache for one attention layer.

    Duck-types as the ``(keys, values)`` tuple the attention path used before,
    so ``cache[0]`` and ``cache[1]`` still read as the key and value tensors and
    an unpacking caller keeps working.
    """

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE, quantized: bool = False):
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = int(block_size)
        self.quantized = bool(quantized)
        self.length = 0
        self._keys = None
        self._values = None
        self._key_scales = None
        self._value_scales = None
        self._dtype = None

    # -- storage ----------------------------------------------------------

    def _blocks_for(self, length: int) -> int:
        return ((length + self.block_size - 1) // self.block_size) * self.block_size

    def _grow(self, tensor, scales, needed: int, like: torch.Tensor):
        """Return a buffer with room for ``needed`` positions, copying what is held."""
        capacity = tensor.shape[2] if tensor is not None else 0
        if capacity >= needed:
            return tensor, scales

        target = self._blocks_for(needed)
        batch, heads, _, dim = like.shape
        store_dtype = torch.int8 if self.quantized else like.dtype
        grown = torch.zeros(
            (batch, heads, target, dim), dtype=store_dtype, device=like.device
        )
        grown_scales = None
        if self.quantized:
            grown_scales = torch.ones(
                (batch, heads, target, 1), dtype=like.dtype, device=like.device
            )

        if tensor is not None and self.length:
            grown[:, :, : self.length] = tensor[:, :, : self.length]
            if self.quantized:
                grown_scales[:, :, : self.length] = scales[:, :, : self.length]
        return grown, grown_scales

    @staticmethod
    def _quantize(tensor):
        """Per token and head absmax scaling into the int8 range."""
        scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        return torch.round(tensor / scale).clamp(-127, 127).to(torch.int8), scale

    # -- interface --------------------------------------------------------

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """Add new positions and return the whole cache as (keys, values)."""
        if keys.shape != values.shape:
            raise ValueError(
                f"keys {tuple(keys.shape)} and values {tuple(values.shape)} must match"
            )

        new = keys.shape[2]
        needed = self.length + new
        self._dtype = keys.dtype

        self._keys, self._key_scales = self._grow(self._keys, self._key_scales, needed, keys)
        self._values, self._value_scales = self._grow(
            self._values, self._value_scales, needed, values
        )

        window = slice(self.length, needed)
        if self.quantized:
            k_q, k_scale = self._quantize(keys)
            v_q, v_scale = self._quantize(values)
            self._keys[:, :, window] = k_q
            self._values[:, :, window] = v_q
            self._key_scales[:, :, window] = k_scale
            self._value_scales[:, :, window] = v_scale
        else:
            self._keys[:, :, window] = keys
            self._values[:, :, window] = values

        self.length = needed
        return self.keys, self.values

    def _read(self, store, scales):
        if store is None:
            return None
        held = store[:, :, : self.length]
        if not self.quantized:
            return held
        return held.to(self._dtype) * scales[:, :, : self.length]

    @property
    def keys(self):
        return self._read(self._keys, self._key_scales)

    @property
    def values(self):
        return self._read(self._values, self._value_scales)

    @property
    def allocated(self) -> int:
        """Positions the buffer holds room for, which is what memory is spent on."""
        return 0 if self._keys is None else self._keys.shape[2]

    def bytes_used(self) -> int:
        """Bytes the cache currently occupies, scales included."""
        if self._keys is None:
            return 0
        total = self._keys.numel() * self._keys.element_size() * 2
        if self.quantized:
            total += self._key_scales.numel() * self._key_scales.element_size() * 2
        return total

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        """Tuple compatibility: 0 is the keys, 1 is the values."""
        if index == 0:
            return self.keys
        if index == 1:
            return self.values
        raise IndexError(f"a KV cache has two entries, not index {index}")

    def __iter__(self):
        yield self.keys
        yield self.values


def cache_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int,
                          quantized: bool = False) -> float:
    """Bytes one token of context costs across every layer.

    The scales are counted, because a quantised cache that ignores them
    under-reports what it actually allocates.
    """
    per_layer = 2 * n_kv_heads * head_dim  # a key and a value
    if not quantized:
        return 2.0 * per_layer * n_layers  # bf16
    scales = 2 * 2 * n_kv_heads  # one bf16 scale per token, head, and tensor
    return float(per_layer + scales) * n_layers

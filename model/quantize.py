"""Weight-only int8 quantisation for serving.

Decoding one token reads every weight once and does little arithmetic with
each, so the time per token is set by how many bytes of weights there are.
Holding the transformer's linear weights as int8 with one scale per output
channel quarters that against fp32 and halves it against bf16, which is also
what decides how large a preset fits on one device.

Only the transformer stack is converted. The embedding stays as it is, and so
does the output head, which is usually tied to it; the head's precision is also
where a small error shows up directly in which token is picked.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_DTYPES = ("auto", "int8")


class Int8Linear(nn.Module):
    """A linear layer holding int8 weights and a per-output-channel scale.

    The scale is applied to the output rather than the weight, so the weight is
    only cast, never rescaled, on the way into the matmul.
    """

    def __init__(self, weight_q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.in_features = weight_q.shape[1]
        self.out_features = weight_q.shape[0]
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        weight_q = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return cls(weight_q, scale.to(linear.weight.dtype), linear.bias)

    # Whether the fused int8 kernel ran on this layer's device. None until the
    # first call finds out.
    _packed_ok = None
    # The fused kernel is a matrix-vector kernel at heart: it wins when few
    # rows are multiplied, which is decode, and loses badly on a prefill chunk,
    # where casting the weight once is amortised over thousands of rows.
    PACKED_MAX_ROWS = 32

    def _packed(self, x: torch.Tensor):
        """The fused int8-weight matmul, which reads the int8 weight directly."""
        flat = x.reshape(-1, self.in_features)
        out = torch._weight_int8pack_mm(flat, self.weight_q, self.scale.to(x.dtype))
        return out.reshape(*x.shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        rows = x.numel() // self.in_features
        packable = rows <= self.PACKED_MAX_ROWS and hasattr(torch, "_weight_int8pack_mm")
        if packable and self._packed_ok is not False:
            try:
                out = self._packed(x)
                self._packed_ok = True
            except (RuntimeError, NotImplementedError):
                # No kernel for this device or dtype; cast instead, which keeps
                # the memory saving and gives up the speed.
                self._packed_ok = False
        if out is None:
            out = F.linear(x, self.weight_q.to(x.dtype)) * self.scale.to(x.dtype)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8"


def quantize_weights_int8(module: nn.Module) -> int:
    """Replace every nn.Linear under ``module`` in place. Returns how many."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear.from_linear(child))
            replaced += 1
        else:
            replaced += quantize_weights_int8(child)
    return replaced


def quantize_model(model: nn.Module, weight_dtype: str = "auto") -> int:
    """Apply ``weight_dtype`` to a FramerModel's transformer layers."""
    if weight_dtype not in WEIGHT_DTYPES:
        raise ValueError(f"weight_dtype must be one of {WEIGHT_DTYPES}, got {weight_dtype!r}")
    if weight_dtype == "auto":
        return 0
    return quantize_weights_int8(model.layers)


def weight_bytes(model: nn.Module) -> int:
    """Bytes held in parameters and buffers, which is what serving spends memory on."""
    tensors = list(model.parameters()) + list(model.buffers())
    seen, total = set(), 0
    for t in tensors:
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total

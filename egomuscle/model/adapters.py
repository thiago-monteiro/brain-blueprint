from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def quantize_to_levels(weight: torch.Tensor, levels: int | None) -> torch.Tensor:
    if levels is None or int(levels) <= 1:
        return weight
    levels = int(levels)
    max_abs = weight.detach().abs().amax().clamp_min(1e-8)
    normalized = (weight / max_abs).clamp(-1.0, 1.0)
    scaled = (normalized + 1.0) * 0.5 * (levels - 1)
    rounded = torch.round(scaled) / (levels - 1) * 2.0 - 1.0
    return rounded * max_abs


class AdapterLinear(nn.Module):
    """Linear layer with optional low-rank adapter and straight-through quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rank: int = 0,
        quantization_levels: int | None = None,
        quantization_mode: str = "none",
    ) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.quantization_levels = quantization_levels
        self.quantization_mode = quantization_mode
        if self.rank > 0:
            self.adapter_a = nn.Parameter(torch.empty(in_features, self.rank))
            self.adapter_b = nn.Parameter(torch.zeros(out_features, self.rank))
            nn.init.kaiming_uniform_(self.adapter_a, a=5**0.5)
        else:
            self.register_parameter("adapter_a", None)
            self.register_parameter("adapter_b", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        quantization_levels: int | None,
        quantization_mode: str,
    ) -> "AdapterLinear":
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            quantization_levels=quantization_levels,
            quantization_mode=quantization_mode,
        )
        wrapped.base.load_state_dict(linear.state_dict())
        return wrapped

    def adapter_weight(self) -> torch.Tensor | None:
        if self.rank <= 0 or self.adapter_a is None or self.adapter_b is None:
            return None
        weight = self.adapter_b @ self.adapter_a.T
        if self.quantization_mode in {"qat", "ste"} and self.quantization_levels is not None:
            quantized = quantize_to_levels(weight, self.quantization_levels)
            weight = weight + (quantized - weight).detach()
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        weight = self.adapter_weight()
        if weight is None:
            return out
        return out + F.linear(x, weight)

    def plastic_penalty(self) -> torch.Tensor:
        weight = self.adapter_weight()
        if weight is None:
            return self.base.weight.new_tensor(0.0)
        return weight.pow(2).mean()

    def capacity_penalty(self) -> torch.Tensor:
        weight = self.adapter_weight()
        if weight is None or self.quantization_levels is None:
            return self.base.weight.new_tensor(0.0)
        quantized = quantize_to_levels(weight.detach(), self.quantization_levels)
        return (weight - quantized).pow(2).mean()

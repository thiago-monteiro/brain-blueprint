from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PBitOutput:
    out: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    sample: torch.Tensor
    entropy_bits: torch.Tensor


class PBitBottleneck(nn.Module):
    """Stochastic binary bottleneck with straight-through sampling options."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_bits: int = 256,
        temperature: float = 0.67,
        mode: str = "stochastic_straight_through",
    ) -> None:
        super().__init__()
        self.num_bits = int(num_bits)
        self.temperature = float(temperature)
        self.mode = mode
        self.to_logits = nn.Linear(input_dim, self.num_bits)
        self.to_output = nn.Linear(self.num_bits, output_dim)

    def forward(self, x: torch.Tensor) -> PBitOutput:
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.nan_to_num(self.to_logits(safe_x), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        probabilities = torch.sigmoid(logits / max(self.temperature, 1e-6))
        probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)

        if self.mode == "off":
            sample = probabilities
        elif self.mode == "deterministic_sigmoid":
            sample = probabilities
        elif self.mode == "stochastic_straight_through":
            hard = torch.bernoulli(probabilities)
            sample = hard + probabilities - probabilities.detach()
        elif self.mode == "gumbel_concrete":
            eps = torch.finfo(probabilities.dtype).eps
            uniform = torch.rand_like(probabilities).clamp(eps, 1.0 - eps)
            logistic = torch.log(uniform) - torch.log1p(-uniform)
            concrete_logits = torch.nan_to_num(logits + logistic, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            sample = torch.sigmoid(concrete_logits / max(self.temperature, 1e-6))
        else:
            raise ValueError(f"Unsupported p-bit mode: {self.mode}")

        entropy_nats = -(
            probabilities * torch.log(probabilities.clamp(1e-6, 1.0 - 1e-6))
            + (1.0 - probabilities) * torch.log((1.0 - probabilities).clamp(1e-6, 1.0 - 1e-6))
        )
        entropy_bits = entropy_nats / torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        return PBitOutput(
            out=self.to_output(sample),
            logits=logits,
            probabilities=probabilities,
            sample=sample,
            entropy_bits=entropy_bits,
        )

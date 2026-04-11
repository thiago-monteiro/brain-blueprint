from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def predictive_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    if mode == "mse":
        return F.mse_loss(pred, target)
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(pred, target)
    raise ValueError(f"Unsupported predictive loss mode: {mode}")


def temporal_consistency_loss(fused: torch.Tensor) -> torch.Tensor:
    if fused.shape[1] <= 1:
        return fused.new_tensor(0.0)
    diff = fused[:, 1:] - fused[:, :-1]
    return diff.pow(2).mean()


def variance_loss(z: torch.Tensor) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    denom = max(flat.shape[0] - 1, 1)
    cov = (flat.T @ flat) / denom
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / z.shape[-1]


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fused: torch.Tensor,
    loss_mode: str = "mse",
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active_weights = {"pred": 1.0, "temp": 0.10, "var": 0.04, "cov": 0.04}
    if weights is not None:
        active_weights.update({name: value for name, value in weights.items() if name in active_weights})

    losses = {
        "pred": predictive_loss(pred, target, mode=loss_mode),
        "temp": temporal_consistency_loss(fused),
        "var": variance_loss(fused),
        "cov": covariance_loss(fused),
    }
    total = sum(active_weights[name] * value for name, value in losses.items())
    return total, losses

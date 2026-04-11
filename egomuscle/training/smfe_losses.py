from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from egomuscle.model.egomuscle import EgoMuscleOutput
from egomuscle.training.losses import covariance_loss, temporal_consistency_loss, variance_loss


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = mu.float()
    log_var = log_var.float().clamp(-8.0, 4.0)
    target = target.float()
    return (0.5 * torch.exp(-log_var) * (target - mu).pow(2) + 0.5 * log_var).mean()


def uncertainty_calibration_penalty(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 10,
) -> torch.Tensor:
    mu = mu.float()
    log_var = log_var.float()
    target = target.float()
    predicted_var = torch.exp(log_var.detach().clamp(-8.0, 4.0)).reshape(-1)
    actual_error = (target.detach() - mu.detach()).pow(2).reshape(-1)
    if predicted_var.numel() < num_bins:
        return mu.new_tensor(0.0)
    order = torch.argsort(predicted_var)
    predicted_var = predicted_var[order]
    actual_error = actual_error[order]
    penalties = []
    for chunk_var, chunk_err in zip(torch.chunk(predicted_var, num_bins), torch.chunk(actual_error, num_bins), strict=False):
        if chunk_var.numel() == 0:
            continue
        penalties.append((chunk_var.mean() - chunk_err.mean()).abs())
    if not penalties:
        return mu.new_tensor(0.0)
    return torch.stack(penalties).mean().to(mu.device)


def uncertainty_error_correlation(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = mu.float()
    log_var = log_var.float()
    target = target.float()
    predicted_var = torch.exp(log_var.detach().clamp(-8.0, 4.0)).reshape(-1)
    actual_error = (target.detach() - mu.detach()).pow(2).reshape(-1)
    if predicted_var.numel() < 2:
        return mu.new_tensor(0.0)
    predicted_var = predicted_var - predicted_var.mean()
    actual_error = actual_error - actual_error.mean()
    denom = predicted_var.norm() * actual_error.norm()
    if float(denom.detach().cpu()) == 0.0:
        return mu.new_tensor(0.0)
    return (predicted_var @ actual_error / denom).to(mu.device)


def video_latent_cosine_loss(pred: torch.Tensor | None, target: torch.Tensor | None) -> torch.Tensor:
    if pred is None or target is None:
        if pred is not None:
            return pred.new_tensor(0.0)
        if target is not None:
            return target.new_tensor(0.0)
        return torch.tensor(0.0)
    return (1.0 - F.cosine_similarity(pred.float(), target.detach().float(), dim=-1)).mean()


def _zero_like_output(output: EgoMuscleOutput) -> torch.Tensor:
    return output.fused.new_tensor(0.0)


def smfe_total_loss(
    output: EgoMuscleOutput,
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if output.target is None:
        raise ValueError("SMFE loss requires muscle targets.")
    if output.pred_mu is None:
        raise ValueError("SMFE loss requires Gaussian prediction outputs; set model.predictive_distribution=gaussian.")

    log_var = output.pred_log_var
    if log_var is None:
        log_var = torch.zeros_like(output.pred_mu)

    active_weights = {
        "muscle_nll": 1.0,
        "video_latent": 0.10,
        "fast_kl": 0.01,
        "precision": 0.05,
        "temp": 0.10,
        "var": 0.04,
        "cov": 0.04,
        "entropy": 0.01,
        "homeostasis": 0.01,
        "capacity": 0.001,
        "plastic": 0.001,
    }
    if weights is not None:
        active_weights.update({name: value for name, value in weights.items() if name in active_weights})

    zero = _zero_like_output(output)
    losses = {
        "muscle_nll": gaussian_nll(output.pred_mu, log_var, output.target),
        "mse": F.mse_loss(output.pred_mu.float(), output.target.float()),
        "video_latent": video_latent_cosine_loss(output.pred_video_latent, output.video_latent_target)
        if output.pred_video_latent is not None and output.video_latent_target is not None
        else zero,
        "fast_kl": output.fast_kl if output.fast_kl is not None else zero,
        "precision": uncertainty_calibration_penalty(output.pred_mu, log_var, output.target),
        "temp": temporal_consistency_loss(output.fused.float()),
        "var": variance_loss(output.fused.float()),
        "cov": covariance_loss(output.fused.float()),
        "entropy": output.pbit_entropy_loss.float() if output.pbit_entropy_loss is not None else zero,
        "homeostasis": output.homeostasis_loss.float() if output.homeostasis_loss is not None else zero,
        "capacity": output.capacity_loss.float() if output.capacity_loss is not None else zero,
        "plastic": output.plastic_loss.float() if output.plastic_loss is not None else zero,
        "uncertainty_error_corr": uncertainty_error_correlation(output.pred_mu, log_var, output.target),
    }
    total = sum(active_weights[name] * losses[name] for name in active_weights)
    return total, losses

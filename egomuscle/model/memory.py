from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class FastStateOutput:
    state: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    prior_mu: torch.Tensor
    prior_log_var: torch.Tensor
    kl: torch.Tensor


class FastStateEncoder(nn.Module):
    """Variational recurrent state for within-clip sensorimotor memory."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 128,
        dropout: float = 0.1,
        learned_decay: bool = True,
        decay_init: float = 0.90,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.gru = nn.GRU(input_dim, state_dim, batch_first=True)
        self.norm = nn.LayerNorm(state_dim)
        self.drop = nn.Dropout(dropout)
        self.q_mu = nn.Linear(state_dim, state_dim)
        self.q_log_var = nn.Linear(state_dim, state_dim)
        self.prior = nn.Linear(state_dim, state_dim * 2)
        decay_init = float(min(max(decay_init, 1e-4), 1.0 - 1e-4))
        decay_logit = torch.logit(torch.tensor(decay_init, dtype=torch.float32))
        self.decay_logit = nn.Parameter(decay_logit, requires_grad=learned_decay)

    def forward(self, fused: torch.Tensor) -> FastStateOutput:
        recurrent, _ = self.gru(fused)
        recurrent = self.drop(self.norm(recurrent))
        q_mu = self.q_mu(recurrent)
        q_log_var = self.q_log_var(recurrent).clamp(-8.0, 4.0)

        shifted = F.pad(q_mu[:, :-1], (0, 0, 1, 0))
        decay = torch.sigmoid(self.decay_logit)
        prior_base = decay * shifted
        prior_mu, prior_log_var = self.prior(prior_base).chunk(2, dim=-1)
        prior_log_var = prior_log_var.clamp(-8.0, 4.0)

        if self.training:
            eps = torch.randn_like(q_mu)
            state = q_mu + eps * torch.exp(0.5 * q_log_var)
        else:
            state = q_mu

        kl = 0.5 * (
            prior_log_var
            - q_log_var
            + (torch.exp(q_log_var) + (q_mu - prior_mu).pow(2)) / torch.exp(prior_log_var)
            - 1.0
        )
        return FastStateOutput(
            state=state,
            mu=q_mu,
            log_var=q_log_var,
            prior_mu=prior_mu,
            prior_log_var=prior_log_var,
            kl=kl.mean(),
        )

"""Synthetic data-generating mechanisms for the CSI experiments.

The main synthetic target is a condition-augmented pairwise drift

    b_i^*(x, y) = 1/(N-1) sum_{j != i} phi((y_i, e(x))^T A (y_j, e(x))) u.

This matches the structural assumption used in the paper and lets us evaluate
fixed-time drift estimation without needing a real dataset or an autoencoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class SyntheticConfig:
    d: int = 8
    N: int = 8
    q: int = 4
    rank: int = 2
    num_conditions: int = 4
    noise_std: float = 0.01
    true_scale: float = 1.0
    drift_scale: float = 1.0
    seed: int = 123


class TruePairwiseDrift:
    """Ground-truth condition-augmented low-rank pairwise drift."""

    def __init__(self, cfg: SyntheticConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        g = torch.Generator(device=device)
        g.manual_seed(cfg.seed)

        D_aug = cfg.d + cfg.q
        U = torch.randn(D_aug, cfg.rank, generator=g, device=device) / (D_aug ** 0.5)
        V = torch.randn(D_aug, cfg.rank, generator=g, device=device) / (D_aug ** 0.5)
        self.A = cfg.true_scale * (U @ V.t())
        self.u = torch.randn(cfg.d, generator=g, device=device)
        self.u = self.u / (self.u.norm() + 1e-12)

        if cfg.q > 0:
            emb = torch.randn(cfg.num_conditions, cfg.q, generator=g, device=device)
            emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
            self.condition_embedding = emb
        else:
            self.condition_embedding = torch.empty(cfg.num_conditions, 0, device=device)

    @staticmethod
    def phi(score: torch.Tensor) -> torch.Tensor:
        """Smooth scalar activation used in the true model."""
        return torch.tanh(score) + 0.25 * torch.sin(2.0 * score)

    def augmented_tokens(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return z_i=(y_i,e(x)) with shape [batch, N, d+q]."""
        if self.cfg.q == 0:
            return y
        e = self.condition_embedding[x]  # [batch, q]
        e_rep = e[:, None, :].expand(-1, self.cfg.N, -1)
        return torch.cat([y, e_rep], dim=-1)

    @torch.no_grad()
    def drift(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate B^*(X,Y). Shape: [batch, N, d]."""
        z = self.augmented_tokens(x, y)
        Az = torch.matmul(z, self.A.t())
        scores = torch.matmul(Az, z.transpose(1, 2))  # [B,N,N]
        phi_scores = self.phi(scores)

        # Remove self-interaction j=i.
        eye = torch.eye(self.cfg.N, device=y.device, dtype=torch.bool)[None, :, :]
        phi_scores = phi_scores.masked_fill(eye, 0.0)
        avg_phi = phi_scores.sum(dim=2) / max(self.cfg.N - 1, 1)  # [B,N]
        return self.cfg.drift_scale * avg_phi[:, :, None] * self.u[None, None, :]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "A": self.A.detach().cpu(),
            "u": self.u.detach().cpu(),
            "condition_embedding": self.condition_embedding.detach().cpu(),
        }


def make_fixed_time_dataset(
    true_drift: TruePairwiseDrift,
    num_samples: int,
    noise_std: float,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (X, Y_t, noisy response V, noiseless B^*) samples."""
    cfg = true_drift.cfg
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randint(0, cfg.num_conditions, (num_samples,), generator=g, device=device)
    y = torch.randn(num_samples, cfg.N, cfg.d, generator=g, device=device)
    target_clean = true_drift.drift(x, y)
    noise = noise_std * torch.randn(target_clean.shape, generator=g, device=device)
    target_noisy = target_clean + noise
    return x, y, target_noisy, target_clean


def train_val_split(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    val_fraction: float = 0.1,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    n = x.shape[0]
    n_val = max(1, int(n * val_fraction))
    perm = torch.randperm(n, device=x.device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return (x[train_idx], y[train_idx], v[train_idx]), (x[val_idx], y[val_idx], v[val_idx])

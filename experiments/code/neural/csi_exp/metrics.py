"""Evaluation metrics used by the experiments."""

from __future__ import annotations

import torch


def normalized_token_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute (1/N) E ||pred - target||_F^2.

    pred and target have shape [batch, N, d]. The returned value is a scalar.
    This is the same normalization used in the theoretical statements.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    n_tokens = pred.shape[1]
    return ((pred - target) ** 2).sum(dim=(1, 2)).mean() / n_tokens


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth: float | None = None) -> torch.Tensor:
    """Squared MMD with an RBF kernel.

    x, y are [n, dim]. This is a practical replacement for high-dimensional W2.
    """
    if x.dim() > 2:
        x = x.flatten(start_dim=1)
    if y.dim() > 2:
        y = y.flatten(start_dim=1)

    with torch.no_grad():
        if bandwidth is None:
            # Median heuristic on a subset for stability.
            z = torch.cat([x[:512], y[:512]], dim=0)
            dist2 = torch.cdist(z, z, p=2) ** 2
            med = torch.median(dist2[dist2 > 0])
            bandwidth = float(torch.sqrt(0.5 * med + 1e-12).cpu())
            if not torch.isfinite(torch.tensor(bandwidth)) or bandwidth <= 0:
                bandwidth = 1.0

    gamma = 1.0 / (2.0 * bandwidth ** 2)
    kxx = torch.exp(-gamma * torch.cdist(x, x, p=2) ** 2).mean()
    kyy = torch.exp(-gamma * torch.cdist(y, y, p=2) ** 2).mean()
    kxy = torch.exp(-gamma * torch.cdist(x, y, p=2) ** 2).mean()
    return kxx + kyy - 2.0 * kxy


def sliced_wasserstein2(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    """Approximate squared sliced-Wasserstein-2 distance.

    x, y are samples with the same number of points. They can have shape [n, N, d]
    or [n, dim]. The function flattens all non-batch dimensions.
    """
    if x.dim() > 2:
        x = x.flatten(start_dim=1)
    if y.dim() > 2:
        y = y.flatten(start_dim=1)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    dim = x.shape[1]

    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    directions = torch.randn(num_projections, dim, device=x.device, generator=gen)
    directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)

    proj_x = x @ directions.t()
    proj_y = y @ directions.t()
    proj_x, _ = torch.sort(proj_x, dim=0)
    proj_y, _ = torch.sort(proj_y, dim=0)
    return ((proj_x - proj_y) ** 2).mean()

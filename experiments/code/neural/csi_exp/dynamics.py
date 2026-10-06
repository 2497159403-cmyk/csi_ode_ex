"""ODE simulation utility, extracted unchanged from the archived module."""

from __future__ import annotations

from typing import Callable

import torch


def ode_euler(
    drift_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    z0: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Euler integration for dZ_t = drift_fn(x,Z_t,t) dt on [0,1]."""
    z = z0.clone()
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((z.shape[0],), k * dt, device=z.device, dtype=z.dtype)
        z = z + dt * drift_fn(x, z, t)
    return z

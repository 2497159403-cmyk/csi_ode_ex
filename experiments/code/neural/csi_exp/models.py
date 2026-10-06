"""Neural models used in the experiments.

The main model is a learnable version of the paper's pairwise interaction class:

    g(z_i,z_j) = phi_theta(z_i^T A z_j) u_theta,

where A is parameterized as a low-rank product U V^T. The condition embedding
is concatenated to each token before pairwise interactions are computed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScalarActivationMLP(nn.Module):
    """Small MLP approximating the scalar activation phi."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        orig_shape = score.shape
        out = self.net(score.reshape(-1, 1))
        return out.reshape(orig_shape)


class LowRankPairwiseDrift(nn.Module):
    """Condition-augmented low-rank pairwise drift estimator.

    Parameters
    ----------
    d, N, q:
        Per-token dimension, number of tokens, and condition embedding dimension.
    rank:
        Rank used in A = U V^T.
    num_conditions:
        Number of discrete condition values in the synthetic experiments.
    use_condition:
        If False, the model ignores e(X). This implements the no-condition ablation.
    learn_output_vector:
        If True, learn the vector u in R^d. If False, use an MLP output head instead.
    """

    def __init__(
        self,
        d: int,
        N: int,
        q: int,
        rank: int,
        num_conditions: int,
        hidden: int = 64,
        use_condition: bool = True,
        learn_output_vector: bool = True,
        fixed_condition_embedding: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d = d
        self.N = N
        self.q = q if use_condition else 0
        self.rank = rank
        self.use_condition = use_condition
        self.D_aug = d + self.q

        if self.use_condition and self.q > 0:
            if fixed_condition_embedding is not None:
                # In the theory, the condition embedding e(X) is regarded as fixed.
                # For the synthetic rate experiment, using the same fixed embedding
                # removes an unnecessary nuisance-estimation problem.
                emb = fixed_condition_embedding.detach().float().clone()
                if emb.shape != (num_conditions, self.q):
                    raise ValueError(
                        f"fixed_condition_embedding has shape {tuple(emb.shape)}, "
                        f"expected {(num_conditions, self.q)}"
                    )
                self.embedding = nn.Embedding.from_pretrained(emb, freeze=True)
            else:
                self.embedding = nn.Embedding(num_conditions, self.q)
        else:
            self.embedding = None

        self.U = nn.Parameter(torch.randn(self.D_aug, rank) / (self.D_aug ** 0.5))
        self.V = nn.Parameter(torch.randn(self.D_aug, rank) / (self.D_aug ** 0.5))
        self.phi = ScalarActivationMLP(hidden=hidden)
        self.learn_output_vector = learn_output_vector
        if learn_output_vector:
            self.u = nn.Parameter(torch.randn(d) / (d ** 0.5))
        else:
            self.out_head = nn.Sequential(
                nn.Linear(1, hidden),
                nn.Tanh(),
                nn.Linear(hidden, d),
            )

    def augmented_tokens(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            return y
        e = self.embedding(x)
        e = e[:, None, :].expand(-1, self.N, -1)
        return torch.cat([y, e], dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = self.augmented_tokens(x, y)  # [B,N,d+q]
        A = self.U @ self.V.t()
        Az = torch.matmul(z, A.t())
        scores = torch.matmul(Az, z.transpose(1, 2))
        phi_scores = self.phi(scores)
        eye = torch.eye(self.N, device=y.device, dtype=torch.bool)[None, :, :]
        phi_scores = phi_scores.masked_fill(eye, 0.0)
        avg_phi = phi_scores.sum(dim=2) / max(self.N - 1, 1)  # [B,N]
        if self.learn_output_vector:
            return avg_phi[:, :, None] * self.u[None, None, :]
        return self.out_head(avg_phi.reshape(-1, 1)).reshape(y.shape[0], self.N, self.d)


class IndependentTokenDrift(nn.Module):
    """Ablation model: each token is predicted independently, no pairwise interaction.

    This baseline can optionally use the condition embedding e(X), but it does not
    use pairwise token interactions. The fixed_condition_embedding argument is
    included for compatibility with the theory-matching synthetic experiments.
    """

    def __init__(
        self,
        d: int,
        N: int,
        q: int,
        num_conditions: int,
        hidden: int = 128,
        use_condition: bool = True,
        fixed_condition_embedding: torch.Tensor | None = None,
    ):
        super().__init__()

        self.d = d
        self.N = N
        self.q = q if use_condition else 0
        self.use_condition = use_condition

        if self.use_condition and self.q > 0:
            if fixed_condition_embedding is not None:
                emb = fixed_condition_embedding.detach().float().clone()
                if emb.shape != (num_conditions, self.q):
                    raise ValueError(
                        f"fixed_condition_embedding has shape {tuple(emb.shape)}, "
                        f"expected {(num_conditions, self.q)}"
                    )
                self.embedding = nn.Embedding.from_pretrained(emb, freeze=True)
            else:
                self.embedding = nn.Embedding(num_conditions, self.q)
        else:
            self.embedding = None

        in_dim = d + self.q

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            e = self.embedding(x)[:, None, :].expand(-1, self.N, -1)
            z = torch.cat([y, e], dim=-1)
        else:
            z = y

        return self.net(z)


class TimeConditionedPairwiseDrift(nn.Module):
    """A simple time-conditioned pairwise drift for MNIST / latent-flow experiments.

    This is a practical model for empirical demonstration. It appends a scalar time
    embedding to every token before computing interactions.
    """

    def __init__(self, d: int, N: int, q: int, rank: int, num_conditions: int, hidden: int = 128):
        super().__init__()
        self.d = d
        self.N = N
        self.q = q
        self.rank = rank
        self.D_aug = d + q + 1
        self.embedding = nn.Embedding(num_conditions, q)
        self.U = nn.Parameter(torch.randn(self.D_aug, rank) / (self.D_aug ** 0.5))
        self.V = nn.Parameter(torch.randn(self.D_aug, rank) / (self.D_aug ** 0.5))
        self.phi = ScalarActivationMLP(hidden=hidden)
        self.out_head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t can be [B] or [B,1]. We broadcast it to all tokens.
        if t.dim() == 1:
            t = t[:, None]
        e = self.embedding(x)[:, None, :].expand(-1, self.N, -1)
        tt = t[:, None, :].expand(-1, self.N, 1)
        z = torch.cat([y, e, tt], dim=-1)
        A = self.U @ self.V.t()
        Az = torch.matmul(z, A.t())
        scores = torch.matmul(Az, z.transpose(1, 2))
        phi_scores = self.phi(scores)
        eye = torch.eye(self.N, device=y.device, dtype=torch.bool)[None, :, :]
        phi_scores = phi_scores.masked_fill(eye, 0.0)
        avg_phi = phi_scores.sum(dim=2) / max(self.N - 1, 1)
        return self.out_head(avg_phi.reshape(-1, 1)).reshape(y.shape[0], self.N, self.d)

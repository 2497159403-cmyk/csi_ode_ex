"""Fixed endpoint records, with labels and oracle quantities omitted."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .truth import batch_time, schedule


@dataclass
class EndpointBatch:
    x: torch.Tensor
    y0: torch.Tensor
    y1: torch.Tensor
    eta: torch.Tensor
    ids: list[str]

    def __post_init__(self):
        B = self.y0.shape[0]
        if self.y0.ndim != 3 or self.x.shape != (B, 1) or self.y1.shape != self.y0.shape or self.eta.shape != self.y0.shape:
            raise ValueError("endpoint shapes must be x:[B,1], y0/y1/eta:[B,N,d]")
        self.ids = list(self.ids)
        if len(self.ids) != B or len(set(self.ids)) != B:
            raise ValueError("endpoint IDs must be unique and match batch length")

    def __len__(self):
        return len(self.ids)

    def subset(self, indices):
        if isinstance(indices, slice):
            positions = list(range(len(self)))[indices]
        else:
            index = torch.as_tensor(indices, device="cpu")
            positions = index.nonzero().flatten().tolist() if index.dtype == torch.bool else index.flatten().tolist()
        selected = torch.as_tensor(positions, dtype=torch.long, device=self.x.device)
        return EndpointBatch(*(v.index_select(0, selected.to(v.device)) for v in (self.x, self.y0, self.y1, self.eta)),
                             [self.ids[i] for i in positions])

    def to(self, device=None, dtype=None):
        return EndpointBatch(*(v.to(device=device, dtype=dtype) for v in (self.x, self.y0, self.y1, self.eta)), list(self.ids))


def sample_endpoints(truth, M, seed, id_prefix) -> EndpointBatch:
    if type(M) is not int or M < 1:
        raise ValueError("M must be a positive integer")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    dtype = truth.dtype
    x = torch.rand(M, 1, generator=generator, dtype=dtype)
    shape = (M, truth.N, truth.d)
    y0 = torch.randn(shape, generator=generator, dtype=dtype)
    noise1 = torch.randn(shape, generator=generator, dtype=dtype)
    eta = torch.randn(shape, generator=generator, dtype=dtype)
    u = truth.u.to(dtype=dtype)
    if truth.protocol == "location":
        shift = truth.f(x)[:, :, None] * u
    else:
        p = torch.sigmoid(2 * (truth.c0 + truth.c1 * x))
        label_shape = (M, 1 if truth.protocol == "c2" else truth.N)
        labels = 2 * (torch.rand(label_shape, generator=generator, dtype=dtype) < p).to(dtype) - 1
        shift = truth.m * labels[:, :, None] * u
    y1 = noise1 + shift
    return EndpointBatch(x, y0, y1, eta, [f"{id_prefix}:{i}" for i in range(M)])


def derive(batch, t, protocol):
    s = schedule(protocol, batch_time(t, len(batch), batch.y0))
    def weighted(keys):
        return sum(s[k][:, None, None] * v for k, v in zip(keys, (batch.y0, batch.y1, batch.eta)))
    return weighted(("a", "b", "gamma")), weighted(("da", "db", "dgamma"))


def split_indices(M, val_fraction, seed):
    if type(M) is not int or M < 2 or not 0 < val_fraction < 1:
        raise ValueError("split requires M >= 2 and 0 < val_fraction < 1")
    order = torch.randperm(M, generator=torch.Generator(device="cpu").manual_seed(int(seed)))
    n_val = min(M - 1, max(1, round(M * val_fraction)))
    return order[n_val:], order[:n_val]

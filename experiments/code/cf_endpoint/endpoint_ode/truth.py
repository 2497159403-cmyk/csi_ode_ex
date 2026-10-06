"""Endpoint laws and population references; never passed to the fitting core."""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.special import log_ndtr


def schedule(protocol: str, t: torch.Tensor) -> dict:
    t = torch.as_tensor(t)
    if not t.is_floating_point():
        t = t.to(torch.float64)
    if not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError("time must be finite and in [0,1]")
    if protocol == "location":
        theta = math.pi * t * (t - 0.5)
        speed = math.pi * (2 * t - 0.5)
        a, b = theta.cos(), theta.sin()
        gamma = torch.zeros_like(t)
        da, db, dgamma = -b * speed, a * speed, gamma
    elif protocol in ("c1", "c2"):
        a, b = (math.pi * t / 2).cos().square(), (math.pi * t / 2).sin().square()
        sine = (math.pi * t).sin()
        # Exact endpoint coefficients retain dgamma and its noisy response.
        sine = torch.where((t == 0) | (t == 1), torch.zeros_like(sine), sine)
        gamma = sine / math.sqrt(2)
        db = math.pi / 2 * sine
        da, dgamma = -db, math.pi / math.sqrt(2) * (math.pi * t).cos()
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    return dict(a=a, b=b, gamma=gamma, da=da, db=db, dgamma=dgamma)


def batch_time(t, batch_size, reference):
    t = torch.as_tensor(t, dtype=reference.dtype, device=reference.device)
    if t.ndim == 0:
        return t.expand(batch_size)
    if t.shape != (batch_size,):
        raise ValueError(f"t must be scalar or [{batch_size}]")
    return t


class Truth:
    def __init__(self, config: dict):
        self.protocol = config["protocol"]
        if self.protocol not in ("location", "c1", "c2"):
            raise ValueError("unknown protocol")
        data, params = config["data"], config["truth"]
        self.N, self.d = data["N"], data["d"]
        if type(self.N) is not int or self.N < 2 or type(self.d) is not int or self.d < 1:
            raise ValueError("N >= 2 and d >= 1 must be integers")
        rank = config["model"]["rank"]
        if type(rank) is not int or not 1 <= rank <= self.d + 2:
            raise ValueError("invalid model rank")
        if self.protocol == "c2" and (self.N != 2 or rank < 2):
            raise ValueError("c2 requires N=2 and rank >= 2")
        self.u = torch.as_tensor(params["u"], dtype=torch.float64).clone()
        if self.u.shape != (self.d,) or not torch.isfinite(self.u).all() or self.u.norm() == 0:
            raise ValueError("u must be a finite nonzero vector of length d")
        dtype_name = config.get("runtime", {}).get("dtype", "float32")
        if dtype_name not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        self.dtype = getattr(torch, dtype_name)
        self.factor_budget = float(params["factor_budget"])
        self.activation_bound = float(params["activation_bound"])
        self.slope_bound = float(params["slope_bound"])
        if not all(math.isfinite(v) and v > 0 for v in (self.factor_budget, self.activation_bound, self.slope_bound)):
            raise ValueError("budgets must be finite and positive")
        if self.protocol == "location":
            bump = params["bump"]
            self.J, self.beta, self.delta = bump["J"], float(bump["beta"]), float(bump["delta"])
            self.omega = torch.as_tensor(bump["omega"], dtype=torch.float64)
            self.index_scale = float(params["index_scale"])
            if type(self.J) is not int or self.J < 1 or not math.isfinite(self.beta) or self.beta < 2:
                raise ValueError("J must be positive integer and beta >= 2")
            if not math.isfinite(self.delta) or self.delta <= 0 or self.omega.shape != (self.J,) or not ((self.omega == 0) | (self.omega == 1)).all():
                raise ValueError("delta must be positive and omega must contain J binary entries")
            if not math.isfinite(self.index_scale) or self.index_scale <= 0:
                raise ValueError("index_scale must be finite and positive")
            nuclear_bound = self.index_scale
            amplitude = 1.5 * math.pi * self.delta * self.J ** (-self.beta)
            slope = 1.5 * math.pi * self.delta * self.J ** (1 - self.beta) * (24 / math.e) / self.index_scale
        else:
            mix = params["mixture"]
            self.m, self.c0, self.c1 = (float(mix[k]) for k in ("m", "c0", "c1"))
            norm = self.u.norm().item()
            if not all(math.isfinite(v) for v in (self.m, self.c0, self.c1)):
                raise ValueError("mixture parameters must be finite")
            if not 0 < self.m <= 2 or norm > 2 or self.m * norm > 4 or max(abs(self.c0), abs(self.c0 + self.c1)) > 4:
                raise ValueError("mixture parameters exceed validated numerical range")
            multiplier = 2 if self.protocol == "c2" else 1
            nuclear_bound = multiplier * math.sqrt(self.m**2 * norm**2 + (self.c0**2 + self.c1**2) / multiplier**2)
            amplitude = slope = self.m * math.pi / 2
        self.budget_summary = dict(factor_budget=self.factor_budget, nuclear_norm_bound=nuclear_bound,
                                   activation_bound=self.activation_bound, activation_sup_bound=amplitude,
                                   slope_bound=self.slope_bound, activation_slope_sup_bound=slope)
        if nuclear_bound > self.factor_budget**2 + 1e-12 or amplitude > self.activation_bound + 1e-12 or slope > self.slope_bound + 1e-12:
            raise ValueError("truth exceeds structural or activation budget")
        ev = config.get("evaluation", {})
        self.xtol, self.rtol = float(ev.get("quantile_xtol", 1e-10)), float(ev.get("quantile_rtol", 1e-12))
        self.maxiter = ev.get("quantile_maxiter", 100)
        if not math.isfinite(self.xtol) or self.xtol <= 0 or not math.isfinite(self.rtol) or self.rtol < 0 or type(self.maxiter) is not int or self.maxiter < 1:
            raise ValueError("invalid quantile convergence controls")

    def f(self, x):
        if self.protocol != "location":
            raise ValueError("f is only defined for location")
        centers = (torch.arange(self.J, device=x.device, dtype=x.dtype) + 0.5) / self.J
        s = (x[..., None] - centers) * self.J
        inside = s.abs() < 1 / 3
        denominator = torch.where(inside, 1 - 9 * s.square(), torch.ones_like(s))
        psi = torch.where(inside, torch.exp(1 - denominator.reciprocal()), torch.zeros_like(s))
        return self.delta * self.J ** (-self.beta) * (psi * self.omega.to(x)).sum(-1)

    def _inputs(self, t, x, z):
        if z.ndim != 3 or z.shape[1:] != (self.N, self.d) or x.shape != (z.shape[0], 1):
            raise ValueError("expected x:[B,1], z:[B,N,d]")
        if not torch.isfinite(x).all() or not torch.isfinite(z).all():
            raise ValueError("x and z must be finite")
        return schedule(self.protocol, batch_time(t, z.shape[0], z)), x.to(z), self.u.to(z)

    def drift(self, t, x, z):
        s, x, u = self._inputs(t, x, z)
        if self.protocol == "location":
            scalar = s["db"][:, None] * self.f(x)
        else:
            projection = z @ u
            if self.protocol == "c2":
                projection = projection.sum(1, keepdim=True)
            posterior = (s["b"][:, None] * self.m * projection + self.c0 + self.c1 * x).tanh()
            scalar = self.m * s["db"][:, None] * posterior
        return (scalar[:, :, None] * u).expand_as(z)

    def activation(self, t, scores):
        """Oracle activation for membership tests, never used by a learner."""
        t = torch.as_tensor(t, dtype=scores.dtype, device=scores.device)
        db = schedule(self.protocol, t)["db"]
        while db.ndim < scores.ndim:
            db = db.unsqueeze(-1)
        return db * self.f(scores / self.index_scale) if self.protocol == "location" else self.m * db * scores.tanh()

    def membership(self, t):
        t = torch.as_tensor(t, dtype=torch.float64)
        s = schedule(self.protocol, t)
        D = self.d + 2
        c = torch.zeros(D, dtype=t.dtype, device=t.device)
        c[self.d] = 1
        if self.protocol == "location":
            other = torch.zeros_like(c)
            other[-1] = self.index_scale
            A = (c[:, None] * other[None, :]).expand(*t.shape, D, D)
            rank = 1
        else:
            divisor = 2 if self.protocol == "c2" else 1
            p = torch.cat((s["b"][..., None] * self.m * self.u.to(t),
                           torch.full((*t.shape, 1), self.c0 / divisor, dtype=t.dtype, device=t.device),
                           torch.full((*t.shape, 1), self.c1 / divisor, dtype=t.dtype, device=t.device)), -1)
            A = p[..., :, None] * c[None, :]
            if self.protocol == "c2":
                A = A + c[:, None] * p[..., None, :]
            rank = divisor
        left, singular, right = torch.linalg.svd(A, full_matrices=False)
        U = left[..., :rank] * singular[..., None, :rank].sqrt()
        V = right.transpose(-1, -2)[..., :rank] * singular[..., None, :rank].sqrt()
        return dict(A=A, U=U, V=V, rank_bound=rank, **self.budget_summary)

    def flow(self, t, x, z0, *, return_diagnostics=False):
        s, x, u = self._inputs(t, x, z0)
        if self.protocol == "location":
            result = z0 + s["b"][:, None, None] * self.f(x)[:, :, None] * u
            diagnostics = dict(method="analytic_location", reference_dtype=str(z0.dtype), max_bracket_width=0.0, iterations=0, failures=0)
        else:
            # CPU float64 computations do not clip CDF probabilities, even at ±40.
            z = z0.detach().cpu().double()
            xx = x.detach().cpu().double().numpy()
            tt = batch_time(t, z.shape[0], z).detach().cpu()
            b = schedule(self.protocol, tt)["b"].numpy()[:, None]
            v = self.u / self.u.norm()
            q0 = (z @ v).numpy()
            collective = self.protocol == "c2"
            if collective:
                q0 = q0.sum(1, keepdims=True) / math.sqrt(2)
            kappa = np.broadcast_to(b * self.m * self.u.norm().item() * (math.sqrt(2) if collective else 1), q0.shape)
            ell = np.broadcast_to(self.c0 + self.c1 * xx, q0.shape)
            logp, log1mp = -np.logaddexp(0, -2 * ell), -np.logaddexp(0, 2 * ell)
            lower, upper = q0 - np.abs(kappa), q0 + np.abs(kappa)
            negative = q0 < 0
            target = log_ndtr(np.where(negative, q0, -q0))
            iterations = 0
            for iterations in range(self.maxiter + 1):
                mid = (lower + upper) / 2
                active = upper - lower > self.xtol + self.rtol * np.maximum(1, np.abs(mid))
                if not active.any():
                    break
                if iterations == self.maxiter:
                    raise RuntimeError(f"quantile reference failed for {active.sum()} coordinates after {self.maxiter} iterations")
                logcdf = np.logaddexp(log1mp + log_ndtr(mid + kappa), logp + log_ndtr(mid - kappa))
                logsf = np.logaddexp(log1mp + log_ndtr(-mid - kappa), logp + log_ndtr(-mid + kappa))
                move_lower = np.where(negative, logcdf < target, logsf > target)
                lower = np.where(active & move_lower, mid, lower)
                upper = np.where(active & ~move_lower, mid, upper)
            delta = torch.from_numpy((lower + upper) / 2 - q0)
            if collective:
                delta = delta / math.sqrt(2)
            result = (z + delta[..., None] * v).to(device=z0.device, dtype=z0.dtype)
            # Exact identity includes bitwise preservation of original dtype/device.
            zero = torch.from_numpy((kappa == 0).all(1)).to(z0.device)
            result = torch.where(zero[:, None, None], z0, result)
            diagnostics = dict(method="population_quantile_logcdf_logsf_bisection", reference_dtype="float64", reference_device="cpu",
                               max_bracket_width=float(np.max(upper - lower, initial=0)), iterations=iterations, failures=0,
                               xtol=self.xtol, rtol=self.rtol, output_dtype=str(z0.dtype), output_device=str(z0.device),
                               conversion=f"{z0.device}/{z0.dtype} -> cpu/float64 -> {z0.device}/{z0.dtype}")
        return (result, diagnostics) if return_diagnostics else result


def make_truth(config: dict) -> Truth:
    return Truth(config)

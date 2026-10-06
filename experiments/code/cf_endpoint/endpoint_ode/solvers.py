"""Fixed-grid ODE methods and explicitly qualified refinement diagnostics."""
import math
import torch


def _validate(K, t_end):
    if type(K) is not int or K < 1 or not math.isfinite(t_end) or not 0 <= t_end <= 1:
        raise ValueError("K must be positive integer and t_end in [0,1]")


def _field(field, t, x, z):
    out = field(t, x, z)
    if out.shape != z.shape or not torch.isfinite(out).all():
        raise FloatingPointError("ODE field returned invalid shape or nonfinite values")
    return out


@torch.no_grad()
def euler(field, x, z0, K, t_end=1.0):
    _validate(K, t_end)
    z = z0.clone()
    h = t_end / K
    for k in range(K):
        z = z + h * _field(field, k * h, x, z)
    if not torch.isfinite(z).all():
        raise FloatingPointError("nonfinite Euler endpoint")
    return z


@torch.no_grad()
def rk4(field, x, z0, K, t_end=1.0):
    _validate(K, t_end)
    z = z0.clone()
    h = t_end / K
    for k in range(K):
        t = k * h
        a = _field(field, t, x, z)
        b = _field(field, t + h/2, x, z + h*a/2)
        c = _field(field, t + h/2, x, z + h*b/2)
        d = _field(field, min(t_end, t + h), x, z + h*c)
        z = z + h*(a + 2*b + 2*c + d)/6
    if not torch.isfinite(z).all():
        raise FloatingPointError("nonfinite RK4 endpoint")
    return z


@torch.no_grad()
def refined_reference(field, x, z0, config):
    """Return finest RK4 endpoint; resolution is empirical, never an error bound.

    Comparison uses a fixed diagnostic ensemble and the finest requested Euler K.
    Every refinement recomputes its threshold against the new reference endpoint.
    """
    steps = config["reference_steps"]
    refinements = config["reference_max_refinements"]
    fraction, floor = config["reference_rms_fraction"], config["reference_abs_tol"]
    if type(refinements) is not int or refinements < 0 or not 0 < fraction or not 0 < floor:
        raise ValueError("invalid refinement controls")
    finest_k = max(config["K_values"])
    baseline = euler(field, x, z0, finest_k)
    coarse = rk4(field, x, z0, steps)
    history = []
    resolved = False
    relative_resolved = False
    criterion_basis = "not_compared"
    for _ in range(refinements):
        fine = rk4(field, x, z0, steps * 2)
        change = (fine - coarse).double().square().mean().sqrt().item()
        euler_rms = (baseline - fine).double().square().mean().sqrt().item()
        threshold = max(floor, fraction * euler_rms)
        relative_resolved = change <= fraction * euler_rms
        criterion_basis = "absolute_floor" if floor > fraction * euler_rms else "relative_euler_rms"
        history.append(dict(coarse_steps=steps, fine_steps=steps*2, adjacent_rms=change,
                            finest_euler_rms=euler_rms, threshold=threshold,
                            relative_resolved=relative_resolved, criterion_basis=criterion_basis))
        coarse, steps = fine, steps*2
        if change <= threshold:
            resolved = True
            break
    return coarse, dict(method="rk4_doubling", steps=steps, resolved=resolved, history=history,
                        relative_resolved=relative_resolved, criterion_basis=criterion_basis,
                        eligible_for_relative_slope=resolved and relative_resolved,
                        finest_euler_K=finest_k, rms_fraction=fraction, absolute_floor=floor,
                        dtype=str(z0.dtype), device=str(z0.device), n_diagnostic=len(z0),
                        interpretation="Adjacent-reference per-coordinate RMS criterion, not a rigorous error bound; unresolved finest points must not support slope claims.")

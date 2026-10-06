"""General utilities for all experiments.

This file contains only small helper functions: device selection, random seeds,
run-directory creation, JSON/CSV saving, and log-log slope fitting.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch


def get_device(device_arg: str = "auto") -> torch.device:
    """Return CUDA device when available, otherwise CPU.

    Parameters
    ----------
    device_arg:
        "auto", "cuda", or "cpu".
    """
    device_arg = str(device_arg).lower()
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[warning] --device cuda was requested, but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif device_arg == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device argument: {device_arg}")

    print(f"[device] Using device: {device}")
    if device.type == "cuda":
        print(f"[device] CUDA device name: {torch.cuda.get_device_name(0)}")
    return device


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic flags make debugging easier. They can make training slower.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    """Create outputs/<experiment_name>/<timestamp>/ and return it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def to_serializable(obj: Any) -> Any:
    """Convert common Python objects into JSON-serializable values."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Save dictionary to JSON with readable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(obj), f, indent=2, ensure_ascii=False)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_loglog_slope(xs: Iterable[float], ys: Iterable[float]) -> Tuple[float, float]:
    """Fit log(y) = slope * log(x) + intercept.

    Returns
    -------
    slope, intercept
    """
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)
    return float(slope), float(intercept)


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Save model state, optimizer state, and extra metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def state_dict_sha256(state: Dict[str, torch.Tensor]) -> str:
    """Return a stable SHA256 fingerprint for a tensor state dictionary.

    The fingerprint is saved with each synthetic run so that later pointwise
    reruns can be checked against the original ground-truth system. Two points
    should only be merged when their fingerprints are identical and all other
    experimental settings match.
    """
    import hashlib

    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()

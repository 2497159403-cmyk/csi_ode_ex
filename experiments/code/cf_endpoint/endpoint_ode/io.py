"""Atomic artifacts, deterministic seeds, device and experiment identity."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import tempfile

import numpy as np
import torch

SEED_RULE = "int.from_bytes(SHA256(canonical_json([base_seed,*parts]))[:8], 'big') % (2**63-1)"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def hash_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def derive_seed(base_seed, *parts):
    digest = hashlib.sha256(canonical_json([int(base_seed), *parts]).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def sha256_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def code_hash():
    root = Path(__file__).parent
    return hash_json({p.name: sha256_file(p) for p in sorted(root.glob("*.py"))})


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def atomic_torch_save(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    atomic_bytes(path, buffer.getvalue())


def atomic_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def resolve_device(requested="auto"):
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace not in {":4096:8", ":16:8"}:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA")
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())}
    actual = next((v for v in ("cuda", "mps", "cpu") if available[v])) if requested == "auto" else requested
    if not available[actual]:
        raise RuntimeError(f"Explicit device {requested!r} is unavailable: {available}")
    return torch.device(actual)


def resolve_dtype(name):
    if name not in {"float32", "float64"}:
        raise ValueError("runtime.dtype must be float32 or float64")
    return getattr(torch, name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "generator": generator.get_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "torch_mps": torch.mps.get_rng_state() if hasattr(torch, "mps") and torch.backends.mps.is_available() else None}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    generator.set_state(state["generator"].cpu())
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["torch_cuda"]])
    if state["torch_mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"].cpu())


def environment(requested, device):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "requested_device": requested, "actual_device": str(device),
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            "num_threads": torch.get_num_threads(), "pid": os.getpid(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}

#!/usr/bin/env python3
"""Run the archived neural algorithms using the checksum-verified paper configs.

Runs preserve every recorded scientific option and write to a fresh destination.
The paper's frozen result tables are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
NEURAL = HERE / "neural"
MODULES = {
    "fixed_time": "experiments.synthetic_fixed_time",
    "path_matched": "experiments.ode_generation_error_v2",
    "dimension_sensitivity": "experiments.synthetic_dimension_free",
    "condition_ablation": "experiments.synthetic_condition_ablation",
}


def configuration(experiment):
    relative = f"data/{experiment}/config.json"
    path = ASSETS / relative
    manifest = json.loads((ASSETS / "provenance" / "source_manifest.json").read_text())
    entry = next(row for row in manifest["files"] if row["copied_path"] == relative)
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Frozen experiment config checksum mismatch")
    return json.loads(path.read_text())


def arguments(config):
    argv = []
    for name, value in config.items():
        if name == "data_protocol" or value is None or value is False:
            continue
        argv.append(f"--{name}")
        if value is not True:
            argv.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=MODULES)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh destination, separate from the frozen paper assets")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"),
                        help="Runtime override only; omitted preserves the recorded choice")
    args = parser.parse_args()
    config = configuration(args.experiment)
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError("Output destination already exists; use a fresh directory")
    if destination.is_relative_to(ASSETS.parent):
        raise ValueError("Output must be outside the delivered paper directory")
    if args.device is not None:
        config["device"] = args.device
    import torch
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Recorded CUDA runtime is unavailable; request --device cpu explicitly")
    destination.mkdir(parents=True)
    config["output_dir"] = str(destination)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("MPLBACKEND", "Agg")
    with tempfile.TemporaryDirectory(prefix="neural-reproduce-mpl-") as cache:
        environment.setdefault("MPLCONFIGDIR", cache)
        command = [sys.executable, "-m", MODULES[args.experiment], *arguments(config)]
        return subprocess.run(command, cwd=NEURAL, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())

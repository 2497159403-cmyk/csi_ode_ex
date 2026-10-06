#!/usr/bin/env python3
"""Verify the delivered sources and all frozen paper results; write no output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_delivery() -> int:
    manifest_path = HERE / "provenance" / "delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = {}
    for row in manifest["files"]:
        relative = row["path"]
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in declared:
            raise AssertionError(f"Invalid delivery path: {relative}")
        declared[relative] = row["sha256"]
    actual = {
        path.relative_to(HERE).as_posix()
        for path in HERE.rglob("*")
        if path.is_file() and path != manifest_path
        and path.relative_to(HERE).parts[0] != "data"
        and not any(part in {"outputs", "__pycache__", ".mplcache"}
                    or part.startswith(".venv") for part in path.relative_to(HERE).parts)
        and path.suffix != ".pyc"
    }
    if actual != set(declared):
        raise AssertionError(f"Delivery file set changed: missing={sorted(set(declared)-actual)}, "
                             f"extra={sorted(actual-set(declared))}")
    for relative, expected in sorted(declared.items()):
        if digest(HERE / relative) != expected:
            raise AssertionError(f"Delivery source changed: {relative}")
    formal = HERE / "code" / "cf_endpoint" / "configs" / "cf_sweep_formal.json"
    if digest(formal) != digest(HERE / "data" / "cf_endpoint" / "config.json"):
        raise AssertionError("CF executable and archived formal configs differ")
    return len(declared)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    count = verify_delivery()
    from plot_figures import validate_results
    _, report = validate_results()
    report["delivery_files_verified"] = count
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

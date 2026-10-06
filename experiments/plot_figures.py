#!/usr/bin/env python3
"""Validate frozen results and reproduce the six PDF figures used by the paper.

This entry point only redraws archived results; executable training and CF
generation/solve/scoring sources are provided separately under ``code/``.
Before any output is written, the script verifies all copied inputs
against the frozen manifest, validates every reported selection and identity,
and recomputes means and sample SDs (ddof=1).  An externally supplied
``MPLCONFIGDIR`` is respected; otherwise Matplotlib uses a temporary cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_TEMP_MPL_CACHE: tempfile.TemporaryDirectory[str] | None = None
if os.environ.get("MPLCONFIGDIR"):
    MPL_CACHE = Path(os.environ["MPLCONFIGDIR"]).expanduser()
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
else:
    _TEMP_MPL_CACHE = tempfile.TemporaryDirectory(prefix="ode-experiment-mpl-")
    MPL_CACHE = Path(_TEMP_MPL_CACHE.name)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA = HERE / "data"
FIGURES = HERE.parent / "figures"
RTOL = 1e-10
ATOL = 1e-14
CF_AGGREGATE_IDENTITY_ATOL = 5e-13
CF_RECORDED_POINTWISE_ATOL = 5e-13
EXPECTED_DATA_FILE_COUNT = 20
EXPECTED_DATASET_UNITS = {
    "fixed_time": "noiseless field squared Frobenius error divided by N and test regression examples",
    "path_matched": "learned/reference Euler-200 endpoint squared difference divided by N*d and evaluation trajectories",
    "dimension_sensitivity": "noiseless field squared Frobenius error divided by N and test regression examples",
    "condition_ablation": "noiseless field squared Frobenius error divided by N and test regression examples",
    "cf_endpoint": "squared Frobenius error divided by N and test endpoints",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL):
        raise AssertionError(f"{label}: {actual:.17g} != {expected:.17g}")


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def compare_cell(actual: str, expected: str, label: str) -> None:
    """Compare a CSV cell numerically when possible, otherwise byte-for-byte."""
    try:
        actual_number = float(actual)
        expected_number = float(expected)
    except ValueError:
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    else:
        close(actual_number, expected_number, label)


def verify_manifest(
    data_root: Path = DATA,
    manifest_path: Path = HERE / "provenance" / "source_manifest.json",
) -> dict[str, object]:
    """Verify copied inputs against the existing manifest without rewriting it."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != EXPECTED_DATA_FILE_COUNT:
        raise AssertionError(f"manifest: expected {EXPECTED_DATA_FILE_COUNT} file entries")

    declared: dict[str, str] = {}
    for entry in entries:
        relative = entry.get("copied_path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not relative.startswith("data/"):
            raise AssertionError(f"manifest: invalid copied_path {relative!r}")
        if relative in declared:
            raise AssertionError(f"manifest: duplicate copied_path {relative}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise AssertionError(f"manifest: invalid SHA256 for {relative}")
        declared[relative] = expected_hash

    actual_paths = {
        f"data/{path.relative_to(data_root).as_posix()}"
        for path in data_root.rglob("*")
        if path.is_file()
    }
    if set(declared) != actual_paths:
        missing = sorted(set(declared) - actual_paths)
        extra = sorted(actual_paths - set(declared))
        raise AssertionError(f"manifest file set mismatch; missing={missing}, extra={extra}")
    for relative, expected_hash in sorted(declared.items()):
        path = data_root / Path(relative).relative_to("data")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise AssertionError(f"manifest SHA256 mismatch for {relative}")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise AssertionError("manifest: missing datasets metadata")
    for name, expected_unit in EXPECTED_DATASET_UNITS.items():
        if name not in datasets or datasets[name].get("unit") != expected_unit:
            raise AssertionError(f"manifest: unit metadata mismatch for {name}")
    return manifest


def validate_selection(
    raw: list[dict[str, str]],
    all_rows: list[dict[str, str]],
    group_fields: tuple[str, ...],
    value_field: str,
    expected_restarts: int,
) -> None:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    if len(raw) != len(groups):
        raise AssertionError("selected row count does not match restart groups")
    raw_key_list = [tuple(row[field] for field in group_fields) for row in raw]
    raw_keys = set(raw_key_list)
    if len(raw_key_list) != len(raw_keys):
        raise AssertionError("results_raw contains duplicate group keys")
    if raw_keys != set(groups):
        raise AssertionError("selected group keys are incomplete")
    raw_by_key = {tuple(row[field] for field in group_fields): row for row in raw}
    for key, rows in groups.items():
        if len(rows) != expected_restarts:
            raise AssertionError(f"{key}: expected {expected_restarts} restarts")
        restart_ids = [row["restart"] for row in rows]
        if len(restart_ids) != len(set(restart_ids)):
            raise AssertionError(f"{key}: duplicate restart identifiers")
        selected = [row for row in rows if row["selected"] == "True"]
        if len(selected) != 1:
            raise AssertionError(f"{key}: not exactly one selected restart")
        selected_row = selected[0]
        minimum = min(float(row["val_mse"]) for row in rows)
        close(float(selected_row["val_mse"]), minimum, f"{key} validation choice")

        raw_row = raw_by_key[key]
        raw_restart_field = "restart" if "restart" in raw_row else "best_restart"
        raw_validation_field = "val_mse" if "val_mse" in raw_row else "best_val_mse"
        compare_cell(raw_row[raw_restart_field], selected_row["restart"], f"{key} selected restart binding")
        compare_cell(raw_row[raw_validation_field], selected_row["val_mse"], f"{key} selected validation binding")
        plotted_value = float(raw_row[value_field])
        if not math.isfinite(plotted_value):
            raise AssertionError(f"{key}: non-finite plotted value {value_field}")

        # Bind every directly corresponding metadata/metric field.  The path
        # archive stores its endpoint score only in results_raw, so for that
        # one metric the selected restart and validation/metadata binding is
        # the available provenance link.
        renamed = {raw_restart_field, raw_validation_field}
        for field in sorted((set(raw_row) & set(selected_row)) - renamed):
            compare_cell(raw_row[field], selected_row[field], f"{key} selected field {field}")
        if value_field in selected_row:
            compare_cell(raw_row[value_field], selected_row[value_field], f"{key} plotted value binding")


def summarize_old(
    folder: str,
    group_fields: tuple[str, ...],
    value_field: str,
    summary_keys: tuple[str, ...],
    mean_field: str,
    sd_field: str,
    expected_count: int,
    expected_restarts: int,
    data_root: Path = DATA,
) -> tuple[list[dict[str, str]], dict[tuple[str, ...], tuple[float, float, list[float]]]]:
    base = data_root / folder
    raw = read_rows(base / "results_raw.csv")
    all_rows = read_rows(base / "results_all_restarts.csv")
    summary = read_rows(base / "source_summary.csv")
    if len(raw) != expected_count:
        raise AssertionError(f"{folder}: expected {expected_count} selected rows")
    if raw and "selected" in raw[0] and any(row["selected"] != "True" for row in raw):
        raise AssertionError(f"{folder}: results_raw contains an unselected row")
    validate_selection(raw, all_rows, group_fields, value_field, expected_restarts)
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in raw:
        grouped[tuple(row[field] for field in summary_keys)].append(float(row[value_field]))
    output: dict[tuple[str, ...], tuple[float, float, list[float]]] = {}
    source = {tuple(row[field] for field in summary_keys): row for row in summary}
    if set(grouped) != set(source):
        raise AssertionError(f"{folder}: raw and summary groups differ")
    for key, values in grouped.items():
        mean, sd = mean_sd(values)
        close(mean, float(source[key][mean_field]), f"{folder} {key} mean")
        close(sd, float(source[key][sd_field]), f"{folder} {key} sample SD")
        output[key] = (mean, sd, values)
    return raw, output


def validate_cf(data_root: Path = DATA) -> tuple[list[dict[str, str]], dict[tuple[int, str], dict[str, float]], list[dict[str, str]]]:
    per_seed = read_rows(data_root / "cf_endpoint" / "per_seed.csv")
    source_summary = read_rows(data_root / "cf_endpoint" / "source_summary.csv")
    euler = read_rows(data_root / "cf_endpoint" / "euler.csv")
    seeds = list(range(2026100301, 2026100311))
    sizes = [500, 1000, 2000, 4000, 8000]
    methods = ["CF", "train_mean", "zero"]
    if len(per_seed) != 150:
        raise AssertionError("CF: expected 150 per-seed method rows")
    expected = {(seed, size, method) for seed in seeds for size in sizes for method in methods}
    observed = {(int(r["seed"]), int(r["M"]), r["method"]) for r in per_seed}
    if observed != expected:
        raise AssertionError("CF: incomplete seed x M x method grid")
    for row in per_seed:
        size = int(row["M"])
        if int(row["n_fit"]) != 9 * size // 10 or int(row["n_validation"]) != size // 10:
            raise AssertionError(f"CF {(row['seed'], size, row['method'])}: split metadata mismatch")
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_seed:
        grouped[(int(row["M"]), row["method"])].append(row)
    source = {(int(r["M"]), r["method"]): r for r in source_summary}
    if len(source) != len(source_summary) or set(source) != set(grouped):
        raise AssertionError("CF: source summary groups are duplicate or incomplete")
    derived: dict[tuple[int, str], dict[str, float]] = {}
    for key, rows in grouped.items():
        terminal = [float(r["terminal_per_token"]) for r in rows]
        field = [float(r["field_per_token"]) for r in rows]
        terminal_mean, terminal_sd = mean_sd(terminal)
        field_mean, field_sd = mean_sd(field)
        src = source[key]
        close(terminal_mean, float(src["terminal_mean"]), f"CF {key} terminal mean")
        close(terminal_sd, float(src["terminal_sample_SD"]), f"CF {key} terminal SD")
        close(field_mean, float(src["field_mean"]), f"CF {key} field mean")
        close(field_sd, float(src["field_sample_SD"]), f"CF {key} field SD")
        ratio_residual = max(
            abs(float(r["field_per_token"]) - (math.pi**2 / 4) * float(r["terminal_per_token"]))
            for r in rows
        )
        if ratio_residual > CF_AGGREGATE_IDENTITY_ATOL:
            raise AssertionError(
                f"CF {key}: aggregate field/terminal identity residual "
                f"{ratio_residual:.3e} exceeds {CF_AGGREGATE_IDENTITY_ATOL:.1e}"
            )
        same_precision_pointwise_max = max(float(r["ratio_max_abs"]) for r in rows)
        runtime_pointwise_max = max(float(r["runtime_ratio_max_abs"]) for r in rows)
        if not math.isfinite(same_precision_pointwise_max) or same_precision_pointwise_max < 0:
            raise AssertionError(f"CF {key}: invalid recorded same-precision residual")
        if same_precision_pointwise_max > CF_RECORDED_POINTWISE_ATOL:
            raise AssertionError(
                f"CF {key}: recorded pointwise residual {same_precision_pointwise_max:.3e} "
                f"exceeds {CF_RECORDED_POINTWISE_ATOL:.1e}"
            )
        if not math.isfinite(runtime_pointwise_max) or runtime_pointwise_max < 0:
            raise AssertionError(f"CF {key}: invalid recorded runtime residual")
        close(
            same_precision_pointwise_max,
            float(src["ratio_residual_max"]),
            f"CF {key} pointwise same-precision identity",
        )
        wins = 0
        if key[1] == "CF":
            base = {
                int(r["seed"]): float(r["terminal_per_token"])
                for r in grouped[(key[0], "train_mean")]
            }
            wins = sum(float(r["terminal_per_token"]) < base[int(r["seed"])] for r in rows)
            if wins != int(src["wins_train_mean"]):
                raise AssertionError(f"CF {key}: paired wins mismatch")
        derived[key] = {
            "terminal_mean": terminal_mean,
            "terminal_sd": terminal_sd,
            "field_mean": field_mean,
            "field_sd": field_sd,
            "ratio_residual": ratio_residual,
            "same_precision_pointwise_max": same_precision_pointwise_max,
            "runtime_pointwise_max": runtime_pointwise_max,
            "wins": float(wins),
        }
    if len(euler) != 200:
        raise AssertionError("Euler: expected 200 method x seed x K rows")
    euler_expected = {
        (seed, method, steps)
        for seed in seeds
        for method in ("CF", "train_mean", "zero", "oracle_numerical_reference")
        for steps in (10, 20, 40, 80, 160)
    }
    euler_observed = {(int(r["seed"]), r["method"], int(r["K"])) for r in euler}
    if euler_observed != euler_expected:
        raise AssertionError("Euler: incomplete method x seed x K grid")
    for row in euler:
        exact = float(row["exact_risk"])
        disc = float(row["discretization"])
        roundoff = float(row["rounding"])
        total = float(row["total"])
        round_cross = float(row["rounding_cross"])
        risk_cross = float(row["risk_cross_double"])
        close(total, disc + roundoff + round_cross, "Euler runtime-exact decomposition")
        close(float(row["euler_risk_double"]), exact + disc + risk_cross, "Euler target-risk decomposition")
        if row["method"] == "zero":
            close(disc, 0.0, "zero discretization")
    return per_seed, derived, euler


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_particle_reference(ax: plt.Axes, sizes: list[int], anchor: float) -> None:
    """Visual polynomial reference, not a fitted curve or a data series."""
    beta = 2.0
    x = np.asarray(sizes, dtype=np.float64)
    reference = anchor * (x / x[-1]) ** (-2.0 * beta / (2.0 * beta + 1.0))
    ax.plot(x, reference, color="#666666", linestyle="--", linewidth=1.2,
            label=r"$M^{-4/5}$ reference ($\beta=2$)")


def plot_particle_fixed_time(fixed: dict) -> dict[str, object]:
    sizes = [500, 1000, 2000, 4000, 8000, 16000]
    means = [fixed[(str(m),)][0] for m in sizes]
    sds = [fixed[(str(m),)][1] for m in sizes]
    fig, ax = base_axes()
    ax.errorbar(sizes, means, yerr=sds, marker="o", capsize=3,
                color="#1261a0", label=r"mean $\pm$ sample SD")
    add_particle_reference(ax, sizes, means[-1])
    ax.set(xscale="log", yscale="log", xlabel="total regression sample budget $M$",
           ylabel="field MSE per token")
    ax.set_xticks(sizes, ["500", "1k", "2k", "4k", "8k", "16k"])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.legend(frameon=False)
    save_figure(fig, "fig_particle_fixed_time")
    slope, _ = np.polyfit(np.log(sizes), np.log(means), 1)
    return {"beta_reference": 2.0, "reference_slope": -0.8,
            "anchor_M": sizes[-1], "anchor_mean": means[-1],
            "descriptive_mean_loglog_slope": float(slope),
            "fit_budgets": sizes, "fit_rule": "unweighted; all budget means"}


def plot_particle_dimensions(dim: dict) -> dict[str, object]:
    sizes = [2000, 4000, 8000, 16000]
    fig, ax = base_axes()
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, 5))
    for d, color in zip((4, 8, 16, 32, 64), colors):
        means = [dim[(str(d), str(m))][0] for m in sizes]
        sds = [dim[(str(d), str(m))][1] for m in sizes]
        ax.errorbar(sizes, means, yerr=sds, marker="o", capsize=2,
                    color=color, label=f"$d={d}$")
    anchor = dim[("8", str(sizes[-1]))][0]
    add_particle_reference(ax, sizes, anchor)
    ax.set(xscale="log", yscale="log", xlabel="total regression sample budget $M$",
           ylabel="field MSE per token")
    ax.set_xticks(sizes, ["2k", "4k", "8k", "16k"])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=10.5, columnspacing=1.2)
    save_figure(fig, "fig_dimension_sensitivity")
    return {"beta_reference": 2.0, "reference_slope": -0.8,
            "anchor_d": 8, "anchor_M": sizes[-1], "anchor_mean": anchor}


def base_axes(figsize=(4.9, 3.35)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
    return fig, ax


def reference_statistics(fixed: dict, dim: dict) -> dict[str, object]:
    """Check the beta=2 references and the descriptive all-budget fit."""
    sizes = [500, 1000, 2000, 4000, 8000, 16000]
    means = [fixed[(str(m),)][0] for m in sizes]
    slope, _ = np.polyfit(np.log(sizes), np.log(means), 1)
    close(float(slope), -0.8096480761970609, "fixed-time all-budget log-log slope")
    if f"{slope:.3f}" != "-0.810":
        raise AssertionError("Fixed-time descriptive slope must round to -0.810")
    return {
        "beta_reference": 2.0,
        "reference_slope": -0.8,
        "fixed_time_anchor_M": 16000,
        "fixed_time_anchor_mean": means[-1],
        "dimension_anchor_d": 8,
        "dimension_anchor_M": 16000,
        "dimension_anchor_mean": dim[("8", "16000")][0],
        "descriptive_mean_loglog_slope": float(slope),
        "fit_budgets": sizes,
        "fit_rule": "unweighted; all budget means",
    }


def validate_results() -> tuple[dict, dict[str, object]]:
    """Validate every archived input and statistic without writing output."""
    verify_manifest()
    fixed_raw, fixed = summarize_old(
        "fixed_time", ("M", "repeat"), "test_mse", ("M",),
        "test_mse_mean", "test_mse_std", 42, 2,
    )
    path_raw, path = summarize_old(
        "path_matched", ("M", "repeat"), "coupled_endpoint_mse", ("M",),
        "coupled_endpoint_mse_mean", "coupled_endpoint_mse_std", 35, 4,
    )
    dim_raw, dim = summarize_old(
        "dimension_sensitivity", ("d", "M", "repeat"), "test_mse", ("d", "M"),
        "test_mse_mean", "test_mse_std", 100, 4,
    )
    ablation_raw, ablation = summarize_old(
        "condition_ablation", ("model", "repeat"), "test_mse", ("model",),
        "test_mse_mean", "test_mse_std", 20, 4,
    )
    cf_rows, cf, euler = validate_cf()
    for folder in ("fixed_time", "dimension_sensitivity"):
        config = json.loads((DATA / folder / "config.json").read_text(encoding="utf-8"))
        close(float(config["beta"]), 2.0, f"{folder} reference beta")
    report = {
        "status": "passed",
        "selected_rows": {"fixed_time": len(fixed_raw), "path_matched": len(path_raw),
                          "dimension_sensitivity": len(dim_raw), "condition_ablation": len(ablation_raw)},
        "cf_method_rows": len(cf_rows),
        "euler_rows": len(euler),
        "rtol": RTOL, "atol": ATOL,
        "cf_aggregate_identity_atol": CF_AGGREGATE_IDENTITY_ATOL,
        "manifest_files_verified": EXPECTED_DATA_FILE_COUNT,
        "particle_reference": reference_statistics(fixed, dim),
    }
    return {"fixed": fixed, "path": path, "dim": dim, "ablation": ablation,
            "cf_rows": cf_rows, "cf": cf, "euler": euler}, report


def main() -> None:
    global FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES,
                        help="PDF destination; default: ../figures/")
    args = parser.parse_args()
    FIGURES = args.output_dir.resolve()
    results, report = validate_results()
    fixed, path, dim, ablation = (results[name] for name in ("fixed", "path", "dim", "ablation"))
    cf_rows, cf, euler = (results[name] for name in ("cf_rows", "cf", "euler"))
    plt.rcParams.update(
        {
            # The PDFs are placed at about 0.62--0.65 of their native width in
            # the two-column paper.  These native sizes therefore render at
            # roughly 7.5--8 pt in the compiled manuscript.
            "font.size": 12.5,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 1.5,
            "lines.markersize": 5.5,
        }
    )
    sizes = [500, 1000, 2000, 4000, 8000, 16000]
    plot_particle_fixed_time(fixed)

    cf_sizes = [500, 1000, 2000, 4000, 8000]
    cf_methods = [("CF", "#b2182b", "o"), ("train_mean", "#2166ac", "s"), ("zero", "#4d4d4d", "^")]
    fig, ax = base_axes()
    for method, color, marker in cf_methods:
        method_rows = [r for r in cf_rows if r["method"] == method]
        by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in method_rows:
            by_seed[int(row["seed"])].append(row)
        for rows in by_seed.values():
            rows.sort(key=lambda r: int(r["M"]))
            ax.plot([int(r["M"]) for r in rows], [float(r["terminal_per_token"]) for r in rows], color=color, alpha=0.12, linewidth=0.7)
        means = [cf[(m, method)]["terminal_mean"] for m in cf_sizes]
        ax.plot(cf_sizes, means, color=color, marker=marker, label=method.replace("_", " "))
        for m in cf_sizes:
            rows = [r for r in method_rows if int(r["M"]) == m]
            seed_values = [float(r["terminal_per_token"]) for r in rows]
            ax.scatter([m] * len(rows), seed_values, color=color, s=9, alpha=0.24, edgecolors="none")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="total endpoint budget $M$",
        ylabel="exact-flow terminal\ncoupling MSE\nper token",
    )
    ax.set_xticks(cf_sizes, [str(m) for m in cf_sizes])
    ax.legend(frameon=False)
    save_figure(fig, "fig_cf_endpoint_terminal")

    path_sizes = [500, 1000, 2000, 4000, 8000, 16000, 32000]
    fig, ax = base_axes()
    means = [path[(str(m),)][0] for m in path_sizes]
    sds = [path[(str(m),)][1] for m in path_sizes]
    ax.errorbar(path_sizes, means, yerr=sds, marker="o", capsize=3, color="#6a3d9a", label="mean $\\pm$ sample SD")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="training path states $M$",
        ylabel="Euler-200 endpoint MSE\nper coordinate",
    )
    ax.set_xticks(path_sizes, ["500", "1k", "2k", "4k", "8k", "16k", "32k"])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.legend(frameon=False)
    save_figure(fig, "fig_path_matched_ode")

    egroup: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in euler:
        egroup[(row["method"], int(row["K"]))].append(row)
    fig, ax = base_axes()
    for method, color, marker in (("CF", "#b2182b", "o"), ("oracle_numerical_reference", "#1b7837", "s")):
        means = []
        for steps in (10, 20, 40, 80, 160):
            vals = [float(r["discretization"]) for r in egroup[(method, steps)]]
            means.append(float(np.mean(vals)))
            ax.scatter([steps] * len(vals), vals, color=color, s=9, alpha=0.22, edgecolors="none")
        label = "CF" if method == "CF" else "oracle numerical reference"
        ax.plot((10, 20, 40, 80, 160), means, color=color, marker=marker, label=label)
    k = np.asarray([10, 20, 40, 80, 160], dtype=float)
    cf_last = float(np.mean([float(r["discretization"]) for r in egroup[("CF", 160)]]))
    ax.plot(k, cf_last * (160 / k) ** 2, color="#666666", linestyle="--", linewidth=1.0, label="$K^{-2}$ reference")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Euler steps $K$",
        ylabel="pure Euler-to-exact MSE\nper token",
    )
    ax.set_xticks([10, 20, 40, 80, 160], ["10", "20", "40", "80", "160"])
    ax.legend(frameon=False)
    save_figure(fig, "fig_cf_euler_discretization")

    plot_particle_dimensions(dim)

    models = ["proposed", "full_rank", "no_condition", "independent_token"]
    labels = ["low rank\n+ condition", "full rank\n+ condition", "no\ncondition", "independent\ntoken"]
    fig, ax = base_axes(figsize=(5.2, 3.35))
    x = np.arange(len(models))
    means = [ablation[(model,)][0] for model in models]
    sds = [ablation[(model,)][1] for model in models]
    ax.errorbar(x, means, yerr=sds, fmt="o", color="#b2182b", capsize=4, label="mean $\\pm$ sample SD")
    for index, model in enumerate(models):
        vals = ablation[(model,)][2]
        jitter = np.linspace(-0.06, 0.06, len(vals))
        ax.scatter(index + jitter, vals, color="#2166ac", s=13, alpha=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("field MSE per token")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False)
    save_figure(fig, "fig_condition_ablation")

    report.update(figures_pdf=6, output_dir=str(FIGURES))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Experiment 1 v3: Synthetic fixed-time drift estimation.

This version is designed to produce a more stable rate plot.

Main changes:
1. For each (M, repeat), train several optimizer restarts.
2. Select the restart with the smallest validation loss.
3. Save all restart results, not only the selected one.
4. Save mean / median / trimmed-mean summaries.
5. Save slope fits for all M and for the tail M >= fit_min_M.

The clean test metric is still

    (1/N) E ||B_hat(X,Y) - B^*(X,Y)||_F^2.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from csi_exp.models import LowRankPairwiseDrift
from csi_exp.synthetic import (
    SyntheticConfig,
    TruePairwiseDrift,
    make_fixed_time_dataset,
    train_val_split,
)
from csi_exp.train import evaluate_drift_model, train_drift_model
from csi_exp.utils import (
    count_parameters,
    fit_loglog_slope,
    get_device,
    make_run_dir,
    save_json,
    set_seed,
    state_dict_sha256,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic fixed-time drift estimation v3")

    # Device.
    # --device auto means: use CUDA if torch.cuda.is_available(), otherwise CPU.
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output_dir", type=str, default="outputs")

    # Sample sizes.
    p.add_argument("--M_list", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000, 16000])

    # Repeats control statistical randomness.
    # Restarts control optimization randomness.
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--restarts", type=int, default=2)

    # Training settings.
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)

    # Synthetic model dimensions.
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--num_conditions", type=int, default=4)

    # Signal and noise settings.
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--true_scale", type=float, default=1.0)
    p.add_argument("--drift_scale", type=float, default=3.0)
    p.add_argument("--hidden", type=int, default=128)

    # By default, use the fixed true condition embedding e(X), matching the theory.
    # Turn this on only for an ablation.
    p.add_argument("--learn_condition_embedding", action="store_true")

    # Evaluation and plotting.
    p.add_argument("--test_size", type=int, default=30000)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--fit_min_M", type=int, default=2000)
    p.add_argument("--trim_fraction", type=float, default=0.2)

    # Separate seeds prevent a later pointwise rerun from silently changing
    # the ground-truth system. --seed is kept only as a legacy convenience.
    p.add_argument("--truth_seed", type=int, default=123)
    p.add_argument("--data_seed_base", type=int, default=1000000)
    p.add_argument("--init_seed_base", type=int, default=2000000)
    p.add_argument("--test_seed", type=int, default=3000000)
    p.add_argument("--seed", type=int, default=None, help="Legacy: derive all seeds from one integer.")
    p.add_argument("--save_checkpoints", action="store_true")

    args = p.parse_args()
    if args.seed is not None:
        args.truth_seed = args.seed
        args.data_seed_base = args.seed + 1_000_000
        args.init_seed_base = args.seed + 2_000_000
        args.test_seed = args.seed + 3_000_000
    return args


def trimmed_mean(values: Iterable[float], trim_fraction: float) -> float:
    """Two-sided trimmed mean.

    If repeats are too few, this falls back to the ordinary mean.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")

    arr.sort()
    k = int(math.floor(trim_fraction * arr.size))
    if 2 * k >= arr.size:
        return float(arr.mean())
    return float(arr[k : arr.size - k].mean())


def standard_error(values: Iterable[float]) -> float:
    """Standard error over independent repeats."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def make_summary(raw: pd.DataFrame, trim_fraction: float) -> pd.DataFrame:
    """Aggregate selected runs over repeats for each M."""
    rows = []

    for M, g in raw.groupby("M"):
        test_values = g["test_mse"].to_numpy(dtype=np.float64)

        rows.append(
            {
                "M": int(M),
                "test_mse_mean": float(np.mean(test_values)),
                "test_mse_std": float(np.std(test_values, ddof=1)) if len(test_values) > 1 else float("nan"),
                "test_mse_se": standard_error(test_values),
                "test_mse_median": float(np.median(test_values)),
                "test_mse_trimmed_mean": trimmed_mean(test_values, trim_fraction),
                "train_mse_mean": float(g["train_mse"].mean()),
                "val_mse_mean": float(g["val_mse"].mean()),
                "num_selected_runs": int(len(g)),
            }
        )

    return pd.DataFrame(rows).sort_values("M").reset_index(drop=True)


def fit_many_slopes(summary: pd.DataFrame, beta: float, fit_min_M: int) -> dict:
    """Fit log-log slopes for mean, median, and trimmed mean."""
    theory_slope = -2.0 * beta / (2.0 * beta + 1.0)

    out: dict[str, object] = {
        "theory_slope": theory_slope,
        "fit_min_M": fit_min_M,
    }

    metrics = [
        ("mean", "test_mse_mean"),
        ("median", "test_mse_median"),
        ("trimmed_mean", "test_mse_trimmed_mean"),
    ]

    for name, col in metrics:
        slope, intercept = fit_loglog_slope(summary["M"], summary[col])
        out[f"{name}_all"] = {
            "slope": slope,
            "intercept": intercept,
        }

        tail = summary[summary["M"] >= fit_min_M]
        slope_tail, intercept_tail = fit_loglog_slope(tail["M"], tail[col])
        out[f"{name}_tail"] = {
            "slope": slope_tail,
            "intercept": intercept_tail,
        }

    return out


def plot_rate(
    summary: pd.DataFrame,
    path: Path,
    metric_col: str,
    ylabel: str,
    theory_slope: float,
    title: str,
) -> None:
    """Plot one log-log rate curve."""
    M = summary["M"].to_numpy(dtype=np.float64)
    y = summary[metric_col].to_numpy(dtype=np.float64)

    plt.figure(figsize=(7.5, 5.2))

    if metric_col == "test_mse_mean":
        # For the mean curve, use standard error instead of standard deviation.
        # SE makes the error bar represent uncertainty of the mean.
        yerr = summary["test_mse_se"].to_numpy(dtype=np.float64)
        plt.errorbar(M, y, yerr=yerr, marker="o", linewidth=2, capsize=4, label="mean MSE ± SE")
    else:
        plt.plot(M, y, marker="o", linewidth=2, label=ylabel)

    # Reference theoretical slope, anchored at the first point.
    ref = y[0] * (M / M[0]) ** theory_slope
    plt.plot(M, ref, linestyle="--", linewidth=2, label=f"theory slope {theory_slope:.3f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("M")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    if device.type == "cuda":
        # Speed hint for modern NVIDIA GPUs. Does not change the experiment definition.
        torch.set_float32_matmul_precision("high")

    run_dir = make_run_dir(args.output_dir, "synthetic_fixed_time")
    save_json(vars(args), run_dir / "config.json")
    print(f"[run_dir] {run_dir}")

    cfg = SyntheticConfig(
        d=args.d,
        N=args.N,
        q=args.q,
        rank=args.rank,
        num_conditions=args.num_conditions,
        noise_std=args.noise_std,
        true_scale=args.true_scale,
        drift_scale=args.drift_scale,
        seed=args.truth_seed,
    )

    # Fix one true data-generating system for the whole experiment.
    set_seed(args.truth_seed)
    true_drift = TruePairwiseDrift(cfg, device=device)

    true_state = true_drift.state_dict()
    true_hash = state_dict_sha256(true_state)
    torch.save(true_state, run_dir / "true_drift_parameters.pt")
    save_json(
        {"truth_seed": args.truth_seed, "ground_truth_sha256": true_hash},
        run_dir / "ground_truth_manifest.json",
    )
    print(f"[ground_truth_sha256] {true_hash}")

    noise_floor = args.d * (args.noise_std ** 2)
    save_json({"normalized_label_noise_floor": noise_floor}, run_dir / "noise_floor.json")
    print(f"[noise_floor] d * noise_std^2 = {noise_floor:.6e}")

    # Reuse one large clean test set for all M. This reduces test-set noise.
    x_test, y_test, _, target_test = make_fixed_time_dataset(
        true_drift,
        args.test_size,
        noise_std=0.0,
        device=device,
        seed=args.test_seed,
    )

    selected_rows = []
    restart_rows = []

    for M in args.M_list:
        for rep in range(args.repeats):
            # Data seed controls the sample set.
            data_seed = args.data_seed_base + 100_000 * rep + M
            set_seed(data_seed)

            x, y, v_noisy, _ = make_fixed_time_dataset(
                true_drift,
                M,
                noise_std=args.noise_std,
                device=device,
                seed=data_seed,
            )

            train_tensors, val_tensors = train_val_split(x, y, v_noisy, val_fraction=0.1)

            best_row = None
            best_model_state = None

            for restart in range(args.restarts):
                # Init seed controls random initialization.
                init_seed = args.init_seed_base + 10_000_000 * rep + 10_000 * M + restart
                set_seed(init_seed)

                model = LowRankPairwiseDrift(
                    d=args.d,
                    N=args.N,
                    q=args.q,
                    rank=args.rank,
                    num_conditions=args.num_conditions,
                    hidden=args.hidden,
                    use_condition=True,
                    fixed_condition_embedding=None
                    if args.learn_condition_embedding
                    else true_drift.condition_embedding.detach().cpu(),
                ).to(device)

                print(f"[M={M} rep={rep} restart={restart}] params={count_parameters(model)}")

                history = train_drift_model(
                    model,
                    train_tensors=train_tensors,
                    val_tensors=val_tensors,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    verbose=True,
                )

                train_mse = evaluate_drift_model(model, train_tensors, batch_size=args.batch_size)
                val_mse = evaluate_drift_model(model, val_tensors, batch_size=args.batch_size)
                test_mse = evaluate_drift_model(model, (x_test, y_test, target_test), batch_size=args.batch_size)

                row = {
                    "M": M,
                    "repeat": rep,
                    "restart": restart,
                    "truth_seed": args.truth_seed,
                    "ground_truth_sha256": true_hash,
                    "data_seed": data_seed,
                    "init_seed": init_seed,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "test_mse": test_mse,
                    "num_params": count_parameters(model),
                    "final_train_loss": history["train_loss"][-1],
                    "final_val_loss": history["val_loss"][-1],
                    "selected": False,
                }

                restart_rows.append(row)
                pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)
                save_json(history, run_dir / f"history_M{M}_rep{rep}_restart{restart}.json")

                if best_row is None or val_mse < best_row["val_mse"]:
                    best_row = row.copy()
                    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                print(
                    f"[M={M} rep={rep} restart={restart}] "
                    f"train={train_mse:.6e} val={val_mse:.6e} test={test_mse:.6e}"
                )

            assert best_row is not None

            best_row["selected"] = True
            selected_rows.append(best_row)

            # Mark selected restart in all-restarts table.
            for r in restart_rows:
                if r["M"] == M and r["repeat"] == rep and r["restart"] == best_row["restart"]:
                    r["selected"] = True

            pd.DataFrame(selected_rows).to_csv(run_dir / "results_raw.csv", index=False)
            pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)

            if args.save_checkpoints and best_model_state is not None:
                torch.save(
                    {
                        "model_state_dict": best_model_state,
                        "extra": {
                            "M": M,
                            "repeat": rep,
                            "best_restart": best_row["restart"],
                            "test_mse": best_row["test_mse"],
                            "val_mse": best_row["val_mse"],
                        },
                    },
                    run_dir / f"model_M{M}_rep{rep}_best.pt",
                )

            print(
                f"[selected M={M} rep={rep}] restart={best_row['restart']} "
                f"val={best_row['val_mse']:.6e} test={best_row['test_mse']:.6e}"
            )

    raw = pd.DataFrame(selected_rows)
    summary = make_summary(raw, trim_fraction=args.trim_fraction)
    summary.to_csv(run_dir / "summary.csv", index=False)

    slope_info = fit_many_slopes(summary, beta=args.beta, fit_min_M=args.fit_min_M)
    save_json(slope_info, run_dir / "fit_slope.json")

    theory_slope = slope_info["theory_slope"]

    plot_rate(
        summary,
        run_dir / "synthetic_fixed_time_rate_mean.png",
        metric_col="test_mse_mean",
        ylabel="test_mse_mean",
        theory_slope=theory_slope,
        title="Synthetic fixed-time drift estimation: mean",
    )

    plot_rate(
        summary,
        run_dir / "synthetic_fixed_time_rate_median.png",
        metric_col="test_mse_median",
        ylabel="test_mse_median",
        theory_slope=theory_slope,
        title="Synthetic fixed-time drift estimation: median",
    )

    # Keep old filename as the mean plot.
    plot_rate(
        summary,
        run_dir / "synthetic_fixed_time_rate.png",
        metric_col="test_mse_mean",
        ylabel="test_mse_mean",
        theory_slope=theory_slope,
        title="Synthetic fixed-time drift estimation",
    )


    # Vector PDF versions for LaTeX.
    plot_rate(
        summary,
        run_dir / "synthetic_fixed_time_rate_paper.pdf",
        metric_col="test_mse_mean",
        ylabel="Test MSE",
        theory_slope=theory_slope,
        title="Fixed-time drift estimation",
    )

    print(f"[done] run_dir={run_dir}")
    print(f"[slope mean all] {slope_info['mean_all']['slope']:.4f}")
    print(f"[slope median all] {slope_info['median_all']['slope']:.4f}")
    print(f"[slope trimmed all] {slope_info['trimmed_mean_all']['slope']:.4f}")
    print(f"[theory] {theory_slope:.4f}")


if __name__ == "__main__":
    main()

"""Experiment 2 v2: Dimension-free exponent check.

Goal
----
We vary the per-token latent dimension d and check whether the empirical
log-log slope with respect to M remains close to the theoretical exponent

    -2 beta / (2 beta + 1).

Important:
The dimension-free statement is about the exponent in M, not about identical
absolute MSE values for different d. The constants can change with d.

This version is more stable than the initial quick version:
1. Uses fixed condition embedding e(X), matching the theory.
2. Uses small label noise.
3. Runs multiple optimization restarts for each (d, M, repeat).
4. Selects the restart with the smallest validation loss.
5. Saves all restarts, selected runs, summaries, and slope fits.
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
    p = argparse.ArgumentParser(description="Dimension-free synthetic rate experiment v2")

    # Device.
    # auto = use CUDA if torch.cuda.is_available(), otherwise CPU.
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output_dir", type=str, default="outputs")

    # Experiment grid.
    # d is the per-token latent dimension. Total latent dimension is d * N.
    p.add_argument("--d_list", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--M_list", type=int, nargs="+", default=[2000, 4000, 8000, 16000])

    # Repeats = data randomness.
    # Restarts = optimization randomness.
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--restarts", type=int, default=2)

    # Training hyperparameters.
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--verbose", action="store_true")

    # Model and data dimensions.
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--num_conditions", type=int, default=4)

    # Synthetic signal/noise settings.
    # We use smaller noise than the quick version so that dimension trends are visible.
    p.add_argument("--noise_std", type=float, default=0.005)
    p.add_argument("--true_scale", type=float, default=1.0)
    p.add_argument("--drift_scale", type=float, default=3.0)

    # Model capacity.
    p.add_argument("--hidden", type=int, default=128)

    # By default, use the true fixed e(X), matching the theory.
    # Turning this on makes the experiment harder and is better used as an ablation.
    p.add_argument("--learn_condition_embedding", action="store_true")

    # Evaluation.
    p.add_argument("--test_size", type=int, default=30000)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--fit_min_M", type=int, default=2000)
    p.add_argument("--trim_fraction", type=float, default=0.2)

    # Separate truth/data/init/evaluation seeds. This is the main safeguard
    # against the previous d=64, M=2000 point-replacement problem.
    p.add_argument("--truth_seed", type=int, default=123)
    p.add_argument("--data_seed_base", type=int, default=1000000)
    p.add_argument("--init_seed_base", type=int, default=2000000)
    p.add_argument("--test_seed_base", type=int, default=3000000)
    p.add_argument("--seed", type=int, default=None, help="Legacy: derive all seeds from one integer.")
    p.add_argument("--save_checkpoints", action="store_true")

    args = p.parse_args()
    if args.seed is not None:
        args.truth_seed = args.seed
        args.data_seed_base = args.seed + 1_000_000
        args.init_seed_base = args.seed + 2_000_000
        args.test_seed_base = args.seed + 3_000_000
    return args


def trimmed_mean(values: Iterable[float], trim_fraction: float) -> float:
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
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def make_summary(raw: pd.DataFrame, trim_fraction: float) -> pd.DataFrame:
    rows = []

    for (d, M), g in raw.groupby(["d", "M"]):
        vals = g["test_mse"].to_numpy(dtype=np.float64)

        rows.append(
            {
                "d": int(d),
                "M": int(M),
                "total_latent_dim_dN": int(d) * int(g["N"].iloc[0]),
                "test_mse_mean": float(vals.mean()),
                "test_mse_std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                "test_mse_se": standard_error(vals),
                "test_mse_median": float(np.median(vals)),
                "test_mse_trimmed_mean": trimmed_mean(vals, trim_fraction),
                "train_mse_mean": float(g["train_mse"].mean()),
                "val_mse_mean": float(g["val_mse"].mean()),
                "num_selected_runs": int(len(g)),
                "num_params_mean": float(g["num_params"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["d", "M"]).reset_index(drop=True)


def fit_slopes_by_d(summary: pd.DataFrame, beta: float, fit_min_M: int) -> dict:
    theory_slope = -2.0 * beta / (2.0 * beta + 1.0)

    out: dict[str, object] = {
        "theory_slope": theory_slope,
        "fit_min_M": fit_min_M,
        "note": "Slopes are fitted in log-log scale for each d. Dimension-free exponent means these slopes should stay close to the theory slope; constants may vary with d.",
    }

    metrics = [
        ("mean", "test_mse_mean"),
        ("median", "test_mse_median"),
        ("trimmed_mean", "test_mse_trimmed_mean"),
    ]

    for d, g in summary.groupby("d"):
        d_key = str(int(d))
        out[d_key] = {}

        for name, col in metrics:
            slope_all, intercept_all = fit_loglog_slope(g["M"], g[col])
            tail = g[g["M"] >= fit_min_M]
            slope_tail, intercept_tail = fit_loglog_slope(tail["M"], tail[col])

            out[d_key][f"{name}_all"] = {
                "slope": slope_all,
                "intercept": intercept_all,
            }
            out[d_key][f"{name}_tail"] = {
                "slope": slope_tail,
                "intercept": intercept_tail,
            }

    return out


def plot_rate_curves(summary: pd.DataFrame, path: Path, theory_slope: float) -> None:
    """Plot mean MSE curves for different d."""
    plt.figure(figsize=(8.0, 5.4))

    for d, g in summary.groupby("d"):
        g = g.sort_values("M")
        M = g["M"].to_numpy(dtype=np.float64)
        y = g["test_mse_mean"].to_numpy(dtype=np.float64)
        yerr = g["test_mse_se"].to_numpy(dtype=np.float64)

        plt.errorbar(
            M,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            label=f"d={int(d)}",
        )

    # Add one reference line with theoretical slope.
    # Place it by least-squares vertical alignment to all mean points.
    all_M = summary["M"].to_numpy(dtype=np.float64)
    all_y = summary["test_mse_mean"].to_numpy(dtype=np.float64)
    mask = (all_M > 0) & (all_y > 0)
    intercept_theory = np.mean(np.log(all_y[mask]) - theory_slope * np.log(all_M[mask]))

    ref_M = np.array(sorted(summary["M"].unique()), dtype=np.float64)
    ref_y = np.exp(intercept_theory) * ref_M ** theory_slope

    plt.plot(
        ref_M,
        ref_y,
        linestyle="--",
        linewidth=2,
        label=f"theory slope {theory_slope:.3f}",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sample size M")
    plt.ylabel("normalized clean drift MSE")
    plt.title("Dimension-free exponent check")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_slopes(summary: pd.DataFrame, slope_info: dict, path: Path) -> None:
    """Plot fitted slopes as a function of d."""
    ds = []
    slopes_mean = []
    slopes_median = []
    theory = float(slope_info["theory_slope"])

    for d in sorted(summary["d"].unique()):
        d_key = str(int(d))
        ds.append(int(d))
        slopes_mean.append(float(slope_info[d_key]["mean_tail"]["slope"]))
        slopes_median.append(float(slope_info[d_key]["median_tail"]["slope"]))

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(ds, slopes_mean, marker="o", linewidth=2, label="mean slope")
    plt.plot(ds, slopes_median, marker="s", linewidth=2, label="median slope")
    plt.axhline(theory, linestyle="--", linewidth=2, label=f"theory {theory:.3f}")

    plt.xscale("log", base=2)
    plt.xlabel("per-token latent dimension d")
    plt.ylabel("fitted log-log slope")
    plt.title("Fitted exponent versus latent dimension")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_mean_slope_scatter(summary: pd.DataFrame, slope_info: dict, path: Path) -> None:
    ds = sorted(int(d) for d in summary["d"].unique())
    slopes = [float(slope_info[str(d)]["mean_tail"]["slope"]) for d in ds]
    theory = float(slope_info["theory_slope"])

    plt.figure(figsize=(6.4, 4.6))
    plt.scatter(ds, slopes, s=78, label="Mean fitted slope", zorder=3)
    for d, slope in zip(ds, slopes):
        plt.text(d, slope + 0.025, f"$d={d}$", ha="center", va="bottom", fontsize=9)
    plt.axhline(theory, linestyle="--", linewidth=2, label=f"Theory: {theory:.2f}")
    plt.xscale("log", base=2)
    plt.xticks(ds, [str(d) for d in ds])
    plt.xlabel("Token dimension $d$")
    plt.ylabel("Fitted log--log slope over $M$")
    plt.title("Dimension-free exponent across latent dimensions")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = make_run_dir(args.output_dir, "synthetic_dimension_free")
    save_json(vars(args), run_dir / "config.json")
    print(f"[run_dir] {run_dir}")

    selected_rows = []
    restart_rows = []

    for d in args.d_list:
        print(f"\n========== d={d}, total latent dimension dN={d * args.N} ==========")

        cfg = SyntheticConfig(
            d=d,
            N=args.N,
            q=args.q,
            rank=args.rank,
            num_conditions=args.num_conditions,
            noise_std=args.noise_std,
            true_scale=args.true_scale,
            drift_scale=args.drift_scale,
            seed=args.truth_seed + d,
        )

        d_truth_seed = args.truth_seed + d
        set_seed(d_truth_seed)
        true_drift = TruePairwiseDrift(cfg, device=device)
        true_state = true_drift.state_dict()
        true_hash = state_dict_sha256(true_state)
        torch.save(true_state, run_dir / f"true_drift_d{d}.pt")
        save_json(
            {"d": d, "truth_seed": d_truth_seed, "ground_truth_sha256": true_hash},
            run_dir / f"ground_truth_manifest_d{d}.json",
        )
        print(f"[d={d}] ground_truth_sha256={true_hash}")

        # One clean test set per d, shared over all M/repeats/restarts.
        x_test, y_test, _, target_test = make_fixed_time_dataset(
            true_drift,
            args.test_size,
            noise_std=0.0,
            device=device,
            seed=args.test_seed_base + d,
        )

        noise_floor = d * (args.noise_std ** 2)
        print(f"[d={d}] normalized label-noise floor d * sigma^2 = {noise_floor:.6e}")

        for M in args.M_list:
            for rep in range(args.repeats):
                data_seed = args.data_seed_base + d * 10_000_000 + rep * 100_000 + M
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
                best_state = None

                for restart in range(args.restarts):
                    init_seed = args.init_seed_base + d * 10_000_000 + rep * 100_000 + M * 10 + restart
                    set_seed(init_seed)

                    model = LowRankPairwiseDrift(
                        d=d,
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

                    print(
                        f"[d={d} M={M} rep={rep} restart={restart}] "
                        f"params={count_parameters(model)}"
                    )

                    history = train_drift_model(
                        model,
                        train_tensors=train_tensors,
                        val_tensors=val_tensors,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        verbose=args.verbose,
                    )

                    train_mse = evaluate_drift_model(model, train_tensors, batch_size=args.batch_size)
                    val_mse = evaluate_drift_model(model, val_tensors, batch_size=args.batch_size)
                    test_mse = evaluate_drift_model(
                        model,
                        (x_test, y_test, target_test),
                        batch_size=args.batch_size,
                    )

                    row = {
                        "d": d,
                        "N": args.N,
                        "total_latent_dim_dN": d * args.N,
                        "M": M,
                        "repeat": rep,
                        "restart": restart,
                        "truth_seed": d_truth_seed,
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

                    if best_row is None or val_mse < best_row["val_mse"]:
                        best_row = row.copy()
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                    print(
                        f"[d={d} M={M} rep={rep} restart={restart}] "
                        f"train={train_mse:.6e} val={val_mse:.6e} test={test_mse:.6e}"
                    )

                assert best_row is not None
                best_row["selected"] = True
                selected_rows.append(best_row)

                for r in restart_rows:
                    if (
                        r["d"] == d
                        and r["M"] == M
                        and r["repeat"] == rep
                        and r["restart"] == best_row["restart"]
                    ):
                        r["selected"] = True

                pd.DataFrame(selected_rows).to_csv(run_dir / "results_raw.csv", index=False)
                pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)

                if args.save_checkpoints and best_state is not None:
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "extra": {
                                "d": d,
                                "M": M,
                                "repeat": rep,
                                "best_restart": best_row["restart"],
                                "val_mse": best_row["val_mse"],
                                "test_mse": best_row["test_mse"],
                            },
                        },
                        run_dir / f"model_d{d}_M{M}_rep{rep}_best.pt",
                    )

                print(
                    f"[selected d={d} M={M} rep={rep}] "
                    f"restart={best_row['restart']} "
                    f"val={best_row['val_mse']:.6e} "
                    f"test={best_row['test_mse']:.6e}"
                )

    raw = pd.DataFrame(selected_rows)
    summary = make_summary(raw, trim_fraction=args.trim_fraction)
    summary.to_csv(run_dir / "summary.csv", index=False)

    slope_info = fit_slopes_by_d(summary, beta=args.beta, fit_min_M=args.fit_min_M)
    save_json(slope_info, run_dir / "fit_slopes_by_d.json")

    theory_slope = float(slope_info["theory_slope"])

    plot_rate_curves(summary, run_dir / "dimension_free_rate.png", theory_slope=theory_slope)
    plot_rate_curves(summary, run_dir / "dimension_free_rate_paper.pdf", theory_slope=theory_slope)
    plot_slopes(summary, slope_info, run_dir / "dimension_free_slopes_by_d.png")
    plot_slopes(summary, slope_info, run_dir / "dimension_free_slopes_by_d.pdf")
    plot_mean_slope_scatter(summary, slope_info, run_dir / "dimension_free_slope_mean_scatter.pdf")

    print(f"[done] run_dir={run_dir}")
    print("[slopes]")
    for d in sorted(summary["d"].unique()):
        d_key = str(int(d))
        print(
            f"d={int(d):>3d} "
            f"mean_tail={slope_info[d_key]['mean_tail']['slope']:.4f} "
            f"median_tail={slope_info[d_key]['median_tail']['slope']:.4f}"
        )
    print(f"[theory] {theory_slope:.4f}")


if __name__ == "__main__":
    main()

"""Experiment 4 v2: path-matched ODE endpoint error curve.

The full M-grid is run in one directory. A fixed ground-truth drift and one
shared endpoint evaluation coupling are used for every M and repeat. All points
receive the same number of validation-selected optimizer restarts.

Unlike the original version, regression states are sampled along the true ODE
path: Z_0 ~ N(0, I), a random path time is selected, and the true ODE is evolved
to that time before constructing the regression target B^*(X, Z_t) + noise.
This aligns the regression design with the state distribution used by the ODE
perturbation argument.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from csi_exp.dynamics import ode_euler
from csi_exp.metrics import mmd_rbf, sliced_wasserstein2
from csi_exp.models import LowRankPairwiseDrift
from csi_exp.synthetic import SyntheticConfig, TruePairwiseDrift, train_val_split
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
    p = argparse.ArgumentParser(description="Path-matched ODE generation error experiment")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--M_list", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000, 16000, 32000])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--num_conditions", type=int, default=4)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--true_scale", type=float, default=1.0)
    p.add_argument("--drift_scale", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=8192)
    p.add_argument("--ode_steps", type=int, default=200)
    p.add_argument("--path_steps", type=int, default=200)
    p.add_argument("--num_projections", type=int, default=128)
    p.add_argument("--mmd_max_points", type=int, default=2048)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--fit_min_M", type=int, default=1000)

    p.add_argument("--truth_seed", type=int, default=123)
    p.add_argument("--data_seed_base", type=int, default=1000000)
    p.add_argument("--init_seed_base", type=int, default=2000000)
    p.add_argument("--eval_seed", type=int, default=3000000)
    p.add_argument("--seed", type=int, default=None, help="Legacy: derive all seeds from one integer.")
    p.add_argument("--save_checkpoints", action="store_true")

    args = p.parse_args()
    if args.seed is not None:
        args.truth_seed = args.seed
        args.data_seed_base = args.seed + 1_000_000
        args.init_seed_base = args.seed + 2_000_000
        args.eval_seed = args.seed + 3_000_000
    return args

@torch.no_grad()
def make_path_time_dataset(
    true_drift: TruePairwiseDrift,
    num_samples: int,
    noise_std: float,
    device: torch.device,
    seed: int,
    path_steps: int,
):
    """
    Generate regression samples on the true ODE path.

    Sample X and Z_0 ~ N(0,I), choose a random time t in [0,1],
    evolve the true ODE from Z_0 to Z_t, and use

        B^*(X, Z_t) + noise

    as the regression target.

    This matches the state distribution used in the ODE
    perturbation argument more closely than independent Gaussian Y.
    """
    if path_steps <= 0:
        raise ValueError("path_steps must be a positive integer.")

    cfg = true_drift.cfg

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Condition and initial state.
    x = torch.randint(
        0,
        cfg.num_conditions,
        (num_samples,),
        generator=gen,
        device=device,
    )

    z = torch.randn(
        num_samples,
        cfg.N,
        cfg.d,
        generator=gen,
        device=device,
    )

    # Each observation is assigned a random point on the ODE path.
    # step_index = 0 corresponds to t=0,
    # step_index = path_steps corresponds to t=1.
    step_index = torch.randint(
        0,
        path_steps + 1,
        (num_samples,),
        generator=gen,
        device=device,
    )

    dt = 1.0 / path_steps

    # Vectorized Euler evolution.
    # A sample stops evolving once its assigned time is reached.
    for k in range(path_steps):
        active = (step_index > k).to(z.dtype).view(-1, 1, 1)
        drift = true_drift.drift(x, z)
        z = z + active * drift * dt

    y_t = z

    target_clean = true_drift.drift(x, y_t)

    noise = noise_std * torch.randn(
        target_clean.shape,
        generator=gen,
        device=device,
    )
    target_noisy = target_clean + noise

    return x, y_t, target_noisy, target_clean

def trimmed_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) >= 5:
        values = np.sort(values)[1:-1]
    return float(np.mean(values))


def make_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for M, g in raw.groupby("M"):
        c = g["coupled_endpoint_mse"].to_numpy(float)
        s = g["sliced_w2_sq"].to_numpy(float)
        m = g["mmd_rbf"].to_numpy(float)
        rows.append(
            {
                "M": int(M),
                "coupled_endpoint_mse_mean": float(np.mean(c)),
                "coupled_endpoint_mse_std": float(np.std(c, ddof=1)) if len(c) > 1 else float("nan"),
                "coupled_endpoint_mse_median": float(np.median(c)),
                "coupled_endpoint_mse_trimmed": trimmed_mean(c),
                "sliced_w2_sq_mean": float(np.mean(s)),
                "sliced_w2_sq_std": float(np.std(s, ddof=1)) if len(s) > 1 else float("nan"),
                "sliced_w2_sq_median": float(np.median(s)),
                "sliced_w2_sq_trimmed": trimmed_mean(s),
                "mmd_mean": float(np.mean(m)),
                "mmd_median": float(np.median(m)),
                "mmd_trimmed": trimmed_mean(m),
                "num_repeats": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("M").reset_index(drop=True)


def fit_metric(summary: pd.DataFrame, col: str, fit_min_M: int | None):
    data = summary if fit_min_M is None else summary[summary["M"] >= fit_min_M]
    return fit_loglog_slope(data["M"], data[col])


def save_plot(summary: pd.DataFrame, fit_info: dict, run_dir: Path, fit_min_M: int) -> None:
    M = summary["M"].to_numpy(float)
    y = summary["coupled_endpoint_mse_median"].to_numpy(float)
    yerr = summary["coupled_endpoint_mse_std"].to_numpy(float)
    slope = fit_info["fitted_slope_coupled_endpoint_median_M_ge_fit_min"]
    intercept = fit_info["intercept_coupled_endpoint_median_M_ge_fit_min"]
    theory = fit_info["theory_slope"]

    fit_line = np.exp(intercept) * M ** slope
    anchor = np.exp(np.mean(np.log(y) - theory * np.log(M)))
    theory_line = anchor * M ** theory

    plt.figure(figsize=(6.5, 4.7))
    plt.plot(M, y, marker="o", linewidth=2, label="Median endpoint MSE")
    plt.plot(M[M >= fit_min_M], fit_line[M >= fit_min_M], linestyle="-.", linewidth=1.8, label=f"Empirical fit: {slope:.2f}")
    plt.plot(M, theory_line, linestyle="--", linewidth=1.8, label=f"Reference slope: {theory:.2f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of training samples $M$")
    plt.ylabel("Coupled endpoint MSE")
    plt.title("ODE distribution convergence (path-matched)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "ode_generation_paper_clean.pdf", bbox_inches="tight")
    plt.savefig(run_dir / "ode_generation_paper_clean.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.data_protocol = "true_ode_path_uniform_grid_time"
    device = get_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = make_run_dir(args.output_dir, "ode_generation_error_v2_path")
    save_json(vars(args), run_dir / "config.json")
    checkpoint_dir = run_dir / "checkpoints"
    if args.save_checkpoints:
        checkpoint_dir.mkdir(exist_ok=True)

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
    set_seed(args.truth_seed)
    true_drift = TruePairwiseDrift(cfg, device)
    true_state = true_drift.state_dict()
    true_hash = state_dict_sha256(true_state)
    torch.save(true_state, run_dir / "true_drift_parameters.pt")
    save_json(
        {"truth_seed": args.truth_seed, "ground_truth_sha256": true_hash},
        run_dir / "ground_truth_manifest.json",
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(args.eval_seed)
    x_eval = torch.randint(0, args.num_conditions, (args.num_samples,), generator=gen, device=device)
    z0 = torch.randn(args.num_samples, args.N, args.d, generator=gen, device=device)

    def true_fn(x, z, t):
        return true_drift.drift(x, z)

    with torch.no_grad():
        true_terminal = ode_euler(true_fn, x_eval, z0, args.ode_steps)

    rows: list[dict] = []
    restart_rows: list[dict] = []
    true_embedding = true_drift.condition_embedding.detach().cpu()

    for M in args.M_list:
        for rep in range(args.repeats):
            data_seed = args.data_seed_base + rep * 100_000 + M
            set_seed(data_seed)
            x, y, v_noisy, _ = make_path_time_dataset(
                true_drift=true_drift,
                num_samples=M,
                noise_std=args.noise_std,
                device=device,
                seed=data_seed,
                path_steps=args.path_steps,
            )
            train_tensors, val_tensors = train_val_split(x, y, v_noisy, val_fraction=0.1)

            best_val = float("inf")
            best_restart = None
            best_state = None

            for restart in range(args.restarts):
                init_seed = args.init_seed_base + rep * 10_000_000 + M * 10 + restart
                set_seed(init_seed)
                model = LowRankPairwiseDrift(
                    d=args.d,
                    N=args.N,
                    q=args.q,
                    rank=args.rank,
                    num_conditions=args.num_conditions,
                    hidden=args.hidden,
                    use_condition=True,
                    fixed_condition_embedding=true_embedding,
                ).to(device)
                print(f"[train] M={M} rep={rep} restart={restart} params={count_parameters(model)}")
                train_drift_model(
                    model,
                    train_tensors,
                    val_tensors,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    weight_decay=args.weight_decay,
                    verbose=True,
                )
                val_mse = evaluate_drift_model(model, val_tensors, args.batch_size)
                restart_row = {
                    "M": M,
                    "repeat": rep,
                    "restart": restart,
                    "truth_seed": args.truth_seed,
                    "ground_truth_sha256": true_hash,
                    "data_seed": data_seed,
                    "init_seed": init_seed,
                    "val_mse": val_mse,
                    "selected": False,
                }
                restart_rows.append(restart_row)
                if val_mse < best_val:
                    best_val = val_mse
                    best_restart = restart
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)
                print(f"[restart done] M={M} rep={rep} restart={restart} val={val_mse:.6e}")

            assert best_state is not None and best_restart is not None
            for rr in restart_rows:
                if rr["M"] == M and rr["repeat"] == rep and rr["restart"] == best_restart:
                    rr["selected"] = True

            model = LowRankPairwiseDrift(
                d=args.d,
                N=args.N,
                q=args.q,
                rank=args.rank,
                num_conditions=args.num_conditions,
                hidden=args.hidden,
                use_condition=True,
                fixed_condition_embedding=true_embedding,
            ).to(device)
            model.load_state_dict(best_state)
            model.eval()

            def learned_fn(xb, z, t):
                return model(xb, z)

            with torch.no_grad():
                learned_terminal = ode_euler(learned_fn, x_eval, z0, args.ode_steps)
                coupled_endpoint_mse = float(torch.mean((learned_terminal - true_terminal) ** 2).cpu())
                sw2 = float(
                    sliced_wasserstein2(
                        learned_terminal,
                        true_terminal,
                        num_projections=args.num_projections,
                        seed=args.eval_seed,
                    ).cpu()
                )
                n_mmd = min(args.mmd_max_points, learned_terminal.shape[0])
                mmd = float(mmd_rbf(learned_terminal[:n_mmd], true_terminal[:n_mmd]).cpu())

            row = {
                "M": M,
                "repeat": rep,
                "best_restart": best_restart,
                "best_val_mse": best_val,
                "truth_seed": args.truth_seed,
                "ground_truth_sha256": true_hash,
                "data_seed": data_seed,
                "eval_seed": args.eval_seed,
                "coupled_endpoint_mse": coupled_endpoint_mse,
                "sliced_w2_sq": sw2,
                "mmd_rbf": mmd,
            }
            rows.append(row)

            if args.save_checkpoints:
                torch.save(
                    {"model_state_dict": best_state, "selection": row, "config": vars(args)},
                    checkpoint_dir / f"best_model_M{M}_rep{rep}.pt",
                )

            raw = pd.DataFrame(rows)
            summary = make_summary(raw)
            raw.to_csv(run_dir / "results_raw.csv", index=False)
            pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)
            summary.to_csv(run_dir / "summary.csv", index=False)
            print(
                f"[M={M} rep={rep}] best_restart={best_restart} "
                f"CoupledMSE={coupled_endpoint_mse:.6e} SW2^2={sw2:.6e} MMD={mmd:.6e}"
            )

    raw = pd.DataFrame(rows)
    summary = make_summary(raw)
    raw.to_csv(run_dir / "results_raw.csv", index=False)
    pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)

    theory_slope = -2 * args.beta / (2 * args.beta + 1)
    fit_info = {"fit_min_M": args.fit_min_M, "theory_slope": theory_slope}
    for label, col in [
        ("mean", "coupled_endpoint_mse_mean"),
        ("median", "coupled_endpoint_mse_median"),
        ("trimmed", "coupled_endpoint_mse_trimmed"),
    ]:
        slope_all, intercept_all = fit_metric(summary, col, None)
        slope_tail, intercept_tail = fit_metric(summary, col, args.fit_min_M)
        fit_info[f"fitted_slope_coupled_endpoint_{label}_all_M"] = slope_all
        fit_info[f"intercept_coupled_endpoint_{label}_all_M"] = intercept_all
        fit_info[f"fitted_slope_coupled_endpoint_{label}_M_ge_fit_min"] = slope_tail
        fit_info[f"intercept_coupled_endpoint_{label}_M_ge_fit_min"] = intercept_tail

    save_json(fit_info, run_dir / "fit_slope.json")
    save_plot(summary, fit_info, run_dir, args.fit_min_M)
    print(summary)
    print(f"[ground_truth_sha256] {true_hash}")
    print(f"[done] run_dir={run_dir}")


if __name__ == "__main__":
    main()

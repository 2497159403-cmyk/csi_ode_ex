"""Experiment 3: fair condition/structure ablation with validation-selected restarts.

All models see exactly the same training/validation/test data for a repeat and
receive the same number of optimizer restarts. The condition-using models share
the fixed ground-truth embedding e(X), matching the synthetic rate experiments.
This avoids interpreting optimization failures of one baseline as a structural
failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from csi_exp.models import IndependentTokenDrift, LowRankPairwiseDrift
from csi_exp.synthetic import SyntheticConfig, TruePairwiseDrift, make_fixed_time_dataset, train_val_split
from csi_exp.train import evaluate_drift_model, train_drift_model
from csi_exp.utils import (
    count_parameters,
    get_device,
    make_run_dir,
    save_json,
    set_seed,
    state_dict_sha256,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair condition/structure ablation")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--M", type=int, default=8000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--num_conditions", type=int, default=4)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--true_scale", type=float, default=1.0)
    p.add_argument("--drift_scale", type=float, default=3.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--test_size", type=int, default=30000)

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


def build_model(name: str, args: argparse.Namespace, true_embedding: torch.Tensor, device: torch.device):
    common = dict(
        d=args.d,
        N=args.N,
        q=args.q,
        num_conditions=args.num_conditions,
        hidden=args.hidden,
    )
    if name == "proposed":
        return LowRankPairwiseDrift(
            rank=args.rank,
            use_condition=True,
            fixed_condition_embedding=true_embedding,
            **common,
        ).to(device)
    if name == "no_condition":
        return LowRankPairwiseDrift(
            rank=args.rank,
            use_condition=False,
            **common,
        ).to(device)
    if name == "independent_token":
        return IndependentTokenDrift(
            use_condition=True,
            fixed_condition_embedding=true_embedding,
            **common,
        ).to(device)
    if name == "full_rank":
        return LowRankPairwiseDrift(
            rank=args.d + args.q,
            use_condition=True,
            fixed_condition_embedding=true_embedding,
            **common,
        ).to(device)
    raise ValueError(f"Unknown model: {name}")


def make_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in raw.groupby("model"):
        vals = g["test_mse"].to_numpy(float)
        rows.append(
            {
                "model": name,
                "test_mse_mean": float(np.mean(vals)),
                "test_mse_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                "test_mse_median": float(np.median(vals)),
                "val_mse_mean": float(g["val_mse"].mean()),
                "num_params_mean": float(g["num_params"].mean()),
                "num_selected_runs": int(len(g)),
            }
        )
    order = ["proposed", "full_rank", "no_condition", "independent_token"]
    out = pd.DataFrame(rows)
    out["_order"] = out["model"].map({k: i for i, k in enumerate(order)})
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def save_plot(summary: pd.DataFrame, run_dir: Path) -> None:
    labels = {
        "proposed": "Proposed",
        "full_rank": "Full-rank",
        "no_condition": "w/o condition",
        "independent_token": "Independent-token",
    }
    x = np.arange(len(summary))
    y = summary["test_mse_mean"].to_numpy(float)
    yerr = summary["test_mse_std"].to_numpy(float)

    plt.figure(figsize=(7.2, 4.8))
    bars = plt.bar(x, y, yerr=yerr, capsize=4)
    plt.yscale("log")
    plt.xticks(x, [labels[m] for m in summary["model"]], rotation=10)
    plt.ylabel("Normalized clean drift MSE")
    plt.title("Condition and interaction ablation")
    plt.grid(True, axis="y", which="both", alpha=0.25)

    for bar, value in zip(bars, y):
        plt.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:.2e}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(run_dir / "condition_ablation_paper.pdf", bbox_inches="tight")
    plt.savefig(run_dir / "condition_ablation_paper.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = make_run_dir(args.output_dir, "synthetic_condition_ablation")
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

    x_test, y_test, _, target_test = make_fixed_time_dataset(
        true_drift,
        args.test_size,
        0.0,
        device,
        args.test_seed,
    )

    selected_rows: list[dict] = []
    restart_rows: list[dict] = []
    model_names = ["proposed", "full_rank", "no_condition", "independent_token"]
    true_embedding = true_drift.condition_embedding.detach().cpu()

    for rep in range(args.repeats):
        data_seed = args.data_seed_base + rep * 100_000 + args.M
        set_seed(data_seed)
        x, y, v_noisy, _ = make_fixed_time_dataset(
            true_drift,
            args.M,
            args.noise_std,
            device,
            data_seed,
        )
        train_tensors, val_tensors = train_val_split(x, y, v_noisy, val_fraction=0.1)

        for model_index, name in enumerate(model_names):
            best_row = None
            best_state = None

            for restart in range(args.restarts):
                init_seed = (
                    args.init_seed_base
                    + rep * 10_000_000
                    + model_index * 100_000
                    + restart
                )
                set_seed(init_seed)
                model = build_model(name, args, true_embedding, device)
                print(
                    f"[model={name} rep={rep} restart={restart}] "
                    f"params={count_parameters(model)}"
                )

                history = train_drift_model(
                    model,
                    train_tensors,
                    val_tensors,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    weight_decay=args.weight_decay,
                    verbose=True,
                )
                train_mse = evaluate_drift_model(model, train_tensors, args.batch_size)
                val_mse = evaluate_drift_model(model, val_tensors, args.batch_size)
                test_mse = evaluate_drift_model(model, (x_test, y_test, target_test), args.batch_size)

                row = {
                    "model": name,
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

                if best_row is None or val_mse < best_row["val_mse"]:
                    best_row = row.copy()
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)
                print(
                    f"[done model={name} rep={rep} restart={restart}] "
                    f"val={val_mse:.6e} test={test_mse:.6e}"
                )

            assert best_row is not None
            best_row["selected"] = True
            selected_rows.append(best_row)
            for row in restart_rows:
                if (
                    row["model"] == name
                    and row["repeat"] == rep
                    and row["restart"] == best_row["restart"]
                ):
                    row["selected"] = True

            if args.save_checkpoints and best_state is not None:
                torch.save(
                    {"model_state_dict": best_state, "selection": best_row, "config": vars(args)},
                    checkpoint_dir / f"best_{name}_rep{rep}.pt",
                )

            pd.DataFrame(selected_rows).to_csv(run_dir / "results_raw.csv", index=False)
            pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)
            print(
                f"[selected model={name} rep={rep}] restart={best_row['restart']} "
                f"val={best_row['val_mse']:.6e} test={best_row['test_mse']:.6e}"
            )

    raw = pd.DataFrame(selected_rows)
    summary = make_summary(raw)
    raw.to_csv(run_dir / "results_raw.csv", index=False)
    pd.DataFrame(restart_rows).to_csv(run_dir / "results_all_restarts.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    save_plot(summary, run_dir)

    print(summary)
    print(f"[ground_truth_sha256] {true_hash}")
    print(f"[done] run_dir={run_dir}")


if __name__ == "__main__":
    main()

"""Training loops shared by synthetic experiments."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .metrics import normalized_token_mse


def train_drift_model(
    model: torch.nn.Module,
    train_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> Dict[str, list]:
    """Train a fixed-time drift model.

    train_tensors = (x, y, noisy_target). The validation loss is computed against
    the provided validation target. For test evaluation against the noiseless true
    drift, use evaluate_drift_model separately.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_train, y_train, v_train = train_tensors
    dataset = TensorDataset(x_train, y_train, v_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history = {"train_loss": [], "val_loss": []}
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="training", leave=False)

    for _ in iterator:
        model.train()
        total = 0.0
        count = 0
        for xb, yb, vb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, yb)
            loss = normalized_token_mse(pred, vb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        train_loss = total / max(count, 1)
        history["train_loss"].append(train_loss)

        if val_tensors is not None:
            val_loss = evaluate_drift_model(model, val_tensors)
            history["val_loss"].append(val_loss)
        else:
            history["val_loss"].append(float("nan"))

    return history


@torch.no_grad()
def evaluate_drift_model(
    model: torch.nn.Module,
    tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int = 4096,
) -> float:
    """Evaluate normalized token MSE."""
    model.eval()
    x, y, target = tensors
    dataset = TensorDataset(x, y, target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total = 0.0
    count = 0
    for xb, yb, tb in loader:
        pred = model(xb, yb)
        loss = normalized_token_mse(pred, tb)
        total += float(loss.detach().cpu()) * xb.shape[0]
        count += xb.shape[0]
    return total / max(count, 1)

"""
Training utilities for the asymmetric autoencoder.

Provides:
  - Loss functions (MSE, MAE, Huber, custom combinations)
  - EarlyStopping callback
  - train_epoch / eval_epoch helpers
  - train() — full training loop with logging and early stopping
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from autoencoder import AsymmetricAutoencoder


# ── Loss functions ───────────────────────────────────────────────────────────

LOSSES: Dict[str, nn.Module] = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    "huber": nn.HuberLoss(),
}


def get_loss_fn(name: str) -> nn.Module:
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES.keys())}")
    return LOSSES[name]


def reconstruction_loss(
    x_hat: Tensor,
    x: Tensor,
    loss_fn: nn.Module,
    latent_reg: Optional[Callable[[Tensor], Tensor]] = None,
    z: Optional[Tensor] = None,
    reg_weight: float = 0.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Computes reconstruction loss + optional latent regularisation.

    Args:
        x_hat:        Reconstructed input.
        x:            Original input.
        loss_fn:      Base loss (MSE, MAE, ...).
        latent_reg:   Function z -> scalar regularisation term (e.g. L2 on z).
        z:            Latent codes (required if latent_reg is provided).
        reg_weight:   Weight for the regularisation term.

    Returns:
        total_loss, {"recon": float, "reg": float}
    """
    recon = loss_fn(x_hat, x)
    reg = torch.tensor(0.0, device=x.device)

    if latent_reg is not None and z is not None and reg_weight > 0.0:
        reg = latent_reg(z)

    total = recon + reg_weight * reg
    return total, {"recon": recon.item(), "reg": reg.item()}


# ── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Args:
        patience:   Number of epochs without improvement before stopping.
        min_delta:  Minimum change to count as an improvement.
        mode:       'min' (lower is better) or 'max'.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best = float("inf") if mode == "min" else float("-inf")
        self._counter = 0
        self.best_state: Optional[dict] = None

    def step(self, metric: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        improved = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )
        if improved:
            self._best = metric
            self._counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self._counter += 1

        return self._counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Loads the best checkpoint back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ── Per-epoch helpers ────────────────────────────────────────────────────────

def train_epoch(
    model: AsymmetricAutoencoder,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    latent_reg: Optional[Callable] = None,
    reg_weight: float = 0.0,
    clip_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Runs one training epoch. Returns averaged metrics dict."""
    model.train()
    totals: Dict[str, float] = {"loss": 0.0, "recon": 0.0, "reg": 0.0}
    n_batches = 0

    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        optimizer.zero_grad()
        x_hat, z = model(x)
        loss, metrics = reconstruction_loss(x_hat, x, loss_fn, latent_reg, z, reg_weight)
        loss.backward()

        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        totals["loss"] += loss.item()
        totals["recon"] += metrics["recon"]
        totals["reg"] += metrics["reg"]
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model: AsymmetricAutoencoder,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    latent_reg: Optional[Callable] = None,
    reg_weight: float = 0.0,
) -> Dict[str, float]:
    """Runs one evaluation pass. Returns averaged metrics dict."""
    model.eval()
    totals: Dict[str, float] = {"loss": 0.0, "recon": 0.0, "reg": 0.0}
    n_batches = 0

    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        x_hat, z = model(x)
        loss, metrics = reconstruction_loss(x_hat, x, loss_fn, latent_reg, z, reg_weight)

        totals["loss"] += loss.item()
        totals["recon"] += metrics["recon"]
        totals["reg"] += metrics["reg"]
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ── Full training loop ───────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All hyper-parameters for the training loop in one place."""
    epochs: int = 100
    loss: str = "mse"                      # "mse" | "mae" | "huber"
    reg_weight: float = 0.0                # weight for latent regularisation
    clip_grad_norm: Optional[float] = None
    early_stopping_patience: int = 0       # 0 = disabled
    early_stopping_min_delta: float = 1e-4
    log_every: int = 10                    # print every N epochs (0 = silent)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainHistory:
    train: List[Dict[str, float]] = field(default_factory=list)
    val: List[Dict[str, float]] = field(default_factory=list)


def train(
    model: AsymmetricAutoencoder,
    train_loader: DataLoader,
    optimizer: Optimizer,
    cfg: TrainConfig,
    val_loader: Optional[DataLoader] = None,
    scheduler: Optional[_LRScheduler] = None,
    latent_reg: Optional[Callable[[Tensor], Tensor]] = None,
) -> TrainHistory:
    """
    Full training loop.

    Args:
        model:        AsymmetricAutoencoder instance.
        train_loader: DataLoader for training data.
        optimizer:    Any torch.optim optimiser.
        cfg:          TrainConfig with all hyper-parameters.
        val_loader:   Optional validation DataLoader.
        scheduler:    Optional LR scheduler (called each epoch).
        latent_reg:   Optional z -> scalar regularisation (e.g. L2 sparsity).

    Returns:
        TrainHistory with per-epoch metric dicts for train and val.

    Example:
        cfg = TrainConfig(epochs=200, loss="mse", early_stopping_patience=20)
        history = train(model, train_loader, optimizer, cfg, val_loader)
    """
    device = torch.device(cfg.device)
    model.to(device)
    loss_fn = get_loss_fn(cfg.loss)

    early_stop = (
        EarlyStopping(patience=cfg.early_stopping_patience, min_delta=cfg.early_stopping_min_delta)
        if cfg.early_stopping_patience > 0
        else None
    )

    history = TrainHistory()
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            latent_reg, cfg.reg_weight, cfg.clip_grad_norm,
        )
        history.train.append(train_metrics)

        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            val_metrics = eval_epoch(model, val_loader, loss_fn, device, latent_reg, cfg.reg_weight)
            history.val.append(val_metrics)

        if scheduler is not None:
            scheduler.step()

        if cfg.log_every > 0 and (epoch % cfg.log_every == 0 or epoch == 1):
            elapsed = time.time() - t0
            val_str = f"  val_loss={val_metrics['loss']:.6f}" if val_metrics else ""
            print(
                f"Epoch {epoch:>4}/{cfg.epochs}"
                f"  train_loss={train_metrics['loss']:.6f}"
                f"{val_str}"
                f"  [{elapsed:.1f}s]"
            )

        if early_stop is not None and val_loader is not None:
            stop = early_stop.step(val_metrics["loss"], model)
            if stop:
                print(f"Early stopping at epoch {epoch}.")
                early_stop.restore_best(model)
                break

    return history

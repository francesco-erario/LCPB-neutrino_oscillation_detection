"""
nn_model.py
===========
PyTorch feed-forward neural network for binary classification.

The module exposes:

    get_device         — auto-detect the best available device
    NeutrinoFFNN       — torch.nn.Module subclass (the network itself)
    build_and_train_nn — convenience function: builds, trains, returns
                         the trained model and a history dict compatible
                         with the visualization module.

All hyper-parameters (layer sizes, dropout, learning rate, optimizer,
batch size, epochs …) are configurable at construction / call time so
that the same code can be used both interactively and inside an Optuna
objective.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ======================================================================
# Device helper
# ======================================================================

def get_device() -> torch.device:
    """
    Return the best available device in priority order:
    CUDA → MPS (Apple Silicon) → CPU.

    Usage
    -----
    >>> device = get_device()
    >>> print(device)   # e.g. device(type='mps')
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ======================================================================
# Network definition
# ======================================================================

class NeutrinoFFNN(nn.Module):
    """
    Fully-connected binary classifier.

    Parameters
    ----------
    input_dim    : number of input features
    hidden_dims  : list of hidden-layer widths, e.g. [64, 32]
    dropout      : dropout probability applied after every hidden layer
    activation   : activation constructor (default ``nn.ReLU``)
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: Sequence[int] = (64,),
        dropout:     float = 0.0,
        activation:  type  = nn.ReLU,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        # Sigmoid is applied inside BCEWithLogitsLoss for numerical
        # stability; raw logits are returned during training.
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (shape [batch, 1])."""
        return self.net(x)

    # convenience -------------------------------------------------------
    def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Return P(signal) as a 1-D numpy array — mirrors the XGBoost API
        so that the same visualization / optimization code works.

        The input is automatically moved to whichever device the model
        lives on; the output is always returned as a CPU numpy array.
        """
        self.eval()
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        with torch.no_grad():
            logits = self.forward(x).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# ======================================================================
# Training loop
# ======================================================================

def build_and_train_nn(
    # -- data -----------------------------------------------------------
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    # -- architecture ---------------------------------------------------
    hidden_dims:  Sequence[int] = (64,),
    dropout:      float = 0.0,
    activation:   type  = nn.ReLU,
    # -- training -------------------------------------------------------
    epochs:       int   = 30,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_cls: type = torch.optim.Adam,
    # -- early stopping (patience=0 → disabled) -------------------------
    patience:     int   = 0,
    # -- device ---------------------------------------------------------
    device:       torch.device | str | None = None,
    # -- misc -----------------------------------------------------------
    random_state: int   = 42,
    verbose:      int   = 0,       # 0 = silent, 1 = per-epoch
) -> tuple[NeutrinoFFNN, dict]:
    """
    Build, train, and return a NeutrinoFFNN together with a history dict.

    Parameters
    ----------
    X_train, y_train  : training data (array-like / DataFrame / Tensor)
    X_val, y_val      : optional validation data (used for early-stopping
                        and recorded in ``history["val_loss"]`` etc.)
    hidden_dims       : hidden-layer widths
    dropout           : dropout probability
    activation        : activation class (``nn.ReLU``, ``nn.GELU`` …)
    epochs            : maximum number of training epochs
    batch_size        : mini-batch size
    lr                : learning rate
    weight_decay      : L2 regularization coefficient
    optimizer_cls     : optimizer constructor
    patience          : early-stopping patience (0 → disabled)
    device            : torch.device to train on.  Pass ``None`` to let
                        ``get_device()`` pick automatically (CUDA → MPS
                        → CPU).  You can also pass a string such as
                        ``"cpu"`` or ``"cuda:0"``.
    random_state      : seed for reproducibility
    verbose           : verbosity level (0 = silent, 1 = per-epoch)

    Returns
    -------
    model   : trained NeutrinoFFNN (in eval mode, still on ``device``)
    history : dict with keys ``"train_loss"``, ``"train_acc"``,
              ``"val_loss"``, ``"val_acc"`` — each a list over epochs.
    """

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # ── Resolve device ────────────────────────────────────────────────
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    # ── Convert to tensors (CPU first, moved to device below) ─────────
    def _to_tensor(arr):
        if isinstance(arr, torch.Tensor):
            return arr.float()
        return torch.tensor(np.asarray(arr), dtype=torch.float32)

    X_tr = _to_tensor(X_train)
    y_tr = _to_tensor(y_train).unsqueeze(-1)

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_vl = _to_tensor(X_val).to(device)
        y_vl = _to_tensor(y_val).unsqueeze(-1).to(device)

    # ── Build model and move to device ────────────────────────────────
    input_dim = X_tr.shape[1]
    model = NeutrinoFFNN(
        input_dim=input_dim,
        hidden_dims=list(hidden_dims),
        dropout=dropout,
        activation=activation,
    ).to(device)

    if verbose >= 1:
        print(f"  Training on device: {device}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    # DataLoader keeps tensors on CPU; batches are moved to device in the loop
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    # ── History containers ────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # ── Validation ────────────────────────────────────────────────
        if has_val:
            model.eval()
            with torch.no_grad():
                logits_vl = model(X_vl)
                vl_loss = criterion(logits_vl, y_vl).item()
                vl_preds = (torch.sigmoid(logits_vl) > 0.5).float()
                vl_acc = (vl_preds == y_vl).float().mean().item()
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)

            # Early stopping
            if patience > 0:
                if vl_loss < best_val_loss:
                    best_val_loss = vl_loss
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose >= 1:
                            print(f"  Early stopping at epoch {epoch} "
                                  f"(best val_loss = {best_val_loss:.6f})")
                        break

        if verbose >= 1:
            vl_str = ""
            if has_val:
                vl_str = (f"  val_loss={history['val_loss'][-1]:.4f}"
                          f"  val_acc={history['val_acc'][-1]:.4f}")
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"loss={epoch_loss:.4f}  acc={epoch_acc:.4f}{vl_str}")

    # ── Restore best weights if early stopping was active ─────────────
    if patience > 0 and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, history

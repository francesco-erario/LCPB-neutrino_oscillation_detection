"""
nn_visualization.py
===================
Visualization helpers for the PyTorch neural-network workflow.

NN-specific plots (training curves with loss AND accuracy) live here.
All model-agnostic plots (ROC, PR, confusion matrix, energy spectra,
efficiency, score distribution, threshold scan, score-vs-variable) are
re-exported from ``visualization.py`` so that a single import suffices
in analysis notebooks.

Exported functions
------------------
NN-specific:
    plot_nn_training_curves  — loss and accuracy over epochs (2 panels)

Re-exported from visualization.py:
    plot_score_distribution
    plot_roc_curve
    plot_pr_curve
    plot_threshold_scan
    plot_confusion_matrix
    plot_energy_spectra
    plot_efficiency
    plot_score_vs_variable
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Re-export every model-agnostic visualization function so that
# ``from nn_visualization import *`` gives the full toolkit.
from visualization import (                     # noqa: F401
    plot_score_distribution,
    plot_roc_curve,
    plot_pr_curve,
    plot_threshold_scan,
    plot_confusion_matrix,
    plot_energy_spectra,
    plot_efficiency,
    plot_score_vs_variable,
    _C,   # colour palette
)


# ======================================================================
# NN-specific: training curves (loss + accuracy over epochs)
# ======================================================================

def plot_nn_training_curves(
    history: dict,
    best_epoch: int | None = None,
    title: str = "Training curves",
    figsize: tuple = (14, 5),
    savepath: str | None = None,
) -> plt.Figure:
    """
    Two-panel plot: loss and accuracy over epochs.

    Parameters
    ----------
    history     : dict returned by ``build_and_train_nn``, with keys
                  ``"train_loss"``, ``"val_loss"``, ``"train_acc"``,
                  ``"val_acc"`` (each a list of per-epoch values).
    best_epoch  : epoch index to highlight (e.g. early-stopping point).
                  If *None* the epoch with the lowest validation loss is
                  used (when available).
    title       : super-title for the figure
    figsize     : figure dimensions
    savepath    : optional file path to save the plot

    Returns
    -------
    fig : matplotlib Figure
    """
    has_val = len(history.get("val_loss", [])) > 0
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    if best_epoch is None and has_val:
        best_epoch = int(np.argmin(history["val_loss"])) + 1

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # -- Loss panel ----------------------------------------------------
    ax_loss.plot(epochs, history["train_loss"],
                 label="Train", color=_C["signal"], alpha=0.9)
    if has_val:
        ax_loss.plot(epochs, history["val_loss"],
                     label="Validation", color=_C["background"], alpha=0.9)
    if best_epoch is not None:
        ax_loss.axvline(best_epoch, color=_C["threshold"], linestyle="--",
                        linewidth=1.5, label=f"Best epoch = {best_epoch}")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (BCE)")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    # -- Accuracy panel ------------------------------------------------
    ax_acc.plot(epochs, history["train_acc"],
                label="Train", color=_C["signal"], alpha=0.9)
    if has_val:
        ax_acc.plot(epochs, history["val_acc"],
                    label="Validation", color=_C["background"], alpha=0.9)
    if best_epoch is not None:
        ax_acc.axvline(best_epoch, color=_C["threshold"], linestyle="--",
                       linewidth=1.5, label=f"Best epoch = {best_epoch}")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ======================================================================
# NN-specific: per-fold CV training curves
# ======================================================================

def plot_nn_cv_curves(
    fold_histories: list[dict],
    best_epochs: list[int] | None = None,
    metric: str = "loss",
    title: str = "Cross-validation training curves",
    figsize_per_fold: tuple = (5.5, 4),
    savepath: str | None = None,
) -> plt.Figure:
    """
    One panel per fold showing train/val curves, plus an aggregate panel.

    Parameters
    ----------
    fold_histories   : list of history dicts (one per fold)
    best_epochs      : list of early-stopping epochs (one per fold)
    metric           : ``"loss"`` or ``"acc"``
    title            : super-title
    figsize_per_fold : width, height of each individual panel
    savepath         : optional path to save

    Returns
    -------
    fig : matplotlib Figure
    """
    n_folds = len(fold_histories)
    tr_key = f"train_{metric}"
    vl_key = f"val_{metric}"

    w, h = figsize_per_fold
    fig, axes = plt.subplots(1, n_folds + 1,
                             figsize=((n_folds + 1) * w, h), sharey=True)
    fig.suptitle(title, fontsize=14)

    # -- Per-fold panels -----------------------------------------------
    for i, hist in enumerate(fold_histories):
        ax = axes[i]
        ep = np.arange(1, len(hist[tr_key]) + 1)
        ax.plot(ep, hist[tr_key], label="Train", color=_C["signal"], alpha=0.9)
        if vl_key in hist and len(hist[vl_key]) > 0:
            ax.plot(ep, hist[vl_key], label="Val", color=_C["background"], alpha=0.9)
        if best_epochs is not None:
            ax.axvline(best_epochs[i], color=_C["threshold"], linestyle="--",
                       linewidth=1.5, label=f"Best={best_epochs[i]}")
        ax.set_title(f"Fold {i + 1}", fontsize=12)
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel(metric.capitalize())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # -- Aggregate panel (mean ± std) ----------------------------------
    ax = axes[-1]
    max_len = min(len(h[vl_key]) for h in fold_histories)
    val_curves = np.array([h[vl_key][:max_len] for h in fold_histories])
    train_curves = np.array([h[tr_key][:max_len] for h in fold_histories])
    ep = np.arange(1, max_len + 1)

    ax.plot(ep, train_curves.mean(axis=0), color=_C["signal"], label="Train (mean)")
    ax.fill_between(ep,
                    train_curves.mean(0) - train_curves.std(0),
                    train_curves.mean(0) + train_curves.std(0),
                    color=_C["signal"], alpha=0.15)
    ax.plot(ep, val_curves.mean(axis=0), color=_C["background"], label="Val (mean)")
    ax.fill_between(ep,
                    val_curves.mean(0) - val_curves.std(0),
                    val_curves.mean(0) + val_curves.std(0),
                    color=_C["background"], alpha=0.15)
    if best_epochs is not None:
        mean_best = int(np.mean(best_epochs))
        ax.axvline(mean_best, color=_C["threshold"], linestyle="--",
                   linewidth=1.5, label=f"Mean best = {mean_best}")
    ax.set_title("Aggregate", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig

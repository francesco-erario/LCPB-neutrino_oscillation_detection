"""
nn_optimization.py
==================
Hyper-parameter optimization for the feed-forward neural network using
Optuna with a combined objective:

    score = λ_auc · AUC_val
          − λ_wass · Wasserstein(spectrum_truth, spectrum_selected)
          − λ_chi2 · χ²_reduced(region Ep < energy_threshold)

This mirrors the approach in ``Optimization.py`` (XGBoost) but adapts it
to the PyTorch training loop exposed by ``nn_model.build_and_train_nn``.

Exported functions
------------------
optimize_nn_with_spectra  : main optimization routine
summarize_study           : print a human-readable report of the study
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna.samplers import TPESampler
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

import torch
from nn_model import build_and_train_nn, get_device


# ======================================================================
# Spectral metric helpers (same as in Optimization.py)
# ======================================================================

def _wasserstein_spectra(
    E_truth: np.ndarray,
    E_selected: np.ndarray,
) -> float:
    """Wasserstein distance between two energy samples."""
    if len(E_truth) == 0 or len(E_selected) == 0:
        return np.inf
    return float(wasserstein_distance(E_truth, E_selected))


def _chi2_low_energy(
    E_truth: np.ndarray,
    E_selected: np.ndarray,
    energy_threshold: float,
    n_bins: int,
) -> float:
    """
    Reduced χ² in the low-energy region (E < energy_threshold).

    Counts are normalized (density=True) before comparison so that the
    metric is independent of the total number of selected events.  Bins
    with fewer than 5 truth events are ignored for numerical stability.
    """
    mask_t = E_truth < energy_threshold
    mask_s = E_selected < energy_threshold

    if mask_t.sum() < 10 or mask_s.sum() < 5:
        return np.inf

    bins = np.linspace(0.0, energy_threshold, n_bins + 1)
    h_t, _ = np.histogram(E_truth[mask_t], bins=bins, density=True)
    h_s, _ = np.histogram(E_selected[mask_s], bins=bins, density=True)

    valid = h_t > 5 / (len(E_truth[mask_t]) * (energy_threshold / n_bins))
    if valid.sum() == 0:
        return np.inf

    chi2 = np.sum(((h_s[valid] - h_t[valid]) ** 2) / (h_t[valid] + 1e-12))
    return float(chi2 / valid.sum())


# ======================================================================
# Main optimization function
# ======================================================================

def optimize_nn_with_spectra(
    # -- data -----------------------------------------------------------
    X_train,
    y_train,
    X_val,
    y_val,
    E_val: np.ndarray,
    E_truth_val: np.ndarray,

    # -- classification threshold for spectral metrics ------------------
    selection_threshold: float = 0.5,

    # -- objective weights ----------------------------------------------
    lambda_auc:  float = 1.0,
    lambda_wass: float = 1.0,
    lambda_chi2: float = 0.5,

    # -- spectral metric parameters -------------------------------------
    energy_threshold: float = 3.0,
    n_bins_chi2:      int   = 20,

    # -- search ranges for NN hyper-parameters --------------------------
    n_layers_range:      tuple[int, int]     = (1, 3),
    units_per_layer_range: tuple[int, int]   = (16, 128),
    dropout_range:       tuple[float, float] = (0.0, 0.5),
    lr_range:            tuple[float, float] = (1e-4, 1e-2),
    weight_decay_range:  tuple[float, float] = (0.0, 1e-2),
    batch_size_choices:  tuple[int, ...]     = (64, 128, 256, 512),
    epochs:              int                 = 40,
    patience:            int                 = 8,

    # -- Optuna ---------------------------------------------------------
    n_trials:     int  = 50,
    sampler_seed: int  = 42,
    verbose:      bool = True,

    # -- device ---------------------------------------------------------
    device: "torch.device | str | None" = None,

    # -- reproducibility ------------------------------------------------
    random_state: int = 42,

) -> tuple[dict, optuna.Study]:
    """
    Optimize neural-network hyper-parameters with a combined objective.

    The objective balances classification performance (AUC) against
    spectral fidelity (Wasserstein distance and reduced-χ² in the
    low-energy region), exactly as done for XGBoost in
    ``Optimization.py``.

    Parameters
    ----------
    X_train, y_train      : training data (array-like)
    X_val, y_val          : validation data
    E_val                 : 1-D array of Ep for ALL validation events
    E_truth_val           : 1-D array of Ep for true signal (label==1)
                            in the validation set
    selection_threshold   : score threshold for event selection
    lambda_auc/wass/chi2  : weights of the three objective terms
    energy_threshold      : upper edge of the low-energy region for χ²
    n_bins_chi2           : number of bins for the χ² calculation
    n_layers_range        : (min, max) number of hidden layers
    units_per_layer_range : (min, max) units per hidden layer
    dropout_range         : (min, max) dropout rate
    lr_range              : (min, max) learning rate (log scale)
    weight_decay_range    : (min, max) L2 regularization
    batch_size_choices    : categorical choices for batch size
    epochs                : max epochs per trial
    patience              : early-stopping patience per trial
    n_trials              : Optuna trials
    sampler_seed          : seed for TPE sampler
    verbose               : print progress
    device                : torch.device to train on.  ``None`` → auto-
                            detect via ``get_device()`` (CUDA → MPS → CPU).
    random_state          : global seed

    Returns
    -------
    best_params : dict with the best hyper-parameters found
    study       : ``optuna.Study`` (individual metrics stored as
                  user_attrs: "auc", "wasserstein", "chi2_low_energy",
                  "objective", "n_selected")
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Normalizers estimated at the first trial and then held constant.
    _norm: dict[str, float] = {}

    E_val_arr   = np.asarray(E_val)
    E_truth_arr = np.asarray(E_truth_val)

    def objective(trial: optuna.Trial) -> float:

        # -- Suggest hyper-parameters ----------------------------------
        n_layers = trial.suggest_int("n_layers", *n_layers_range)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dims.append(
                trial.suggest_int(f"units_layer_{i}", *units_per_layer_range)
            )

        dropout = trial.suggest_float("dropout", *dropout_range)
        lr = trial.suggest_float("lr", *lr_range, log=True)
        weight_decay = trial.suggest_float("weight_decay", *weight_decay_range)
        batch_size = trial.suggest_categorical("batch_size", list(batch_size_choices))

        # -- Train the model -------------------------------------------
        model, _ = build_and_train_nn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device=device,
            random_state=random_state,
            verbose=0,
        )

        # -- Compute metrics -------------------------------------------
        scores_val = model.predict_proba(np.asarray(X_val))

        auc_val = roc_auc_score(y_val, scores_val)

        selected_mask = scores_val > selection_threshold
        E_selected = E_val_arr[selected_mask]

        wass = _wasserstein_spectra(E_truth_arr, E_selected)
        chi2 = _chi2_low_energy(
            E_truth_arr, E_selected,
            energy_threshold, n_bins_chi2,
        )

        # -- Normalization (warm-up at first trial) --------------------
        if not _norm:
            _norm["auc"]  = max(abs(auc_val), 1e-9)
            _norm["wass"] = max(abs(wass), 1e-9) if np.isfinite(wass) else 1.0
            _norm["chi2"] = max(abs(chi2), 1e-9) if np.isfinite(chi2) else 1.0

        auc_norm  = auc_val / _norm["auc"]
        wass_norm = (wass / _norm["wass"]) if np.isfinite(wass) else 5.0
        chi2_norm = (chi2 / _norm["chi2"]) if np.isfinite(chi2) else 5.0

        score = (
            lambda_auc  * auc_norm
            - lambda_wass * wass_norm
            - lambda_chi2 * chi2_norm
        )

        # -- Store individual metrics ----------------------------------
        trial.set_user_attr("auc",             round(auc_val, 6))
        trial.set_user_attr("wasserstein",     round(float(wass), 6)
                                               if np.isfinite(wass) else None)
        trial.set_user_attr("chi2_low_energy", round(float(chi2), 6)
                                               if np.isfinite(chi2) else None)
        trial.set_user_attr("objective",       round(score, 6))
        trial.set_user_attr("n_selected",      int(selected_mask.sum()))
        trial.set_user_attr("hidden_dims",     hidden_dims)

        return score

    # -- Create and run the study --------------------------------------
    sampler = TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    callbacks = [_ProgressCallback(n_trials)] if verbose else []
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=False,
    )

    best_params = study.best_params
    # Reconstruct hidden_dims from the flat trial params
    best_params["hidden_dims"] = study.best_trial.user_attrs["hidden_dims"]

    if verbose:
        bt = study.best_trial
        print(f"\n{'='*70}")
        print(f"  Best trial: #{bt.number}")
        print(f"  Objective   : {bt.value:.6f}")
        print(f"  AUC         : {bt.user_attrs['auc']:.6f}")
        print(f"  Wasserstein : {bt.user_attrs['wasserstein']}")
        print(f"  χ² low-E    : {bt.user_attrs['chi2_low_energy']}")
        print(f"  N selected  : {bt.user_attrs['n_selected']}")
        print(f"{'='*70}")
        print("  Best hyper-parameters:")
        for k, v in best_params.items():
            print(f"    {k:22s}: {v}")

    return best_params, study


# ======================================================================
# Progress callback
# ======================================================================

class _ProgressCallback:
    """One-line-per-trial progress printer."""

    def __init__(self, n_trials: int):
        self.n_trials = n_trials
        self._header_printed = False

    def __call__(self, study: optuna.Study, trial: optuna.FrozenTrial):
        if not self._header_printed:
            print(f"\n{'Trial':>6} │ {'Objective':>10} │ {'AUC':>8} │"
                  f" {'Wasserstein':>12} │ {'χ² low-E':>10} │ {'N sel':>7} │ Status")
            print("─" * 75)
            self._header_printed = True

        attrs = trial.user_attrs
        wass = (f"{attrs.get('wasserstein', 'inf'):>12.4f}"
                if attrs.get("wasserstein") is not None else f"{'inf':>12}")
        chi2 = (f"{attrs.get('chi2_low_energy', 'inf'):>10.4f}"
                if attrs.get("chi2_low_energy") is not None else f"{'inf':>10}")

        status = "★ BEST" if trial.number == study.best_trial.number else ""
        print(
            f"{trial.number:>6} │ {attrs.get('objective', 0.0):>10.6f} │"
            f" {attrs.get('auc', 0.0):>8.6f} │"
            f" {wass} │ {chi2} │ {attrs.get('n_selected', 0):>7} │ {status}"
        )


# ======================================================================
# Report utility
# ======================================================================

def summarize_study(
    study: optuna.Study,
    top_n: int = 5,
) -> None:
    """Print a concise report of the top-N trials."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    print(f"\n{'='*80}")
    print(f"  STUDY: {len(completed)} completed trials")
    print(f"{'='*80}")
    print(f"{'Rank':>5} │ {'Trial':>6} │ {'Objective':>10} │ "
          f"{'AUC':>8} │ {'Wasserstein':>12} │ {'χ² low-E':>10} │ {'N sel':>7}")
    print("─" * 80)

    for rank, t in enumerate(completed[:top_n], 1):
        a = t.user_attrs
        wass_str = (f"{a['wasserstein']:.4f}"
                    if a.get("wasserstein") is not None else "inf")
        chi2_str = (f"{a['chi2_low_energy']:.4f}"
                    if a.get("chi2_low_energy") is not None else "inf")
        print(
            f"{rank:>5} │ {t.number:>6} │ {t.value:>10.6f} │"
            f" {a.get('auc', 0.0):>8.6f} │ {wass_str:>12} │"
            f" {chi2_str:>10} │ {a.get('n_selected', 0):>7}"
        )

    print(f"{'='*80}")
    print(f"\n  Best hyper-parameters (trial #{study.best_trial.number})")
    print("─" * 40)
    for k, v in study.best_params.items():
        print(f"  {k:22s}: {v}")

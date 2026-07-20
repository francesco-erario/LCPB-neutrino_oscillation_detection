"""
optimization.py
===============
Ottimizzazione degli iperparametri XGBoost con funzione obiettivo combinata:

    score = λ_auc · AUC_val
          − λ_wass · Wasserstein(spettro_truth, spettro_selezionato)
          − λ_chi2 · χ²_ridotto(regione E1 < energy_threshold)

In questo modo Optuna non cerca solo il miglior classificatore in senso
statistico, ma anche il modello i cui eventi selezionati riproducono meglio
lo spettro vero di E1 — in particolare nella regione a bassa energia dove la
de-correlazione è più difficile.

Funzioni esportate
------------------
optimize_with_spectra  : ottimizzazione principale
summarize_study        : stampa un report dello studio e dei migliori trial
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from typing import Sequence


# ===========================================================================
# Funzioni di metrica spettrale (usate internamente ma esportate per utilità)
# ===========================================================================

def _wasserstein_spectra(
    E_truth:    np.ndarray,
    E_selected: np.ndarray,
) -> float:
    """
    Wasserstein distance (Earth Mover's Distance) tra due campioni di energia.
    Non richiede binning — agisce direttamente sugli array.
    Restituisce 0 se uno dei due campioni è vuoto.
    """
    if len(E_truth) == 0 or len(E_selected) == 0:
        return np.inf
    return float(wasserstein_distance(E_truth, E_selected))


def _chi2_low_energy(
    E_truth:          np.ndarray,
    E_selected:       np.ndarray,
    energy_threshold: float,
    n_bins:           int,
) -> float:
    """
    χ² ridotto bin-per-bin nella regione E1 < energy_threshold.

    I conteggi vengono normalizzati (density=True) prima del confronto,
    così la metrica è indipendente dal numero totale di eventi selezionati.
    Bin con meno di 5 eventi nella distribuzione truth vengono ignorati
    per evitare instabilità numerica.

    Restituisce np.inf se i campioni nella regione sono troppo pochi.
    """
    mask_t = E_truth    < energy_threshold
    mask_s = E_selected < energy_threshold

    if mask_t.sum() < 10 or mask_s.sum() < 5:
        return np.inf

    bins = np.linspace(0.0, energy_threshold, n_bins + 1)
    h_t, _ = np.histogram(E_truth[mask_t],    bins=bins, density=True)
    h_s, _ = np.histogram(E_selected[mask_s], bins=bins, density=True)

    valid = h_t > 5 / (len(E_truth[mask_t]) * (energy_threshold / n_bins))
    if valid.sum() == 0:
        return np.inf

    chi2 = np.sum(((h_s[valid] - h_t[valid]) ** 2) / (h_t[valid] + 1e-12))
    return float(chi2 / valid.sum())


# ===========================================================================
# Funzione principale di ottimizzazione
# ===========================================================================

def optimize_with_spectra(
    # ── Dati ────────────────────────────────────────────────────────────────
    X_train,
    y_train,
    X_val,
    y_val,
    E_val:             np.ndarray,   # energia E1 del validation set
                                     # (già ritrasformata se vuoi visualizzarla
                                     #  in unità fisiche)

    # ── Energia del segnale vero (label==1) nel validation set ───────────────
    # Passa direttamente l'array con la trasformazione che preferisci.
    # Es: E_truth_val = val_df.loc[val_df["label1"]==1, "E1"].values
    #     oppure E_truth_val = scaler.inverse_transform(E_val[y_val==1])
    E_truth_val:       np.ndarray,

    # ── Soglia di classificazione per ricavare gli eventi selezionati ────────
    # Usata per calcolare E_selected durante ogni trial.
    # Suggerimento: passa la soglia ottimale trovata con plot_threshold_scan,
    # oppure un valore fisso (es. 0.5).
    selection_threshold: float = 0.5,

    # ── Pesi della funzione obiettivo ────────────────────────────────────────
    lambda_auc:    float = 1.0,   # peso AUC (da massimizzare)
    lambda_wass:   float = 1.0,   # peso Wasserstein (penalità)
    lambda_chi2:   float = 0.5,   # peso χ² low-energy (penalità)

    # ── Parametri per le metriche spettrali ──────────────────────────────────
    energy_threshold:  float = 3.0,  # soglia per la regione low-energy nel χ²
    n_bins_chi2:       int   = 20,   # bin per il χ² low-energy

    # ── Range di ricerca iperparametri XGBoost ───────────────────────────────
    n_estimators_range:      tuple[int, int]     = (50, 300),
    max_depth_range:         tuple[int, int]     = (3, 8),
    learning_rate_range:     tuple[float, float] = (0.01, 0.3),
    subsample_range:         tuple[float, float] = (0.5, 1.0),
    colsample_bytree_range:  tuple[float, float] = (0.5, 1.0),
    min_child_weight_range:  tuple[int, int]     = (1, 20),
    reg_lambda_range:        tuple[float, float] = (0.1, 5.0),
    reg_alpha_range:         tuple[float, float] = (0.0, 2.0),

    # ── Optuna ───────────────────────────────────────────────────────────────
    n_trials:        int  = 50,
    sampler_seed:    int  = 42,
    verbose:         bool = True,

    # ── XGBoost fissi ────────────────────────────────────────────────────────
    tree_method:           str = "hist",
    eval_metric:           str = "logloss",
    early_stopping_rounds: int = 15,
    random_state:          int = 42,

) -> tuple[dict, optuna.Study]:
    """
    Ottimizzazione con funzione obiettivo combinata:

        objective = λ_auc · AUC_val
                  − λ_wass · Wasserstein(E_truth_val, E_selected_val)
                  − λ_chi2 · χ²_ridotto(E1 < energy_threshold)

    Tutti i termini vengono normalizzati internamente rispetto al primo trial
    (warm-up), in modo che i λ rappresentino pesi relativi comparabili
    indipendentemente dalle scale assolute delle metriche.

    Parametri
    ----------
    X_train, y_train     : training set (features e label)
    X_val, y_val         : validation set (features e label)
    E_val                : array 1-D con i valori di E1 di TUTTI gli eventi
                           del validation set, nella scala che preferisci.
                           Usato per ricavare E_selected selezionando gli
                           eventi in base allo score e alla selection_threshold.
    E_truth_val          : array 1-D con i valori di E1 degli eventi di
                           segnale vero (label==1) del validation set,
                           nella stessa scala di E_val.
                           Puoi passarlo direttamente come:
                               E_val[y_val == 1]
                           oppure con qualsiasi trasformazione tu voglia.
    selection_threshold  : soglia per selezionare gli eventi durante ogni
                           trial (score > threshold → evento selezionato).
    lambda_auc           : peso del termine AUC nella funzione obiettivo
    lambda_wass          : peso (penalità) del termine Wasserstein
    lambda_chi2          : peso (penalità) del termine χ² low-energy
    energy_threshold     : soglia energetica per il χ² low-energy
    n_bins_chi2          : bin per il χ² low-energy
    *_range              : range (min, max) per ogni iperparametro XGBoost
    n_trials             : numero di trial
    sampler_seed         : seed per TPESampler
    verbose              : se True, stampa info durante l'ottimizzazione
    tree_method          : "hist" o "exact"
    eval_metric          : metrica interna XGBoost
    early_stopping_rounds: early stopping round
    random_state         : seed XGBoost

    Restituisce
    -----------
    best_params : dict con i migliori iperparametri trovati
    study       : oggetto optuna.Study completo
                  (ogni trial contiene le metriche individuali via user_attrs:
                   "auc", "wasserstein", "chi2_low_energy", "objective")
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Normalizzatori: vengono stimati al primo trial (warm-up) e poi fissi.
    # Questo garantisce che i λ rappresentino pesi relativi stabili.
    _norm: dict[str, float] = {}

    E_val_arr      = np.asarray(E_val)
    E_truth_arr    = np.asarray(E_truth_val)

    def objective(trial: optuna.Trial) -> float:

        # ── Suggerisci iperparametri ─────────────────────────────────────────
        params = dict(
            n_estimators          = trial.suggest_int(
                                        "n_estimators", *n_estimators_range),
            max_depth             = trial.suggest_int(
                                        "max_depth", *max_depth_range),
            learning_rate         = trial.suggest_float(
                                        "learning_rate", *learning_rate_range, log=True),
            subsample             = trial.suggest_float(
                                        "subsample", *subsample_range),
            colsample_bytree      = trial.suggest_float(
                                        "colsample_bytree", *colsample_bytree_range),
            min_child_weight      = trial.suggest_int(
                                        "min_child_weight", *min_child_weight_range),
            reg_lambda            = trial.suggest_float(
                                        "reg_lambda", *reg_lambda_range),
            reg_alpha             = trial.suggest_float(
                                        "reg_alpha", *reg_alpha_range),
            tree_method           = tree_method,
            eval_metric           = eval_metric,
            early_stopping_rounds = early_stopping_rounds,
            random_state          = random_state,
        )

        # ── Allena il modello ────────────────────────────────────────────────
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            verbose  = False,
        )

        # ── Calcola le metriche ──────────────────────────────────────────────
        scores_val  = model.predict_proba(X_val)[:, 1]

        # AUC
        auc_val = roc_auc_score(y_val, scores_val)

        # Spettro degli eventi selezionati
        selected_mask = scores_val > selection_threshold
        E_selected    = E_val_arr[selected_mask]

        # Wasserstein distance
        wass = _wasserstein_spectra(E_truth_arr, E_selected)

        # χ² ridotto nella regione low-energy
        chi2 = _chi2_low_energy(
            E_truth_arr, E_selected,
            energy_threshold, n_bins_chi2,
        )

        # ── Normalizzazione al primo trial ───────────────────────────────────
        if not _norm:
            _norm["auc"]  = max(abs(auc_val), 1e-9)
            _norm["wass"] = max(abs(wass),    1e-9) if np.isfinite(wass) else 1.0
            _norm["chi2"] = max(abs(chi2),    1e-9) if np.isfinite(chi2) else 1.0

        auc_norm  = auc_val / _norm["auc"]
        wass_norm = (wass / _norm["wass"]) if np.isfinite(wass) else 5.0
        chi2_norm = (chi2 / _norm["chi2"]) if np.isfinite(chi2) else 5.0

        # ── Funzione obiettivo combinata (da massimizzare) ───────────────────
        score = (
              lambda_auc  *  auc_norm
            - lambda_wass *  wass_norm
            - lambda_chi2 *  chi2_norm
        )

        # Salva le metriche individuali nel trial per analisi post-hoc
        trial.set_user_attr("auc",            round(auc_val, 6))
        trial.set_user_attr("wasserstein",    round(float(wass), 6)
                                              if np.isfinite(wass) else None)
        trial.set_user_attr("chi2_low_energy",round(float(chi2), 6)
                                              if np.isfinite(chi2) else None)
        trial.set_user_attr("objective",      round(score, 6))
        trial.set_user_attr("n_selected",     int(selected_mask.sum()))

        return score

    # ── Crea ed esegui lo studio ─────────────────────────────────────────────
    sampler = TPESampler(seed=sampler_seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    callbacks = [_ProgressCallback(n_trials)] if verbose else []
    study.optimize(objective, n_trials=n_trials,
                   callbacks=callbacks, show_progress_bar=False)

    best_params = study.best_params

    if verbose:
        best_t = study.best_trial
        print(f"\n{'='*70}")
        print(f"  Best trial: #{best_t.number}")
        print(f"  Objective   : {best_t.value:.6f}")
        print(f"  AUC         : {best_t.user_attrs['auc']:.6f}")
        print(f"  Wasserstein : {best_t.user_attrs['wasserstein']}")
        print(f"  χ² low-E    : {best_t.user_attrs['chi2_low_energy']}")
        print(f"  N selected  : {best_t.user_attrs['n_selected']}")
        print(f"{'='*70}")
        print("  Best hyperparameters:")
        for k, v in best_params.items():
            print(f"    {k:22s}: {v}")

    return best_params, study


# ===========================================================================
# Callback per la progress bar custom
# ===========================================================================

class _ProgressCallback:
    """Stampa una riga per ogni trial con le metriche principali."""

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
        wass  = f"{attrs.get('wasserstein', 'inf'):>12.4f}" \
                if attrs.get("wasserstein") is not None else f"{'inf':>12}"
        chi2  = f"{attrs.get('chi2_low_energy', 'inf'):>10.4f}" \
                if attrs.get("chi2_low_energy") is not None else f"{'inf':>10}"

        status = "★ BEST" if trial.number == study.best_trial.number else ""
        print(
            f"{trial.number:>6} │ {attrs.get('objective', 0.0):>10.6f} │"
            f" {attrs.get('auc', 0.0):>8.6f} │"
            f" {wass} │ {chi2} │ {attrs.get('n_selected', 0):>7} │ {status}"
        )


# ===========================================================================
# Funzione di report
# ===========================================================================

def summarize_study(
    study:  optuna.Study,
    top_n:  int = 5,
) -> None:
    """
    Stampa un report dello studio con le tre metriche per i migliori N trial.

    Parametri
    ----------
    study : oggetto optuna.Study restituito da optimize_with_spectra
    top_n : numero di trial da mostrare
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    print(f"\n{'='*80}")
    print(f"  STUDIO: {len(completed)} trial completati")
    print(f"{'='*80}")
    print(f"{'Rank':>5} │ {'Trial':>6} │ {'Objective':>10} │ "
          f"{'AUC':>8} │ {'Wasserstein':>12} │ {'χ² low-E':>10} │ {'N sel':>7}")
    print("─" * 80)

    for rank, t in enumerate(completed[:top_n], 1):
        a  = t.user_attrs
        wass_str = f"{a['wasserstein']:.4f}" \
                   if a.get("wasserstein") is not None else "inf"
        chi2_str = f"{a['chi2_low_energy']:.4f}" \
                   if a.get("chi2_low_energy") is not None else "inf"
        print(
            f"{rank:>5} │ {t.number:>6} │ {t.value:>10.6f} │"
            f" {a.get('auc', 0.0):>8.6f} │ {wass_str:>12} │"
            f" {chi2_str:>10} │ {a.get('n_selected', 0):>7}"
        )

    print(f"{'='*80}")
    print("\n  Best hyperparameters (trial #{})".format(study.best_trial.number))
    print("─" * 40)
    for k, v in study.best_params.items():
        print(f"  {k:22s}: {v}")


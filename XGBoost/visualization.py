"""
visualization.py
================
Funzioni di visualizzazione per il training e la valutazione del modello XGBoost.

Ogni funzione riceve direttamente i dati da plottare (array, Series, dict) —
non il modello grezzo — in modo che tu possa applicare qualsiasi trasformazione
prima di passarli (rescaling, back-transform, selezione, ecc.).

Funzioni disponibili
--------------------
plot_training_curves    : logloss train/val durante il training (overfitting check)
plot_feature_importance : importanza delle feature
plot_score_distribution : distribuzione dello score per classe
plot_roc_curve          : curva ROC con AUC
plot_pr_curve           : curva Precision-Recall con AP
plot_threshold_scan     : TP/FP/FN, Precision/Recall/F1 al variare della soglia
plot_confusion_matrix   : matrice di confusione
plot_energy_spectra     : confronto spettri energetici (ground truth vs selezionati)
plot_efficiency         : efficienza di selezione in funzione dell'energia
plot_score_vs_energy    : scatter score vs grandezza continua (es. E1) per
                          visualizzare la correlazione residua
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from typing import Sequence

# ---------------------------------------------------------------------------
# Stile globale
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    1.2,
    "axes.grid":         True,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "legend.fontsize":   11,
    "legend.framealpha": 0.85,
    "font.family":       "sans-serif",
    "lines.linewidth":   2.0,
})

# Palette coerente
_C = {
    "signal":     "#2196F3",   # blu
    "background": "#F44336",   # rosso
    "tp":         "#4CAF50",   # verde
    "fp":         "#F44336",   # rosso
    "fn":         "#FF9800",   # arancione
    "tn":         "#9E9E9E",   # grigio
    "truth":      "#2196F3",   # blu
    "predicted":  "#F44336",   # rosso
    "subtracted": "#4CAF50",   # verde
    "threshold":  "#9C27B0",   # viola
    "neutral":    "#607D8B",   # grigio-blu
}


# ===========================================================================
# 1. Training curves
# ===========================================================================

def plot_training_curves(
    evals_result: dict,
    metric:       str  = "logloss",
    train_key:    str  = "validation_0",
    val_key:      str  = "validation_1",
    best_iteration: int | None = None,
    title:        str  = "Training curves",
    figsize:      tuple = (8, 5),
    savepath:     str | None = None,
) -> plt.Figure:
    """
    Curve di logloss (o altra metrica) su train e validation.

    Parametri
    ----------
    evals_result  : dizionario da model.evals_result(), es.
                    {"validation_0": {"logloss": [...]},
                     "validation_1": {"logloss": [...]}}
    metric        : chiave della metrica (es. "logloss", "auc", "error")
    train_key     : chiave del training set in evals_result
    val_key       : chiave del validation set in evals_result
    best_iteration: numero del best round (model.best_iteration) — traccia
                    una linea verticale tratteggiata se fornito
    title         : titolo del grafico
    figsize       : dimensioni della figura
    savepath      : se fornito, salva il file (es. "plot.png")

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    train_vals = evals_result[train_key][metric]
    val_vals   = evals_result[val_key][metric]

    ax.plot(train_vals, label="Train",      color=_C["signal"],     alpha=0.9)
    ax.plot(val_vals,   label="Validation", color=_C["background"], alpha=0.9)

    if best_iteration is not None:
        ax.axvline(best_iteration, color=_C["threshold"], linestyle="--",
                   linewidth=1.5, label=f"Best iter={best_iteration}")

    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# 2. Feature importance
# ===========================================================================

def plot_feature_importance(
    importances:   Sequence[float],
    feature_names: Sequence[str],
    top_n:         int = 15,
    title:         str = "Feature importance",
    figsize:       tuple = (8, 5),
    savepath:      str | None = None,
) -> plt.Figure:
    """
    Bar chart orizzontale dell'importanza delle feature.

    Parametri
    ----------
    importances   : array di importanza (model.feature_importances_)
    feature_names : lista di nomi corrispondenti (X_train.columns)
    top_n         : numero di feature da mostrare
    title         : titolo del grafico
    figsize       : dimensioni
    savepath      : path di salvataggio opzionale

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    importances   = np.array(importances)
    feature_names = np.array(feature_names)

    idx    = np.argsort(importances)[::-1][:top_n]
    imp    = importances[idx][::-1]
    names  = feature_names[idx][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(imp)))
    ax.barh(names, imp, color=colors, edgecolor="white")
    ax.set_xlabel("Importance score")
    ax.set_title(title)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# 3. Score distribution per classe
# ===========================================================================

def plot_score_distribution(
    scores:     np.ndarray,
    y_true:     np.ndarray,
    thresholds: Sequence[float] | None = None,
    threshold_labels: Sequence[str] | None = None,
    n_bins:     int  = 80,
    log_y:      bool = False,
    title:      str  = "Score distribution",
    figsize:    tuple = (8, 5),
    savepath:   str | None = None,
) -> plt.Figure:
    """
    Istogramma dello score separato per signal (y=1) e background (y=0).

    Parametri
    ----------
    scores     : array di score (predict_proba[:, 1])
    y_true     : array di label vere (0/1)
    thresholds : lista di soglie da visualizzare come linee verticali
    threshold_labels : etichette per le soglie (stesso ordine)
    n_bins     : numero di bin
    log_y      : scala logaritmica sull'asse y
    title      : titolo
    figsize    : dimensioni
    savepath   : path di salvataggio opzionale

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.linspace(0, 1, n_bins)

    ax.hist(scores[y_true == 1], bins=bins, histtype="step", linewidth=2,
            color=_C["signal"],     density=True, label="Signal (y=1)")
    ax.hist(scores[y_true == 0], bins=bins, histtype="step", linewidth=2,
            color=_C["background"], density=True, label="Background (y=0)")

    thr_colors = [_C["threshold"], _C["neutral"], "#795548", "#009688"]
    if thresholds is not None:
        for i, thr in enumerate(thresholds):
            label = threshold_labels[i] if threshold_labels else f"thr={thr:.3f}"
            ax.axvline(thr, linestyle="--", linewidth=1.5,
                       color=thr_colors[i % len(thr_colors)], label=label)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Classifier score")
    ax.set_ylabel("Normalized counts")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# 4. ROC curve
# ===========================================================================

def plot_roc_curve(
    y_true:   np.ndarray,
    y_score:  np.ndarray,
    label:    str  = "Model",
    log_x:    bool = False,
    title:    str  = "ROC curve",
    figsize:  tuple = (7, 6),
    savepath: str | None = None,
) -> plt.Figure:
    """
    Curva ROC con AUC. Supporta anche scala logaritmica sull'asse FPR.

    Parametri
    ----------
    y_true   : label vere (0/1)
    y_score  : score del modello (predict_proba[:, 1])
    label    : nome del modello in legenda
    log_x    : se True, usa scala log sull'asse x (utile per fisica HEP)
    title    : titolo
    figsize  : dimensioni
    savepath : path di salvataggio opzionale

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color=_C["signal"], linewidth=2,
            label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=_C["neutral"],
            linewidth=1.2, label="Random classifier")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# 5. Precision-Recall curve
# ===========================================================================

def plot_pr_curve(
    y_true:   np.ndarray,
    y_score:  np.ndarray,
    label:    str  = "Model",
    title:    str  = "Precision-Recall curve",
    figsize:  tuple = (7, 6),
    savepath: str | None = None,
) -> plt.Figure:
    """
    Curva Precision-Recall con Average Precision (AP).

    Parametri
    ----------
    y_true   : label vere (0/1)
    y_score  : score del modello (predict_proba[:, 1])
    label    : nome del modello in legenda
    title    : titolo
    figsize  : dimensioni
    savepath : path di salvataggio opzionale

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color=_C["signal"], linewidth=2,
            label=f"{label} (AP = {ap:.4f})")
    ax.axhline(baseline, linestyle="--", color=_C["neutral"],
               linewidth=1.2, label=f"Baseline (signal fraction = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# 6. Threshold scan
# ===========================================================================

def plot_threshold_scan(
    y_true:      np.ndarray,
    y_score:     np.ndarray,
    n_points:    int  = 200,
    best_thr:    float | None = None,
    title:       str  = "Threshold scan",
    figsize:     tuple = (18, 5),
    savepath:    str | None = None,
) -> tuple[plt.Figure, float]:
    """
    Tre pannelli: distribuzione score | TP/FP/FN vs soglia | Prec/Rec/F1 vs soglia.
    Calcola automaticamente la soglia che massimizza F1.

    Parametri
    ----------
    y_true    : label vere (0/1)
    y_score   : score del modello
    n_points  : numero di soglie da scansionare
    best_thr  : soglia da evidenziare (se None, usa quella con F1 massimo)
    title     : titolo principale
    figsize   : dimensioni
    savepath  : path di salvataggio opzionale

    Restituisce
    -----------
    fig      : matplotlib Figure
    best_thr : soglia ottimale (F1 max)
    """
    thresholds = np.linspace(0, 1, n_points)

    tp_list, fp_list, fn_list = [], [], []
    prec_list, rec_list, f1_list = [], [], []

    for thr in thresholds:
        yp = (y_score > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    if best_thr is None:
        best_thr = float(thresholds[np.argmax(f1_list)])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # ── 1. Distribuzione score ──────────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 80)
    ax.hist(y_score[y_true == 0], bins=bins, histtype="step", lw=2,
            density=True, color=_C["background"], label="Background")
    ax.hist(y_score[y_true == 1], bins=bins, histtype="step", lw=2,
            density=True, color=_C["signal"], label="Signal")
    ax.axvline(best_thr, linestyle="--", lw=1.5,
               color=_C["threshold"], label=f"Best F1 thr={best_thr:.3f}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Normalized counts")
    ax.set_title("Score distribution")
    ax.legend()

    # ── 2. TP / FP / FN vs soglia ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(thresholds, tp_list, color=_C["tp"], lw=2, label="True Positive")
    ax.plot(thresholds, fp_list, color=_C["fp"], lw=2, label="False Positive")
    ax.plot(thresholds, fn_list, color=_C["fn"], lw=2, label="False Negative")
    ax.axvline(best_thr, linestyle="--", lw=1.5,
               color=_C["threshold"], label=f"Best F1 thr={best_thr:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Counts")
    ax.set_title("TP / FP / FN vs threshold")
    ax.legend()

    # ── 3. Precision / Recall / F1 vs soglia ────────────────────────────────
    ax = axes[2]
    ax.plot(thresholds, prec_list, color=_C["signal"],     lw=2, label="Precision")
    ax.plot(thresholds, rec_list,  color=_C["background"], lw=2, label="Recall")
    ax.plot(thresholds, f1_list,   color=_C["tp"],         lw=2, label="F1")
    ax.axvline(best_thr, linestyle="--", lw=1.5,
               color=_C["threshold"], label=f"Best F1 thr={best_thr:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs threshold")
    ax.legend()

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, best_thr


# ===========================================================================
# 7. Confusion matrix
# ===========================================================================

def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    labels:     Sequence[str] = ("Background", "Signal"),
    threshold:  float | None = None,
    title:      str = "Confusion Matrix",
    figsize:    tuple = (6, 5),
    savepath:   str | None = None,
) -> plt.Figure:
    """
    Confusion matrix con metriche testuali (precision, recall, F1).

    Parametri
    ----------
    y_true    : label vere (0/1)
    y_pred    : label predette (0/1) — non gli score; applica già la soglia
                che preferisci (es. y_pred = (scores > 0.5).astype(int))
    labels    : nomi delle classi
    threshold : valore della soglia usata, mostrato nel titolo
    title     : titolo
    figsize   : dimensioni
    savepath  : path di salvataggio opzionale

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    if threshold is not None:
        title = f"{title} (threshold={threshold:.3f})"

    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    stats = f"Precision={prec:.3f}   Recall={rec:.3f}   F1={f1:.3f}"
    fig.text(0.5, -0.02, stats, ha="center", fontsize=11, color="#444444")

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# 8. Energy spectra (confronto spettri)
# ===========================================================================

def plot_energy_spectra(
    E_true:          np.ndarray,
    E_selected:      np.ndarray,
    E_tp:            np.ndarray | None = None,
    E_background:    np.ndarray | None = None,
    E_subtracted:    np.ndarray | None = None,
    bins:            np.ndarray | int  = 80,
    log_y:           bool  = False,
    x_label:         str   = "Energy",
    title:           str   = "Energy spectrum",
    figsize:         tuple = (9, 6),
    savepath:        str | None = None,
) -> plt.Figure:
    """
    Confronto degli spettri energetici: ground truth, selezione del modello,
    true positive, background stimato, segnale sottratto.

    **Tutti gli array di energia che passi devono essere già nel dominio
    che vuoi visualizzare** (es. già ritrasformati se hai applicato uno
    scaling durante il training).

    Parametri
    ----------
    E_true        : energie del segnale vero (label==1)
    E_selected    : energie degli eventi selezionati dal modello
    E_tp          : energie dei true positive (opzionale)
    E_background  : stime del background (es. da shifted dataset) (opzionale)
    E_subtracted  : E_selected - E_background, già calcolato (opzionale)
    bins          : array di bin bordi o numero intero
    log_y         : scala log sull'asse y
    x_label       : etichetta asse x
    title         : titolo
    figsize       : dimensioni
    savepath      : path di salvataggio opzionale

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    # Auto-crea i bin se viene passato un intero
    if isinstance(bins, int):
        all_values = np.concatenate([E_true, E_selected])
        bins = np.linspace(all_values.min(), all_values.max(), bins)

    fig, ax = plt.subplots(figsize=figsize)

    ax.step(bins[:-1], np.histogram(E_true,     bins=bins)[0],
            where="post", lw=2, color=_C["truth"],     label="Ground truth")
    ax.step(bins[:-1], np.histogram(E_selected, bins=bins)[0],
            where="post", lw=2, color=_C["predicted"], label="Selected by model")

    if E_tp is not None:
        ax.step(bins[:-1], np.histogram(E_tp, bins=bins)[0],
                where="post", lw=2, linestyle="--",
                color=_C["tp"], label="True Positive")

    if E_background is not None:
        ax.step(bins[:-1], np.histogram(E_background, bins=bins)[0],
                where="post", lw=2, #linestyle="-",
                color=_C["neutral"], label="Estimated background")

    if E_subtracted is not None:
        ax.step(bins[:-1], np.histogram(E_subtracted, bins=bins)[0],
                where="post", lw=2, color=_C["subtracted"],
                label="Selected − Background")

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# 9. Efficiency vs energy
# ===========================================================================

def plot_efficiency(
    E_before:  np.ndarray,
    E_after:   np.ndarray,
    bins:      np.ndarray | int  = 30,
    log_x:     bool  = False,
    x_label:   str   = "Energy",
    title:     str   = "Selection efficiency",
    figsize:   tuple = (8, 5),
    savepath:  str | None = None,
) -> plt.Figure:
    """
    Efficienza di selezione in funzione dell'energia:
        efficiency(bin) = N_after(bin) / N_before(bin)

    **Gli array passati devono essere già ritrasformati se necessario.**

    Parametri
    ----------
    E_before : energia di tutti gli eventi di segnale prima della selezione
    E_after  : energia degli eventi di segnale dopo la selezione (True Positive)
    bins     : bordi bin o numero intero; se int e log_x=True, usa scala log
    log_x    : scala logaritmica sull'asse x (e usa bin log-spaziati se bins è int)
    x_label  : etichetta asse x
    title    : titolo
    figsize  : dimensioni
    savepath : path di salvataggio opzionale

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    if isinstance(bins, int):
        if log_x:
            bins = np.logspace(
                np.log10(max(E_before.min(), 1e-9)),
                np.log10(E_before.max()),
                bins,
            )
        else:
            bins = np.linspace(E_before.min(), E_before.max(), bins)

    h_before, _ = np.histogram(E_before, bins=bins)
    h_after,  _ = np.histogram(E_after,  bins=bins)
    efficiency  = h_after / (h_before + 1e-12)
    bin_centers = np.sqrt(bins[:-1] * bins[1:]) if log_x else 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(bin_centers, efficiency, marker="o", markersize=5,
            color=_C["signal"], linewidth=2)
    ax.axhline(1.0, linestyle="--", color=_C["neutral"], linewidth=1.0, label="100%")
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim([0, 1.15])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Efficiency")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# 10. Score vs continuous variable (correlazione residua)
# ===========================================================================

def plot_score_vs_variable(
    variable:    np.ndarray,
    scores:      np.ndarray,
    y_true:      np.ndarray | None = None,
    threshold:   float | None = None,
    n_bins:      int  = 40,
    show_mean:   bool = True,
    var_label:   str  = "Variable",
    title:       str  = "Score vs variable",
    figsize:     tuple = (9, 6),
    savepath:    str | None = None,
) -> plt.Figure:
    """
    Scatter plot e profilo (media per bin) dello score in funzione di una
    variabile continua (tipicamente l'energia E1).

    Utile per verificare la correlazione residua tra score e l'energia
    dopo l'adversarial training.

    **Passa i dati già ritrasformati se hai usato uno scaling.**

    Parametri
    ----------
    variable   : array della variabile continua (es. E1)
    scores     : score del modello (predict_proba[:, 1])
    y_true     : label vere; se fornito colora i punti per classe
    threshold  : soglia di classificazione — traccia una linea orizzontale
    n_bins     : numero di bin per il profilo
    show_mean  : se True, sovrappone la media per bin (profilo)
    var_label  : etichetta asse x
    title      : titolo
    figsize    : dimensioni
    savepath   : path di salvataggio opzionale

    Restituisce
    -----------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if y_true is not None:
        ax.scatter(variable[y_true == 0], scores[y_true == 0],
                   s=3, alpha=0.2, color=_C["background"], label="Background", rasterized=True)
        ax.scatter(variable[y_true == 1], scores[y_true == 1],
                   s=3, alpha=0.2, color=_C["signal"],     label="Signal",     rasterized=True)
    else:
        ax.scatter(variable, scores, s=3, alpha=0.2, color=_C["neutral"], rasterized=True)

    if show_mean:
        bins        = np.linspace(variable.min(), variable.max(), n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_ids     = np.digitize(variable, bins)
        means       = []
        for i in range(1, len(bins)):
            mask = bin_ids == i
            means.append(scores[mask].mean() if mask.sum() > 0 else np.nan)
        ax.plot(bin_centers, means, color="black", linewidth=2.5,
                marker="o", markersize=4, label="Mean per bin")

    if threshold is not None:
        ax.axhline(threshold, linestyle=":", linewidth=1.5,
                   color=_C["threshold"], label=f"threshold={threshold:.3f}")

    ax.set_xlabel(var_label)
    ax.set_ylabel("Classifier score")
    ax.set_title(title)
    ax.legend(markerscale=3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ===========================================================================
# Esempio d'uso
# ===========================================================================
if __name__ == "__main__":
    import polars as pl
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    ENERGY_FEATURES = ["E_sum", "E_diff", "E_ratio", "E1"]
    DROP_ALWAYS     = ["i", "j", "target", "is_edge", "label1", "label2"]

    def split_xy(df, drop_extra=None):
        drop_cols = DROP_ALWAYS.copy()
        if drop_extra is not None:
            drop_cols += drop_extra
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df["target"]
        return X, y

    train = pl.read_parquet("full_train.parquet").to_pandas()
    val   = pl.read_parquet("full_val.parquet").to_pandas()
    test  = pl.read_parquet("full_test.parquet").to_pandas()

    X_train, y_train = split_xy(train, drop_extra=ENERGY_FEATURES)
    X_val,   y_val   = split_xy(val,   drop_extra=ENERGY_FEATURES)
    X_test,  y_test  = split_xy(test,  drop_extra=ENERGY_FEATURES)

    # modello di esempio
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    scores_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (scores_test > 0.5).astype(int)

    # ── 1. Training curves ──────────────────────────────────────────────────
    plot_training_curves(
        model.evals_result(),
        metric="logloss",
        best_iteration=model.best_iteration,
    )

    # ── 2. Feature importance ───────────────────────────────────────────────
    plot_feature_importance(model.feature_importances_, X_train.columns)

    # ── 3. Score distribution ───────────────────────────────────────────────
    plot_score_distribution(scores_test, y_test.values, thresholds=[0.5])

    # ── 4. ROC curve ────────────────────────────────────────────────────────
    plot_roc_curve(y_test.values, scores_test)

    # ── 5. Precision-Recall curve ───────────────────────────────────────────
    plot_pr_curve(y_test.values, scores_test)

    # ── 6. Threshold scan ───────────────────────────────────────────────────
    _, best_thr = plot_threshold_scan(y_test.values, scores_test)

    # ── 7. Confusion matrix ─────────────────────────────────────────────────
    plot_confusion_matrix(y_test.values, y_pred_test, threshold=0.5)

    # ── 8. Energy spectra ───────────────────────────────────────────────────
    # NOTA: E1 non è nel training set, la prendo direttamente dal DataFrame
    # Se avessi applicato un trasformazione, qui passeresti E1_back_transformed
    E_true     = test.loc[test["label1"] == 1, "E1"].values
    E_selected = test.loc[scores_test > best_thr, "E1"].values
    E_tp       = test.loc[(test["label1"] == 1) & (scores_test > best_thr), "E1"].values

    plot_energy_spectra(
        E_true     = E_true,
        E_selected = E_selected,
        E_tp       = E_tp,
        log_y      = True,
        x_label    = "Prompt energy E1",
    )

    # ── 9. Efficiency ───────────────────────────────────────────────────────
    plot_efficiency(E_true, E_tp, bins=30, x_label="Prompt energy E1")

    # ── 10. Score vs E1 (correlazione residua) ─────────────────────────────
    plot_score_vs_variable(
        variable  = test["E1"].values,
        scores    = scores_test,
        y_true    = y_test.values,
        threshold = best_thr,
        var_label = "Prompt energy E1",
        title     = "Residual correlation: score vs E1",
    )

    plt.show()
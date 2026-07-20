"""
build_pairs.py
==============
Model-independent construction of event-pair datasets, in two modes that
share the **same feature engineering** (the full, rich feature set defined
in ``dataset_setup.ipynb`` with identical transformations):

1. **Accidental background** (``generate_background_pairs``) -- pairs are
   built with a large temporal ``shift`` so that events far apart in real
   time get their time separation relabelled into the coincidence window.
   This destroys any time correlation and yields a data-driven estimate of
   the accidental-background contamination (used for the post-training
   test / background subtraction).

2. **Real coincidence candidates** (``build_real_pairs``) -- the *same*
   construction with ``shift = 0``: a pair ``(i, j>i)`` is kept whenever
   its *true* time separation ``times[j] - times[i]`` falls inside the
   physical window ``(0, max_dt)``.  These are the actual pairs on which a
   trained classifier is run at **inference** time (no labels, no shift).

Rationale
---------
The accidental-background contamination of a selected sample can be
estimated data-drivenly by pairing events that are far apart in real time
(applying a large negative ``shift`` to the inter-event time difference).
The genuine coincidence candidates are obtained from exactly the same
kernel with ``shift = 0``; nothing else changes, so a network trained on
the ``dataset_setup`` features can be applied directly to both.

This module is deliberately **model-agnostic**: it only produces feature
DataFrames.  Applying a scaler + classifier, building the subtracted
spectrum, or running inference is left to the calling code so that any
model (XGBoost, NN, ...) can reuse it.

Exported objects
----------------
FEATURE_COLUMNS            : ordered list of the produced columns
DEFAULT_TRANSFORM_CONFIG   : per-feature transformation map (matches
                             dataset_setup.ipynb)
compute_detector_geometry  : R_DET and Z_MAX from the raw events
apply_feature_transforms   : apply the transformation map to a DataFrame
build_shifted_pairs_df     : raw (untransformed) pairs for one shift
generate_background_pairs  : produce one (transformed) df per shift
build_real_pairs           : genuine coincidence candidates (shift = 0)
estimate_background_spectrum : average model-selected spectra over shifts
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from numba import njit


# ======================================================================
# Physical constants (identical to dataset_setup.ipynb)
# ======================================================================

_C = 299792458.0          # speed of light [m/s]

# Output column order -- must match the radio/pairs DataFrame of
# dataset_setup.ipynb so that the same DROP_COLS selection applies.
FEATURE_COLUMNS: list[str] = [
    "p_time", "Ep",
    "d_time", "Ed",
    "delta_x", "delta_y", "delta_z",
    "zp", "zd",
    "edist",
    "rp", "rd",
    "dist_wall_p", "dist_wall_d",
    "dist_wall_h_p", "dist_wall_h_d",
    "edist_over_r",
    "theta_p", "theta_d",
    "delta_r",
    "delta_theta",
    "delta_t",
    "diffusion",
    "E_sum", "E_diff", "E_div",
    "E_asym",
    "ds2",
    "is_event",
]

# Per-feature transformations -- identical to the ``config`` dict in
# dataset_setup.ipynb.  Columns not listed are left unchanged.
DEFAULT_TRANSFORM_CONFIG: dict[str, Callable] = {
    "delta_x":       np.sqrt,
    "delta_y":       np.sqrt,
    "delta_z":       np.sqrt,
    "edist":         np.sqrt,
    "edist_over_r":  np.log10,
    "dist_wall_p":   np.sqrt,
    "dist_wall_d":   np.sqrt,
    "dist_wall_h_p": np.sqrt,
    "dist_wall_h_d": np.sqrt,
    "delta_r":       np.sqrt,
    "delta_t":       np.sqrt,
    "diffusion":     np.log10,
    "E_diff":        np.sqrt,
    "E_div":         np.log,
    "ds2":           lambda x: np.log10(x + 5),
}


# ======================================================================
# Detector geometry
# ======================================================================

def compute_detector_geometry(
    events: pd.DataFrame | np.ndarray,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
) -> tuple[float, float]:
    """
    Compute R_DET (max cylindrical radius) and Z_MAX (max |z|) from the
    single-event table, exactly as in dataset_setup.ipynb.

    Parameters
    ----------
    events : DataFrame with columns ``x``, ``y``, ``z`` (or an array with
             those quantities in columns 2, 3, 4 -- see ``raw_data``
             convention below).
    """
    if isinstance(events, pd.DataFrame):
        x = events[x_col].to_numpy(dtype=np.float64)
        y = events[y_col].to_numpy(dtype=np.float64)
        z = events[z_col].to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(events, dtype=np.float64)
        x, y, z = arr[:, 2], arr[:, 3], arr[:, 4]

    r_det = float(np.max(np.sqrt(x ** 2 + y ** 2)))
    z_max = float(np.max(np.abs(z)))
    return r_det, z_max


# ======================================================================
# Numba kernel -- build shifted pairs with the full feature set
# ======================================================================

@njit
def _build_shifted_pairs_arrays(
    times, energy, x, y, z,
    R_DET, Z_MAX, shift, max_dt,
):
    """
    Two-pass (count then fill) construction of pairs.

    A pair (i, j>i) is kept when its *shifted* time difference
    ``aux_dt = (times[j] - times[i]) + shift`` lies in ``(0, max_dt)``.
    All time-dependent features (delta_t, diffusion, ds2) use ``aux_dt``,
    i.e. the pair is treated as a coincidence with that separation.

    With ``shift = 0`` this reduces to the *genuine* coincidence pairs
    (real time separation inside the window); with a large offset it
    produces accidental-background pairs.

    Returns a tuple of 1-D arrays, one per column in FEATURE_COLUMNS.
    """
    N = len(times)

    # ---- Pass 1: count ----------------------------------------------
    n_pairs = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            aux_dt = (times[j] - times[i]) + shift
            if aux_dt > max_dt:
                break
            if 0.0 < aux_dt < max_dt:
                n_pairs += 1

    # ---- Allocate ----------------------------------------------------
    p_time = np.empty(n_pairs, dtype=np.float64)
    Ep     = np.empty(n_pairs, dtype=np.float64)
    d_time = np.empty(n_pairs, dtype=np.float64)
    Ed     = np.empty(n_pairs, dtype=np.float64)

    delta_x = np.empty(n_pairs, dtype=np.float64)
    delta_y = np.empty(n_pairs, dtype=np.float64)
    delta_z = np.empty(n_pairs, dtype=np.float64)

    zp = np.empty(n_pairs, dtype=np.float64)
    zd = np.empty(n_pairs, dtype=np.float64)

    edist = np.empty(n_pairs, dtype=np.float64)

    rp = np.empty(n_pairs, dtype=np.float64)
    rd = np.empty(n_pairs, dtype=np.float64)

    dist_wall_p = np.empty(n_pairs, dtype=np.float64)
    dist_wall_d = np.empty(n_pairs, dtype=np.float64)
    dist_wall_h_p = np.empty(n_pairs, dtype=np.float64)
    dist_wall_h_d = np.empty(n_pairs, dtype=np.float64)

    edist_over_r = np.empty(n_pairs, dtype=np.float64)

    theta_p = np.empty(n_pairs, dtype=np.float64)
    theta_d = np.empty(n_pairs, dtype=np.float64)

    delta_r = np.empty(n_pairs, dtype=np.float64)
    delta_theta = np.empty(n_pairs, dtype=np.float64)

    delta_t = np.empty(n_pairs, dtype=np.float64)
    diffusion = np.empty(n_pairs, dtype=np.float64)

    E_sum = np.empty(n_pairs, dtype=np.float64)
    E_diff = np.empty(n_pairs, dtype=np.float64)
    E_div = np.empty(n_pairs, dtype=np.float64)
    E_asym = np.empty(n_pairs, dtype=np.float64)

    ds2 = np.empty(n_pairs, dtype=np.float64)
    is_event = np.zeros(n_pairs, dtype=np.float64)

    # ---- Pass 2: fill ------------------------------------------------
    idx = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            aux_dt = (times[j] - times[i]) + shift
            if aux_dt > max_dt:
                break
            if aux_dt <= 0.0 or aux_dt >= max_dt:
                continue

            dx = np.abs(x[j] - x[i])
            dy = np.abs(y[j] - y[i])
            dz = np.abs(z[j] - z[i])

            dist = np.sqrt((x[j] - x[i]) ** 2 +
                           (y[j] - y[i]) ** 2 +
                           (z[j] - z[i]) ** 2)

            rp_val = np.sqrt(x[i] ** 2 + y[i] ** 2)
            rd_val = np.sqrt(x[j] ** 2 + y[j] ** 2)

            theta_p_val = np.arctan2(x[i], y[i])
            theta_d_val = np.arctan2(x[j], y[j])
            dtheta = np.arctan2(np.sin(theta_d_val - theta_p_val),
                                np.cos(theta_d_val - theta_p_val))

            p_time[idx] = times[i]
            d_time[idx] = times[j]
            Ep[idx] = energy[i]
            Ed[idx] = energy[j]

            delta_x[idx] = dx
            delta_y[idx] = dy
            delta_z[idx] = dz

            zp[idx] = z[i]
            zd[idx] = z[j]

            edist[idx] = dist

            rp[idx] = rp_val
            rd[idx] = rd_val

            dist_wall_p[idx] = R_DET - rp_val
            dist_wall_d[idx] = R_DET - rd_val
            dist_wall_h_p[idx] = Z_MAX - np.abs(z[i])
            dist_wall_h_d[idx] = Z_MAX - np.abs(z[j])

            edist_over_r[idx] = dist / (rp_val + rd_val)

            theta_p[idx] = theta_p_val
            theta_d[idx] = theta_d_val

            delta_r[idx] = np.abs(rd_val - rp_val)
            delta_theta[idx] = dtheta

            # Time-dependent features use the (possibly shifted) dt
            delta_t[idx] = aux_dt
            diffusion[idx] = (dist ** 2) / (aux_dt + 1e-8)

            E_sum[idx] = energy[j] + energy[i]
            E_diff[idx] = np.abs(energy[j] - energy[i])
            E_div[idx] = energy[j] / energy[i]
            E_asym[idx] = (energy[j] - energy[i]) / (energy[j] + energy[i])

            ds2[idx] = (_C * aux_dt * 1e-9) ** 2 - dist ** 2

            idx += 1

    return (
        p_time, Ep, d_time, Ed,
        delta_x, delta_y, delta_z,
        zp, zd,
        edist,
        rp, rd,
        dist_wall_p, dist_wall_d,
        dist_wall_h_p, dist_wall_h_d,
        edist_over_r,
        theta_p, theta_d,
        delta_r, delta_theta,
        delta_t, diffusion,
        E_sum, E_diff, E_div, E_asym,
        ds2, is_event,
    )


# ======================================================================
# DataFrame builders
# ======================================================================

def _raw_to_arrays(raw_data) -> tuple[np.ndarray, ...]:
    """
    Normalize the raw single-event input into (times, energy, x, y, z).

    Accepts either a DataFrame with columns
    ``["time", "energy", "x", "y", "z"]`` or a 2-D array whose first five
    columns are in that order.
    """
    if isinstance(raw_data, pd.DataFrame):
        times  = raw_data["time"].to_numpy(dtype=np.float64)
        energy = raw_data["energy"].to_numpy(dtype=np.float64)
        x = raw_data["x"].to_numpy(dtype=np.float64)
        y = raw_data["y"].to_numpy(dtype=np.float64)
        z = raw_data["z"].to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(raw_data, dtype=np.float64)
        times, energy, x, y, z = (arr[:, 0], arr[:, 1],
                                  arr[:, 2], arr[:, 3], arr[:, 4])
    return times, energy, x, y, z


def build_shifted_pairs_df(
    raw_data,
    shift: float,
    R_DET: float,
    Z_MAX: float,
    max_dt: float = 2e6,
) -> pd.DataFrame:
    """
    Build the (untransformed) pair DataFrame for a single shift.

    Parameters
    ----------
    raw_data : single-event table (DataFrame or array, see _raw_to_arrays).
               **Must be sorted by time.**
    shift    : temporal offset added to every inter-event dt (same units
               as the time column, e.g. ns).  ``shift = 0`` gives the
               genuine coincidence pairs.
    R_DET    : detector radius (from ``compute_detector_geometry``).
    Z_MAX    : detector half-height.
    max_dt   : coincidence window upper edge.

    Returns
    -------
    DataFrame with columns == FEATURE_COLUMNS (raw, before transforms).
    """
    times, energy, x, y, z = _raw_to_arrays(raw_data)
    cols = _build_shifted_pairs_arrays(
        times, energy, x, y, z,
        float(R_DET), float(Z_MAX), float(shift), float(max_dt),
    )
    return pd.DataFrame(dict(zip(FEATURE_COLUMNS, cols)))


def apply_feature_transforms(
    df: pd.DataFrame,
    config: dict[str, Callable] | None = None,
) -> pd.DataFrame:
    """
    Apply the per-feature transformations (sqrt / log10 / log ...).

    Equivalent to ``build_transformed_df`` in dataset_setup.ipynb:
    columns present in ``config`` are transformed in place, the rest are
    left unchanged.  Returns a transformed copy.
    """
    if config is None:
        config = DEFAULT_TRANSFORM_CONFIG
    out = df.copy()
    for col, fn in config.items():
        if col in out.columns:
            out[col] = fn(out[col])
    return out


def generate_background_pairs(
    raw_data,
    shifts: Sequence[float],
    R_DET: float | None = None,
    Z_MAX: float | None = None,
    max_dt: float = 2e6,
    apply_transforms: bool = True,
    transform_config: dict[str, Callable] | None = None,
) -> tuple[list[pd.DataFrame], list[float]]:
    """
    Generate one **accidental-background** DataFrame per temporal shift,
    with the full feature set and (optionally) the dataset_setup
    transformations.

    Parameters
    ----------
    raw_data         : single-event table (DataFrame or array), sorted by
                       time, columns ``[time, energy, x, y, z]``.
    shifts           : iterable of temporal offsets.  Large offsets
                       (e.g. ``1e6 * np.arange(-20, -10)``) give pure
                       accidental pairs.
    R_DET, Z_MAX     : detector geometry.  If ``None`` they are computed
                       from ``raw_data`` via ``compute_detector_geometry``.
    max_dt           : coincidence window upper edge.
    apply_transforms : if True, apply ``transform_config`` to each df.
    transform_config : transformation map (defaults to
                       DEFAULT_TRANSFORM_CONFIG, matching dataset_setup).

    Returns
    -------
    bg_dfs : list of DataFrames (one per shift), columns == FEATURE_COLUMNS,
             transformed if requested.  The caller then drops the unused
             columns (DROP_COLS) and keeps ``Ep`` for the spectrum.  Here
             ``is_event`` is a genuine label (= 0, background).
    shifts : the shift values used (echoed back).

    Example
    -------
    >>> R_DET, Z_MAX = compute_detector_geometry(raw_events)
    >>> bg_dfs, shifts = generate_background_pairs(
    ...     raw_events,
    ...     shifts=1e6 * np.arange(-20, -10),
    ...     R_DET=R_DET, Z_MAX=Z_MAX,
    ... )
    """
    if R_DET is None or Z_MAX is None:
        R_DET, Z_MAX = compute_detector_geometry(raw_data)

    bg_dfs: list[pd.DataFrame] = []
    shifts_list = list(shifts)

    for s in shifts_list:
        df = build_shifted_pairs_df(raw_data, s, R_DET, Z_MAX, max_dt)
        if apply_transforms:
            df = apply_feature_transforms(df, transform_config)
        bg_dfs.append(df)

    return bg_dfs, shifts_list


def build_real_pairs(
    raw_data,
    R_DET: float | None = None,
    Z_MAX: float | None = None,
    max_dt: float = 2e6,
    apply_transforms: bool = True,
    transform_config: dict[str, Callable] | None = None,
    keep_label_column: bool = False,
) -> pd.DataFrame:
    """
    Build the **genuine coincidence-candidate** pairs (no temporal shift),
    i.e. the *real* pairs used at inference time.

    This is the same construction as ``generate_background_pairs`` but with
    ``shift = 0``: a pair ``(i, j>i)`` is kept whenever its *true* time
    separation ``times[j] - times[i]`` falls inside ``(0, max_dt)``.  No
    time correlation is destroyed, so these are the actual coincidence
    candidates on which the trained classifier is run.  The feature
    engineering and transformations are identical to the training data, so
    the same scaler + model apply directly.

    Because inference data is unlabelled, the ``is_event`` column produced
    by the kernel carries **no meaning** here (it is a placeholder, not a
    label) and is dropped by default.  Set ``keep_label_column=True`` only
    if you want a column-for-column match with the background DataFrames.

    Parameters
    ----------
    raw_data         : single-event table (DataFrame or array), sorted by
                       time, columns ``[time, energy, x, y, z]``.
    R_DET, Z_MAX     : detector geometry.  If ``None`` they are computed
                       from ``raw_data`` via ``compute_detector_geometry``.
    max_dt           : coincidence window upper edge -- use the *same*
                       physical window as the training pairs.
    apply_transforms : if True, apply ``transform_config`` to the df.
    transform_config : transformation map (defaults to
                       DEFAULT_TRANSFORM_CONFIG, matching dataset_setup).
    keep_label_column: keep the dummy ``is_event`` column (default False).

    Returns
    -------
    DataFrame of real coincidence pairs, with the same feature columns and
    transformations as the training/background data.  The caller then drops
    the unused columns (DROP_COLS), scales, and runs ``model.predict``.

    Example
    -------
    >>> R_DET, Z_MAX = compute_detector_geometry(raw_events)
    >>> pairs = build_real_pairs(raw_events, R_DET=R_DET, Z_MAX=Z_MAX)
    >>> X_infer = pairs[feature_cols]          # same columns as training
    >>> X_infer = scaler.transform(X_infer)
    >>> y_score = model.predict_proba(X_infer)[:, 1]
    """
    if R_DET is None or Z_MAX is None:
        R_DET, Z_MAX = compute_detector_geometry(raw_data)

    df = build_shifted_pairs_df(raw_data, 0.0, R_DET, Z_MAX, max_dt)
    if apply_transforms:
        df = apply_feature_transforms(df, transform_config)
    if not keep_label_column and "is_event" in df.columns:
        df = df.drop(columns="is_event")
    return df


# ======================================================================
# Spectrum estimation
# ======================================================================

def estimate_background_spectrum(
    bg_dfs: list[pd.DataFrame],
    selected_masks: list[np.ndarray],
    bins: np.ndarray,
    energy_col: str = "Ep",
    norm_factor: float = 1.0,
) -> np.ndarray:
    """
    Average the energy histogram of model-selected background events over
    all shifts and apply a normalization factor.

    Reproduces the workflow_ordinato.ipynb estimate:
        histo_background = norm_factor * mean_over_shifts( hist(selected Ep) )

    Parameters
    ----------
    bg_dfs         : list of background DataFrames (from
                     ``generate_background_pairs``).
    selected_masks : list of boolean masks (one per df) marking the pairs
                     the classifier selected as signal.
    bins           : histogram bin edges (shared with the signal spectra).
    energy_col     : prompt-energy column name (``"Ep"``).
    norm_factor    : multiplicative normalization (e.g. the ratio of the
                     physical to the shifted time window -- 0.28 in
                     workflow_ordinato).

    Returns
    -------
    histo_background : 1-D array of averaged & normalized bin counts.
    """
    histograms = []
    for df, mask in zip(bg_dfs, selected_masks):
        h, _ = np.histogram(df.loc[mask, energy_col], bins=bins)
        histograms.append(h)

    histo_background = norm_factor * np.mean(histograms, axis=0)
    return histo_background

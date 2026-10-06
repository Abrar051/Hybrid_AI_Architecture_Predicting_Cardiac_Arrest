"""Shared evaluation utilities: labels, sequences, bootstrap CIs, alert replay.

Single source of truth for the protocol fixes of the rebuild:
  - horizon_labels applies the pre-arrest floor + post-arrest exclusion as NaN
    labels (plan rule 4) everywhere, so train and replay can never diverge
  - patient_bootstrap takes the patient IDs explicitly (the old notebook
    closure over 2 replay patients produced a degenerate CI)
  - true_alert counts an alert as true only when the arrest falls within the
    alert horizon (fixes the "first true alert 21.7 h" mislabel)
"""
import json

import numpy as np
import pandas as pd

from config import (CACHE_DIR, EMBED_DIR, HORIZONS, SPLITS_FILE,
                    SEQ_LEN_WINDOWS, TRAIN_STEP)
from ews_risk import ENG_COLS, VITAL_COLS, build_features_masked


# ---------- labels ----------------------------------------------------------

def horizon_labels(tte_h, horizons=HORIZONS, floor_h=0.25):
    """Per-window labels per horizon from hours-to-arrest.

    tte_h: (n,) hours to arrest (NaN for controls). Windows in the final
    floor_h before arrest and after arrest get NaN labels (excluded); metric
    helpers must select finite labels. Controls -> 0 for every horizon."""
    tte = np.asarray(tte_h, dtype=float)
    out = {}
    for H in horizons:
        lab = np.full(tte.shape, np.nan, dtype=float)
        lab[np.isnan(tte)] = 0.0                        # controls: negative
        lab[(tte > 0) & (tte <= floor_h)] = np.nan      # responding window
        lab[tte <= 0] = np.nan                          # post-arrest
        lab[(tte > floor_h) & (tte <= H)] = 1.0
        lab[tte > H] = 0.0
        out[H] = lab
    return out


def finite_mask(labels, horizons=HORIZONS):
    """Boolean mask of windows with finite (non-excluded) labels at every
    horizon."""
    m = np.ones(len(next(iter(labels.values()))), bool)
    for H in horizons:
        m &= np.isfinite(labels[H])
    return m


# ---------- sequences -------------------------------------------------------

def make_sequences(meta, feats, labels, tte, step=TRAIN_STEP,
                   seq_len=SEQ_LEN_WINDOWS, ids=None, masks=None):
    """Sliding per-patient sequences (left-padded to seq_len), every `step`-th
    window a target. Returns (X, L, Y, T) plus ID / M when ids/masks given."""
    meta = meta.reset_index(drop=True)
    X, L, Y, T, ID, M = [], [], [], [], [], []
    for _, g in meta.groupby("patient_id"):
        pos = g.index.to_numpy()
        for j in range(0, len(pos), step):
            lo = max(0, j - seq_len + 1)
            seq = feats[pos[lo:j + 1]]
            X.append(np.pad(seq, ((seq_len - len(seq), 0), (0, 0))))
            L.append(len(seq))
            Y.append([labels[H][pos[j]] for H in labels])
            T.append(tte[pos[j]])
            if ids is not None:
                ID.append(ids[pos[j]])
            if masks is not None:
                M.append(np.pad(masks[pos[lo:j + 1]], ((seq_len - len(seq), 0), (0, 0))))
    out = (np.stack(X).astype(np.float32), np.array(L, np.int64),
           {H: np.array([y[k] for y in Y], np.float32)
            for k, H in enumerate(labels)},
           np.array(T, np.float32))
    if ids is not None:
        out += (np.array(ID, np.int64),)
    if masks is not None:
        out += (np.stack(M).astype(np.float32),)
    return out


def make_sequences_memmap(meta, feats, labels, tte, cache_f, step=TRAIN_STEP,
                          seq_len=SEQ_LEN_WINDOWS, masks=None):
    """RAM-safe make_sequences: writes X (and M when masks is given) row by
    row into preallocated float32 memmaps (np.stack would need ~2x the tensor
    size transiently)."""
    meta = meta.reset_index(drop=True)
    n_seq = sum(len(range(0, len(g), step)) for _, g in meta.groupby("patient_id"))
    X = np.lib.format.open_memmap(cache_f, mode="w+", dtype=np.float32,
                                  shape=(n_seq, seq_len, feats.shape[1]))
    Mm = None
    if masks is not None:
        Mm = np.lib.format.open_memmap(str(cache_f) + ".masks", mode="w+",
                                       dtype=np.float32,
                                       shape=(n_seq, seq_len, masks.shape[1]))
    L = np.zeros(n_seq, np.int64)
    T = np.zeros(n_seq, np.float32)
    Y = {H: np.zeros(n_seq, np.float32) for H in labels}
    k = 0
    for _, g in meta.groupby("patient_id"):
        pos = g.index.to_numpy()
        for j in range(0, len(pos), step):
            lo = max(0, j - seq_len + 1)
            n_hist = j - lo + 1
            X[k, seq_len - n_hist:] = feats[pos[lo:j + 1]]
            if Mm is not None:
                Mm[k, seq_len - n_hist:] = masks[pos[lo:j + 1]]
            L[k] = n_hist
            for H in labels:
                Y[H][k] = labels[H][pos[j]]
            T[k] = tte[pos[j]]
            k += 1
    assert k == n_seq, "sequence count mismatch"
    X.flush()
    if Mm is not None:
        Mm.flush()
        return X, L, Y, T, Mm
    return X, L, Y, T


def sequence_ids(meta, ids=None, step=TRAIN_STEP):
    """Per-sequence patient codes matching make_sequences[_memmap] output
    order (ids defaults to factorized patient codes of meta)."""
    meta = meta.reset_index(drop=True)
    if ids is None:
        ids = pd.factorize(meta["patient_id"])[0]
    out = []
    for _, g in meta.groupby("patient_id"):
        pos = g.index.to_numpy()
        for j in range(0, len(pos), step):
            out.append(ids[pos[j]])
    return np.array(out, np.int64)


# ---------- metrics ---------------------------------------------------------

def patient_bootstrap(metric, y, p, patients, n_iter=1000, seed=42):
    """Patient-level bootstrap CI (plan rule 8). Resamples patient IDs with
    replacement; draws missing either class are skipped. Returns
    (point, lo, hi, n_valid) where point is the metric on all data."""
    patients = np.asarray(patients)
    rs = np.random.default_rng(seed)
    point = metric(y, p)
    stats = []
    for _ in range(n_iter):
        pids = rs.choice(np.unique(patients), size=len(np.unique(patients)),
                         replace=True)
        m = np.isin(patients, pids)
        yb, pb = y[m], p[m]
        if yb.sum() > 0 and (yb == 0).sum() > 0:
            stats.append(metric(yb, pb))
    if not stats:
        return float(point), float("nan"), float("nan"), 0
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(point), float(lo), float(hi), len(stats)


# ---------- alert replay ----------------------------------------------------

def true_alert(t_alert, arrest_h, horizon=6):
    """An alert at time t_alert (hours from stay start) is TRUE for the
    horizon-h task when the arrest falls inside the alert window:
    0 < arrest_h - t_alert <= horizon."""
    return bool((arrest_h - t_alert > 0) and (arrest_h - t_alert <= horizon))


def alert_from_risks(t_h, risk6, arrest_h, threshold=0.5, refractory_h=1.0):
    """Alert flags from a per-window risk series (identical semantics to
    ews_risk.simulate_patient but vectorized over one patient's stay).

    Returns (alert_bool, true_bool, first_true_lead) with true_bool restricted
    to alerts whose arrest falls within the 6 h horizon."""
    last = -np.inf
    alert = np.zeros(len(t_h), bool)
    for i, (t, r) in enumerate(zip(t_h, risk6)):
        if r >= threshold and (t - last) >= refractory_h:
            alert[i] = True
            last = t
    true = np.zeros(len(t_h), bool)
    if np.isfinite(arrest_h):
        for i in np.where(alert)[0]:
            true[i] = true_alert(t_h[i], arrest_h)
    first_true_lead = None
    if true.any():
        first_true_lead = arrest_h - t_h[np.where(true)[0][0]]
    return alert, true, first_true_lead


# ---------- loading ---------------------------------------------------------

def load_splits():
    """Split pids from cache/splits.json (written by build_manifests)."""
    return json.loads(SPLITS_FILE.read_text())


def load_manifest_features(tag):
    """Load a manifest's meta, engineered features (with 30-min trends already
    joined by build_manifests from the FULL vitals table), window embeddings,
    and per-window modality masks. Returns (meta, feats, masks, labels, tte,
    ids) with labels from horizon_labels."""
    meta = pd.read_csv(EMBED_DIR / f"{tag}_meta.csv")
    eng = pd.read_parquet(EMBED_DIR / f"{tag}_engineered.parquet")
    ecg = np.load(EMBED_DIR / f"{tag}_window_ecgfm.npy")
    ppg = np.load(EMBED_DIR / f"{tag}_window_papagei.npy")
    assert len(meta) == len(ecg) == len(ppg) == len(eng), \
        f"{tag} cached arrays out of sync"
    feats = np.stack([build_features_masked(meta.iloc[i],
                                            eng[ENG_COLS].iloc[i].astype(float),
                                            ecg[i], ppg[i])[0]
                      for i in range(len(meta))]).astype(np.float32)
    masks = np.ones((len(meta), 4), np.float32)
    tte = (meta["arrest_h"] - meta["t_end_h"]).to_numpy()
    labels = horizon_labels(tte)
    ids = pd.factorize(meta["patient_id"])[0]
    return meta, feats, masks, labels, tte, ids


def iter_test_patients(meta, feats, masks, labels, tte):
    """Yield (pid, is_case, arrest_h, idx, X, L, Y, T, M) per test patient
    (step=1 sequences) for streaming evaluation."""
    for pid, g in meta.groupby("patient_id"):
        idx = g.index.to_numpy()
        X, L, Y, T, M = make_sequences(meta.loc[idx], feats[idx],
                                       {H: labels[H][idx] for H in HORIZONS},
                                       tte[idx], step=1, masks=masks[idx])
        yield (int(pid), bool(g["is_case"].iloc[0]),
               float(g["arrest_h"].iloc[0]), idx, X, L, Y, T, M)
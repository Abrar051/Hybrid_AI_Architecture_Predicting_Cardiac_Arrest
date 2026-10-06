"""Selftest for eval_utils (plan rule 7: test helpers on synthetic before use).

Covers: horizon_labels floor/exclusion semantics, true_alert boundaries,
patient_bootstrap both-classes guard, memmap sequence round-trip.

Usage: python scripts/test_eval_utils.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from eval_utils import (horizon_labels, true_alert, patient_bootstrap,  # noqa: E402
                        make_sequences, make_sequences_memmap)
from config import HORIZONS  # noqa: E402


def main():
    # ---- 1. horizon_labels -------------------------------------------------
    tte = np.array([np.nan, 0.5, 0.1, -1.0, 3.0, 7.0])     # ctrl, pos1h, resp, post, pos6h, neg
    lab = horizon_labels(tte, horizons=(1, 6), floor_h=0.25)
    assert np.allclose(lab[1], [0, 1, np.nan, np.nan, 0, 0], equal_nan=True), lab[1]
    assert np.allclose(lab[6], [0, 1, np.nan, np.nan, 1, 0], equal_nan=True), lab[6]
    # floor boundary: exactly 0.25 h is excluded
    tte2 = np.array([0.25, 0.25001])
    lab2 = horizon_labels(tte2, horizons=(1,), floor_h=0.25)
    assert np.isnan(lab2[1][0]) and lab2[1][1] == 1.0
    # the reviewer's arithmetic: 12 windows/hour, final 15 min (3 windows)
    # excluded -> 1 h positives drop from 12 to 9
    tte3 = np.arange(12, 0, -1) / 12.0                      # tte 1.0 .. 0.083 h
    lab3 = horizon_labels(tte3, horizons=(1,), floor_h=0.25)
    assert int(np.nansum(lab3[1])) == 9, lab3[1]
    print("PASS 1: horizon_labels floor + exclusion semantics")

    # ---- 2. true_alert ------------------------------------------------------
    arrest = 25.8
    assert not true_alert(4.2, arrest, 6)      # 21.6 h out: FALSE for 6 h horizon
    assert true_alert(20.5, arrest, 6)         # 5.3 h out: true
    assert not true_alert(25.9, arrest, 6)     # after arrest
    assert not true_alert(19.79, arrest, 6)    # 6.01 h out: just outside
    assert true_alert(19.8, arrest, 6)         # 6.0 h out: boundary inclusive
    print("PASS 2: true_alert boundaries (the 21.7 h 'true alert' is now false)")

    # ---- 3. patient_bootstrap ----------------------------------------------
    rs = np.random.default_rng(0)
    n_patients = 40
    pats = np.repeat(np.arange(n_patients), 20)
    p = rs.uniform(0.3, 0.9, len(pats))
    y = (rs.uniform(0, 1, len(pats)) < p).astype(int)
    y[pats < 10] = 1                                        # guarantee both classes
    point, lo, hi, n_valid = patient_bootstrap(roc_auc_score, y, p, pats,
                                               n_iter=200, seed=42)
    assert np.isfinite(point) and np.isfinite(lo) and np.isfinite(hi)
    assert lo <= point <= hi, f"point {point} outside CI [{lo}, {hi}]"
    assert n_valid > 100, f"too many skipped draws: {n_valid}"
    # degenerate case: 2 patients, one per class -> single-value CI
    p2, lo2, hi2, nv2 = patient_bootstrap(roc_auc_score, np.array([0, 1]),
                                          np.array([0.1, 0.9]),
                                          np.array([0, 1]), n_iter=50)
    assert lo2 == hi2 == p2, f"2-patient bootstrap should be degenerate"
    print("PASS 3: patient_bootstrap valid on 40 patients, degenerate on 2 "
          "(the old notebook's printed CI could not have come from its code)")

    # ---- 4. memmap sequences -----------------------------------------------
    meta = pd.DataFrame({
        "patient_id": [0, 0, 0, 0, 1, 1, 1, 1],
        "t_start_h": np.arange(8) * 5.0 / 60.0})
    feats = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    labels = {H: np.array([0, 0, 1, 1, 0, 0, 0, 0], np.float32) for H in (1, 6)}
    tte = np.array([3.0, 2.0, 1.0, 0.5, np.nan, np.nan, np.nan, np.nan])
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "seq.npy"
        Xm, Lm, Ym, Tm = make_sequences_memmap(meta, feats, labels, tte,
                                               cache_f=f, step=2, seq_len=3)
        Xs, Ls, Ys, Ts = make_sequences(meta, feats, labels, tte,
                                        step=2, seq_len=3)
        assert Xm.shape == Xs.shape
        assert np.allclose(np.asarray(Xm), Xs)
        assert (Lm == Ls).all()
        for H in (1, 6):
            assert np.allclose(Ym[H], Ys[H])
        assert np.allclose(Tm, Ts, equal_nan=True)
    print("PASS 4: memmap sequences round-trip matches make_sequences")

    print("\nPASS: all eval_utils selftests")


if __name__ == "__main__":
    main()
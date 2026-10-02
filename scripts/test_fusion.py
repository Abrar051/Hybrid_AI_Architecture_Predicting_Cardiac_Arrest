"""Synthetic test for multimodal fusion variants + missing-modality masks.

Plan rule 7 (synthetic before real data) for the fusion upgrade:
  concat - legacy single Linear projection (control, v1 architecture)
  gate   - hierarchical gated fusion of per-block projections
  attn   - per-window modality attention weights
  weight - static learned per-modality weights

Masks: gate/attn/weight route blocks whose mask bit is 0 to a learned
per-modality missing embedding. Two mask experiments on the synthetic cohort:
  1. training with 10% of windows' PPG masked (simulated missing stream)
  2. evaluation with all PPG masked on the validation patients - predictions
     must stay finite and the 6 h AUC should not collapse (synthetic PPG
     carries little signal, so a modest drop is expected)

Usage: python scripts/test_fusion.py [--epochs 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from ews_risk import FEAN, SEQ_LEN, build_features_masked, train_fean  # noqa: E402
from test_fean_v2 import load_synthetic, make_sequences  # noqa: E402

HORIZONS = (1, 6, 24)
SEED = 7
PPG_BLOCK = 3                      # index of the PPG block in the 4-bit mask


def make_sequences_masked(meta, feats, labels, tte, ids, masks, step=3):
    """Like make_sequences, but sequences also carry per-window masks
    (padded history positions get mask 0)."""
    meta = meta.reset_index(drop=True)
    X, L, Y, T, ID, M = [], [], [], [], [], []
    for _, g in meta.groupby("patient_id"):
        pos = g.index.to_numpy()
        for j in range(0, len(pos), step):
            lo = max(0, j - SEQ_LEN + 1)
            n_hist = j - lo + 1
            X.append(np.pad(feats[pos[lo:j + 1]], ((SEQ_LEN - n_hist, 0), (0, 0))))
            M.append(np.pad(masks[pos[lo:j + 1]], ((SEQ_LEN - n_hist, 0), (0, 0))))
            L.append(n_hist)
            Y.append([labels[H][pos[j]] for H in HORIZONS])
            T.append(tte[pos[j]])
            ID.append(ids[pos[j]])
    return (np.stack(X).astype(np.float32), np.array(L, np.int64),
            {H: np.array([y[k] for y in Y], np.float32) for k, H in enumerate(HORIZONS)},
            np.array(T, np.float32), np.array(ID, np.int64),
            np.stack(M).astype(np.float32))


def eval_auc(m, X, L, M, Y, masks_override=None):
    """6 h / 24 h AUROC with an optional per-sequence mask override."""
    Mm = M if masks_override is None else masks_override
    with torch.no_grad():
        out = m(torch.from_numpy(X), torch.from_numpy(L), mask=torch.from_numpy(Mm))
    auc = {H: roc_auc_score(Y[H], torch.sigmoid(out["risk"][H]).numpy())
           for H in HORIZONS if Y[H].sum() > 0 and (Y[H] == 0).sum() > 0}
    assert np.isfinite(torch.sigmoid(out["risk"][6]).numpy()).all(), \
        "non-finite predictions under mask override"
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    meta, feats, labels, tte, ids = load_synthetic()
    masks = np.ones((len(meta), 4), np.float32)      # all blocks present
    case_pids = sorted(set(meta.loc[meta["is_case"], "patient_id"]))
    ctrl_pids = sorted(set(meta.loc[~meta["is_case"], "patient_id"]))
    val_pids = {case_pids[-1], ctrl_pids[-1]}
    tr = ~meta.patient_id.isin(val_pids).to_numpy()
    va = ~tr
    assert not (set(meta.loc[tr, "patient_id"]) & set(val_pids))
    mu, sd = feats[tr].mean(0), feats[tr].std(0)
    z = (feats - mu) / (sd + 1e-8)

    # simulate 10% missing PPG on train windows (zeroed block + mask bit 0)
    rs = np.random.default_rng(SEED)
    missing_tr = np.zeros(len(meta), bool)
    missing_tr[np.where(tr)[0][rs.choice(int(tr.sum()), int(0.1 * tr.sum()),
                                         replace=False)]] = True
    z_ppgoff = z.copy()
    z_ppgoff[missing_tr, -512:] = 0.0
    masks_tr = masks.copy()
    masks_tr[missing_tr, PPG_BLOCK] = 0.0

    Xtr, Ltr, Ytr, Ttr, idtr, Mtr = make_sequences_masked(
        meta.loc[tr], z_ppgoff[tr], {H: labels[H][tr] for H in HORIZONS},
        tte[tr], ids[tr], masks_tr[tr])
    Xva, Lva, Yva, Tva, idva, Mva = make_sequences_masked(
        meta.loc[va], z[va], {H: labels[H][va] for H in HORIZONS},
        tte[va], ids[va], masks[va])

    print(f"train seqs {len(Xtr)} (10% PPG-masked), val seqs {len(Xva)} | seed {SEED}")
    torch.manual_seed(SEED)

    results = {}
    for mode in ("concat", "gate", "attn", "weight"):
        print(f"\n=== {mode} ({args.epochs} epochs, full batch) ===")
        m = FEAN(fusion=mode)
        if mode == "concat":
            m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr,
                                 Xva=Xva, Lva=Lva, Yva=Yva,
                                 epochs=args.epochs, seed=SEED)
        else:
            m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr, masks_tr=Mtr,
                                 Xva=Xva, Lva=Lva, Yva=Yva, masks_va=Mva,
                                 epochs=args.epochs, seed=SEED)
        assert all(np.isfinite(h["train_loss"]) for h in hist), f"NaN loss in {mode}"
        assert hist[-1]["train_loss"] < hist[0]["train_loss"], f"{mode} loss flat"
        auc_va = eval_auc(m, Xva, Lva, Mva, Yva)
        results[mode] = dict(val_auc=auc_va, last_train_loss=hist[-1]["train_loss"])
        print(f"  val AUROC: " + " | ".join(f"{H}h {auc_va[H]:.3f}" for H in auc_va))

    # robustness: all val PPG masked out (mask 0 + zeroed block) for the
    # mask-aware modes; concat keeps zeros (its legacy semantics)
    print("\nval AUROC with ALL PPG removed at evaluation (robustness):")
    for mode in ("gate", "attn", "weight"):
        Mva_off = Mva.copy()
        Mva_off[:, :, PPG_BLOCK] = 0.0
        Xva_off = Xva.copy()
        Xva_off[:, :, -512:] = 0.0
        torch.manual_seed(SEED)
        m = FEAN(fusion=mode)
        m, _ = train_fean(m, Xtr, Ltr, Ytr, Ttr, masks_tr=Mtr,
                          Xva=Xva, Lva=Lva, Yva=Yva, masks_va=Mva,
                          epochs=args.epochs, seed=SEED, verbose=0)
        auc_off = eval_auc(m, Xva_off, Lva, Mva_off, Yva)
        results[mode]["val_auc_ppg_off"] = auc_off
        print(f"  {mode:<6} " + " | ".join(f"{H}h {auc_off[H]:.3f}" for H in auc_off))

    print("\nSummary (val AUROC 6h):")
    for mode, r in results.items():
        extra = f" | PPG-off {r['val_auc_ppg_off'][6]:.3f}" if "val_auc_ppg_off" in r else ""
        print(f"  {mode:<6} {r['val_auc'][6]:.3f}{extra} | train_loss "
              f"{r['last_train_loss']:.4f}")

    print("\nPASS: all fusion modes train without NaNs; mask-aware modes stay "
          "finite with the PPG stream removed at evaluation time.")
    print("NOTE: on synthetic data the PPG embedding carries little signal, so "
          "fusion gains here are expected to be small; the real comparison is "
          "on SDDB, where PPG is entirely absent and the mask matters most.")


if __name__ == "__main__":
    main()
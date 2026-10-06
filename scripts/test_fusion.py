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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from ews_risk import FEAN, train_fean  # noqa: E402
from eval_utils import (load_manifest_features, load_splits,  # noqa: E402
                        make_sequences)

HORIZONS = (1, 6, 24)
SEED = 7
PPG_BLOCK = 3                      # index of the PPG block in the 4-bit mask
OUTPUT_DIR = PROJECT / "outputs"


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
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    meta, feats, masks, labels, tte, ids = load_manifest_features("dev")
    splits = load_splits()
    tr = meta.patient_id.isin(set(splits["train"])).to_numpy()
    va = meta.patient_id.isin(set(splits["val"])).to_numpy()
    assert tr.any() and va.any()
    assert not (set(meta.loc[tr, "patient_id"]) & set(splits["val"]))
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

    Xtr, Ltr, Ytr, Ttr, idtr, Mtr = make_sequences(
        meta.loc[tr], z_ppgoff[tr], {H: labels[H][tr] for H in HORIZONS},
        tte[tr], ids=ids[tr], masks=masks_tr[tr])
    Xva, Lva, Yva, Tva, idva, Mva = make_sequences(
        meta.loc[va], z[va], {H: labels[H][va] for H in HORIZONS},
        tte[va], ids=ids[va], masks=masks[va])

    print(f"train seqs {len(Xtr)} (10% PPG-masked), val seqs {len(Xva)} | "
          f"val patients {len(splits['val'])} | seed {SEED}")
    torch.manual_seed(SEED)

    results = {}
    for mode in ("concat", "gate", "attn", "weight"):
        print(f"\n=== {mode} ({args.epochs} epochs, batch {args.batch_size}) ===")
        m = FEAN(fusion=mode)
        if mode == "concat":
            m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr,
                                 Xva=Xva, Lva=Lva, Yva=Yva,
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 seed=SEED)
        else:
            m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr, masks_tr=Mtr,
                                 Xva=Xva, Lva=Lva, Yva=Yva, masks_va=Mva,
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 seed=SEED)
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
                          epochs=args.epochs, batch_size=args.batch_size,
                          seed=SEED, verbose=0)
        auc_off = eval_auc(m, Xva_off, Lva, Mva_off, Yva)
        results[mode]["val_auc_ppg_off"] = auc_off
        print(f"  {mode:<6} " + " | ".join(f"{H}h {auc_off[H]:.3f}" for H in auc_off))

    results["n_train_patients"] = len(splits["train"])
    results["n_val_patients"] = len(splits["val"])
    results["seed"] = SEED
    results["epochs"] = args.epochs
    (OUTPUT_DIR / "fusion_metrics.json").write_text(
        json.dumps(results, indent=1, default=str))
    print(f"\nexported -> {OUTPUT_DIR / 'fusion_metrics.json'}")

    print("\nSummary (val AUROC 6h):")
    for mode, r in results.items():
        if not isinstance(r, dict) or "val_auc" not in r:
            continue
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
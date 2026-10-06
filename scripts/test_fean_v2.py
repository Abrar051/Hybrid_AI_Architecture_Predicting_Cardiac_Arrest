"""Synthetic test for FEAN v2 (identity-adversarial head + PCGrad).

Plan rule 7: test section 7 v2 on synthetic data before real data. Reuses the
cached synthetic cohort embeddings from the notebook run (cache/embeddings/
train_*), mirrors the Section 7 patient-level split, and compares:

  plain      - v1 model, unchanged architecture (regression control)
  adv        - identity-adversarial head with gradient reversal
  adv_pcgrad - identity-adversarial + PCGrad across task gradients

An identity probe (logistic regression on the context vector) measures
identity leakage, but on synthetic data case/control IS a patient-level
property, so the BCE tasks rebuild identity info the adversary removes: the
probe is expected to stay high for all variants here. The synthetic test
checks the machinery (loss decreases, no NaNs, plain control learns); the
meaningful identity probe comparison runs on SDDB, where outcome varies
within patient.

Usage: python scripts/test_fean_v2.py [--epochs 15] [--batch-size N]
       [--adv-lam 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from ews_risk import FEAN, train_fean  # noqa: E402
from eval_utils import (load_manifest_features, load_splits, make_sequences,  # noqa: E402
                        make_sequences_memmap, sequence_ids)

EMBED_DIR = PROJECT / "cache" / "embeddings"
HORIZONS = (1, 6, 24)
SEED = 7
PROBE_MAX_SEQS = 3000              # probe on a subset: full-train encode is ~6 GB RAM


def probe_identity(m, Xtr, Ltr, idtr, val_frac=0.2, seed=7):
    """Linear identity probe on train windows: hold out 20% of each train
    patient's windows, fit a logistic regression (linear in the context
    vector, like the adversarial id head itself) on the rest, and classify
    the held-out windows. Chance is 1 / n_patients."""
    with torch.no_grad():
        ctx = m.encode(torch.from_numpy(Xtr), torch.from_numpy(Ltr)).numpy()
    rs = np.random.default_rng(seed)
    held, fit = [], []
    for c in np.unique(idtr):
        pos = np.where(idtr == c)[0]
        rs.shuffle(pos)
        k = max(1, int(val_frac * len(pos)))
        held.append(pos[:k])
        fit.append(pos[k:])
    held = np.concatenate(held)
    fit = np.concatenate(fit)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(ctx[fit], idtr[fit])
    return float((clf.predict(ctx[held]) == idtr[held]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--adv-lam", type=float, default=0.5)
    args = ap.parse_args()

    meta, feats, masks, labels, tte, ids = load_manifest_features("dev")
    splits = load_splits()
    train_pids, val_pids = set(splits["train"]), set(splits["val"])
    tr = meta.patient_id.isin(train_pids).to_numpy()
    va = meta.patient_id.isin(val_pids).to_numpy()
    assert tr.any() and va.any()
    assert not (set(meta.loc[tr, "patient_id"]) & set(val_pids))
    n_train_patients = len(train_pids)
    mu, sd = feats[tr].mean(0), feats[tr].std(0)
    z = (feats - mu) / (sd + 1e-8)
    Xtr, Ltr, Ytr, Ttr = make_sequences_memmap(
        meta.loc[tr], z[tr], {H: labels[H][tr] for H in HORIZONS},
        tte[tr], cache_f=EMBED_DIR.parent / "seqs_train_feanv2.npy")
    idtr = sequence_ids(meta.loc[tr])    # factorized within the train subset (0..n_train-1)
    Xva, Lva, Yva, Tva = make_sequences(meta.loc[va], z[va],
                                        {H: labels[H][va] for H in HORIZONS},
                                        tte[va])
    idva = sequence_ids(meta.loc[va])   # factorized within the val subset
    print(f"train seqs {len(Xtr)}, val seqs {len(Xva)} | "
          f"val patients {len(val_pids)} | train patients {n_train_patients} "
          f"| seed {SEED}")
    torch.manual_seed(SEED)          # seed model init (train_fean re-seeds too)

    results = {}
    for name, n_ids, adv_lam, pcgrad in [
            ("plain", 0, 0.0, False),
            ("adv", n_train_patients, args.adv_lam, False),
            ("adv_pcgrad", n_train_patients, args.adv_lam, True)]:
        print(f"\n=== {name} ({args.epochs} epochs, batch {args.batch_size}) ===")
        m = FEAN(n_ids=n_ids)
        m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr, ids_tr=idtr if n_ids else None,
                             Xva=Xva, Lva=Lva, Yva=Yva, ids_va=idva if n_ids else None,
                             epochs=args.epochs, batch_size=args.batch_size,
                             adv_lam=adv_lam, adv_ramp_epochs=3,
                             use_pcgrad=pcgrad, seed=SEED)
        assert all(np.isfinite(h["train_loss"]) for h in hist), f"NaN loss in {name}"
        assert hist[-1]["train_loss"] < hist[0]["train_loss"], f"{name} loss did not decrease"
        with torch.no_grad():
            out = m(torch.from_numpy(Xva), torch.from_numpy(Lva))
        auc = {H: roc_auc_score(Yva[H], torch.sigmoid(out["risk"][H]).numpy())
               for H in HORIZONS if Yva[H].sum() > 0 and (Yva[H] == 0).sum() > 0}
        probe = probe_identity(m, Xtr[:PROBE_MAX_SEQS], Ltr[:PROBE_MAX_SEQS],
                               idtr[:PROBE_MAX_SEQS])
        results[name] = dict(val_auc=auc, id_probe_acc=probe,
                             last_train_loss=hist[-1]["train_loss"],
                             last_val_loss=hist[-1].get("val_loss"))
        print(f"  val AUROC: " + " | ".join(f"{H}h {auc[H]:.3f}" for H in auc))
        print(f"  identity probe accuracy on held-out train windows: {probe:.3f} "
              f"(chance 1/{n_train_patients} = {1 / n_train_patients:.3f}; "
              f"lower = identity removed)")

    print("\nSummary:")
    for name, r in results.items():
        print(f"  {name:<11} train_loss {r['last_train_loss']:.4f} | "
              f"val_auc6h {r['val_auc'].get(6, float('nan')):.3f} | "
              f"id_probe {r['id_probe_acc']:.3f}")

    assert results["plain"]["val_auc"].get(1, 0) >= 0.9, "plain control undertrained"
    print("\nPASS: v2 variants train without NaNs, losses decrease, and the "
          "plain control learns the synthetic signal.")
    print("NOTE: the identity probe stays high for ALL variants here because on "
          "synthetic data patient identity IS the outcome signal (case/control "
          "is per patient), so the BCE tasks rebuild identity info that the "
          "adversary removes. The meaningful identity probe comparison is on "
          "SDDB, where outcome varies within patient.")


if __name__ == "__main__":
    main()
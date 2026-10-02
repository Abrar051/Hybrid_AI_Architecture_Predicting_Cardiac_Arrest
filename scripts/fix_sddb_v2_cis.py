"""Correct the bootstrap CIs of the first SDDB v2 run.

The first v2 run had a bug in bootstrap_metric (resampled window indices
instead of records), so every CI collapsed to the point estimate. The means
are unaffected. This script reloads the per-fold models saved under
cache/models/sddb_v2/, rebuilds predictions, and rewrites
outputs/sddb_v2_metrics.json with record-level bootstrap CIs (plan rule 8).

Usage: python scripts/fix_sddb_v2_cis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from analyze_sddb import (EMBED_DIR, MODEL_DIR, OUTPUT_DIR, SEED, HORIZONS,  # noqa: E402
                          assemble, make_sequences, bootstrap_metric)
from ews_risk import FEAN, mask_blocks  # noqa: E402

N_IDS = {"plain": 0, "adv": 20, "adv_pcgrad": 20, "no_emb": 0, "no_eng": 0}
MASK = {"no_emb": dict(use_eng=True, use_emb=False),
        "no_eng": dict(use_eng=False, use_emb=True)}


def fold_splits(n_records):
    rs = np.random.default_rng(SEED)
    order = rs.permutation(n_records)
    return np.array_split(order, 5)


def main():
    meta = pd.read_csv(EMBED_DIR / "sddb_meta.csv")
    emb = np.load(EMBED_DIR / "sddb_sddb_emb.npy")
    n = len(meta)
    emb = emb.reshape(n, -1, 768).mean(axis=1)
    feats, masks, labels, tte, ids = assemble(meta, emb)
    kind_codes = pd.factorize(meta["kind"])[0]
    rec_codes = pd.factorize(meta["record"])[0]
    records = np.unique(rec_codes)
    folds = fold_splits(len(records))

    metrics = json.loads((OUTPUT_DIR / "sddb_v2_metrics.json").read_text())

    for name in N_IDS:
        if not (MODEL_DIR / f"{name}_fold0.pt").exists():
            print(f"skip {name}: no saved fold models")
            continue
        f = mask_blocks(feats, **MASK[name]) if name in MASK else feats
        pred = {H: np.zeros(n) for H in HORIZONS}
        for k, test_idx in enumerate(folds):
            te = np.isin(rec_codes, records[test_idx])
            tr = ~te
            mu, sd = f[tr].mean(0), f[tr].std(0)
            z = (f - mu) / (sd + 1e-8)
            Xte, Lte, Yte, Tte, idte, Mte = make_sequences(
                rec_codes[te], kind_codes[te], z[te],
                {H: labels[H][te] for H in HORIZONS}, tte[te], ids[te], masks[te])
            model = FEAN(n_ids=N_IDS[name])
            ckpt = torch.load(MODEL_DIR / f"{name}_fold{k}.pt", map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(Xte), torch.from_numpy(Lte))
            for H in HORIZONS:
                pred[H][te] = torch.sigmoid(out["risk"][H]).numpy()
        if name in MASK:
            section = metrics.setdefault("ablations", {}).setdefault(name, {})
        else:
            section = metrics.setdefault("cv", {}).setdefault(name, {})
        for H in HORIZONS:
            if labels[H].sum() == 0 or (labels[H] == 0).sum() == 0:
                continue
            auc, alo, ahi = bootstrap_metric(roc_auc_score, labels[H], pred[H],
                                             rec_codes)
            ap, aplo, aphi = bootstrap_metric(average_precision_score, labels[H],
                                              pred[H], rec_codes)
            section[f"{H}h"]["auroc"] = [auc, alo, ahi]
            section[f"{H}h"]["auprc"] = [ap, aplo, aphi]
            print(f"{name:<11} {H:>2}h: AUROC {auc:.3f} [{alo:.3f}, {ahi:.3f}] | "
                  f"AUPRC {ap:.3f} [{aplo:.3f}, {aphi:.3f}]")
    metrics["ci_fix_note"] = ("bootstrap CIs recomputed from saved fold models "
                              "with record-level resampling (first run's CIs "
                              "were degenerate due to a window-level resampling bug)")
    (OUTPUT_DIR / "sddb_v2_metrics.json").write_text(
        json.dumps(metrics, indent=1, default=str))
    print(f"\nrewrote -> {OUTPUT_DIR / 'sddb_v2_metrics.json'}")


if __name__ == "__main__":
    main()
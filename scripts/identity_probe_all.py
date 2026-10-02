"""Identity invariance probe for every saved SDDB v2 variant.

Measures how strongly each model's context vector encodes patient identity:
per fold, a logistic regression on 80% of the train windows' contexts
classifies the held out 20% into patients. Chance = 1 / n_train_records
(= 0.05 with 16 train records). Runs on the saved fold models, no retraining.

Usage: python scripts/identity_probe_all.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from analyze_sddb import (EMBED_DIR, MODEL_DIR, SEED, HORIZONS,  # noqa: E402
                          assemble, make_sequences, identity_probe)
from ews_risk import FEAN, mask_blocks  # noqa: E402

N_IDS = {"plain": 0, "adv": 20, "adv_pcgrad": 20, "gate": 0, "attn": 0,
         "weight": 0, "no_emb": 0, "no_eng": 0}
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
    print(f"identity probe (chance 1/{len(records)} = {1 / len(records):.3f}):")
    for name in N_IDS:
        if not (MODEL_DIR / f"{name}_fold0.pt").exists():
            continue
        fusion = name if name in ("gate", "attn", "weight") else "concat"
        f = mask_blocks(feats, **MASK[name]) if name in MASK else feats
        probes = []
        for k, test_idx in enumerate(folds):
            te = np.isin(rec_codes, records[test_idx])
            tr = ~te
            mu, sd = f[tr].mean(0), f[tr].std(0)
            z = (f - mu) / (sd + 1e-8)
            Xtr, Ltr, Ytr, Ttr, idtr, Mtr = make_sequences(
                rec_codes[tr], kind_codes[tr], z[tr],
                {H: labels[H][tr] for H in HORIZONS}, tte[tr], ids[tr], masks[tr])
            model = FEAN(n_ids=N_IDS[name], fusion=fusion)
            ckpt = torch.load(MODEL_DIR / f"{name}_fold{k}.pt", map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model.eval()
            probes.append(identity_probe(model, Xtr, Ltr, idtr,
                                         Mtr if fusion != "concat" else None))
        print(f"  {name:<11} {np.mean(probes):.3f} (per fold: "
              f"{['%.3f' % p for p in probes]})")


if __name__ == "__main__":
    main()
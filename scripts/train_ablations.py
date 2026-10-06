"""Train the four feature-block ablation variants on the dev split (cached to
cache/models/ablations_v2). Same split, seed, and normalization as the main
model so the tables come from one run.

Usage: python scripts/train_ablations.py [--epochs 20]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from config import ABLATE_DIR, CACHE_DIR, HORIZONS, BATCH_SIZE, TRAIN_STEP  # noqa: E402
from ews_risk import FEAN, mask_blocks, train_fean  # noqa: E402
from eval_utils import (load_manifest_features, load_splits,  # noqa: E402
                        make_sequences, make_sequences_memmap)

VARIANTS = {
    "full":              dict(use_eng=True, use_emb=True),
    "vitals_only":       dict(use_eng=False, use_emb=False),
    "vitals+engineered": dict(use_eng=True, use_emb=False),
    "vitals+embeddings": dict(use_eng=False, use_emb=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    meta, feats, masks, labels, tte, ids = load_manifest_features("dev")
    splits = load_splits()
    tr = meta.patient_id.isin(set(splits["train"])).to_numpy()
    va = meta.patient_id.isin(set(splits["val"])).to_numpy()

    mu, sd = feats[tr].mean(0), feats[tr].std(0)
    z = (feats - mu) / (sd + 1e-8)

    ABLATE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, kw in VARIANTS.items():
        ckpt = ABLATE_DIR / f"{name}.pt"
        f = mask_blocks(z, **kw)          # variant's own feature view
        # val predictions use the SAME masked features the variant trained on
        # (the old notebook fed full features to every variant here)
        Xva, Lva, Yva, Tva = make_sequences(
            meta.loc[va], f[va], {H: labels[H][va] for H in HORIZONS}, tte[va],
            step=TRAIN_STEP)
        if ckpt.exists():
            m = FEAN()
            m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
            m.eval()
            print(f"{name:<18} cached")
        else:
            Xtr, Ltr, Ytr, Ttr = make_sequences_memmap(
                meta.loc[tr], f[tr], {H: labels[H][tr] for H in HORIZONS},
                tte[tr], cache_f=CACHE_DIR / f"seqs_train_{name}.npy",
                step=TRAIN_STEP)
            pos_w = {H: float((Ytr[H] == 0).sum() / max(1, (Ytr[H] == 1).sum()))
                     for H in HORIZONS}
            torch.manual_seed(42)
            m = FEAN()
            m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr,
                                 Xva=Xva, Lva=Lva, Yva=Yva,
                                 epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 pos_weight=pos_w, seed=42, verbose=0)
            assert np.isfinite(hist[-1]["train_loss"]), f"NaN loss in {name}"
            torch.save({"model": m.state_dict()}, ckpt)
            print(f"{name:<18} trained {args.epochs} epochs -> {ckpt.name}")

        with torch.no_grad():
            out = m(torch.from_numpy(Xva), torch.from_numpy(Lva))
        auc = {H: roc_auc_score(Yva[H], torch.sigmoid(out["risk"][H]).numpy())
               for H in HORIZONS}
        summary[name] = {str(H): float(a) for H, a in auc.items()}
        print(f"{name:<18} val AUROC " + "  ".join(f"{H}h={a:.3f}"
                                                   for H, a in auc.items()))

    (ABLATE_DIR / "val_aucs.json").write_text(json.dumps(summary, indent=1))
    print("PASS: ablations trained; val AUROCs exported.")


if __name__ == "__main__":
    main()
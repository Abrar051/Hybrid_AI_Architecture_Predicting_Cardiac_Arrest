"""Train the main FEAN model on the dev split for several seeds.

Sequences are written to a float32 memmap (RAM-safe at cohort scale), training
uses batch_size=256, and each seed's RiskPipeline is saved to
cache/models/ews_v2/seed{s}/. The seed-42 model is copied to
cache/models/ews_v2/ as the canonical checkpoint (used by the SDDB zero-shot
section and the notebook replay).

Usage: python scripts/train_fean_models.py [--seeds 42 1 2] [--epochs 20]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from config import (CACHE_DIR, EWS_MODEL_DIR, HORIZONS, MODEL_SEEDS,  # noqa: E402
                    BATCH_SIZE, TRAIN_STEP, ALERT)
from ews_risk import FEAN, RiskPipeline, train_fean  # noqa: E402
from eval_utils import (load_manifest_features, load_splits,  # noqa: E402
                        make_sequences, make_sequences_memmap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(MODEL_SEEDS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    meta, feats, masks, labels, tte, ids = load_manifest_features("dev")
    splits = load_splits()
    train_pids, val_pids = set(splits["train"]), set(splits["val"])
    tr = meta.patient_id.isin(train_pids).to_numpy()
    va = meta.patient_id.isin(val_pids).to_numpy()
    assert tr.any() and va.any(), "empty train/val split"
    assert not (set(meta.loc[tr, "patient_id"]) & set(meta.loc[va, "patient_id"]))
    for H in HORIZONS:
        assert np.isfinite(labels[H]).all(), f"NaN labels at {H}h in dev manifest"
    n_cases_tr = meta.loc[tr, "is_case"].astype(bool).groupby(
        meta.loc[tr, "patient_id"]).first().sum()
    print(f"dev manifest: {len(meta)} windows | train {tr.sum()} windows "
          f"({len(train_pids)} patients, {int(n_cases_tr)} cases) | "
          f"val {va.sum()} windows ({len(val_pids)} patients)")

    mu, sd = feats[tr].mean(0), feats[tr].std(0)
    z = (feats - mu) / (sd + 1e-8)

    seq_cache = CACHE_DIR / "seqs_train.npy"
    Xtr, Ltr, Ytr, Ttr = make_sequences_memmap(
        meta.loc[tr], z[tr], {H: labels[H][tr] for H in HORIZONS}, tte[tr],
        cache_f=seq_cache, step=TRAIN_STEP)
    Xva, Lva, Yva, Tva = make_sequences(
        meta.loc[va], z[va], {H: labels[H][va] for H in HORIZONS}, tte[va],
        step=TRAIN_STEP)
    pos_w = {H: float((Ytr[H] == 0).sum() / max(1, (Ytr[H] == 1).sum()))
             for H in HORIZONS}
    print(f"sequences: train {len(Xtr)} | val {len(Xva)} | "
          f"pos_weight { {H: round(w, 1) for H, w in pos_w.items()} }")

    EWS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    val_aucs = {}
    for seed in args.seeds:
        seed_dir = EWS_MODEL_DIR / f"seed{seed}"
        if (seed_dir / "model.pt").exists():
            pipe = RiskPipeline.load(seed_dir)
            print(f"seed {seed}: cached -> {seed_dir}")
        else:
            torch.manual_seed(seed)      # model init (train_fean re-seeds too)
            model = FEAN()
            model, hist = train_fean(
                model, Xtr, Ltr, Ytr, Ttr, Xva=Xva, Lva=Lva, Yva=Yva,
                epochs=args.epochs, batch_size=args.batch_size,
                pos_weight=pos_w, seed=seed, verbose=1)
            assert all(np.isfinite(h["train_loss"]) for h in hist), \
                f"NaN train loss (seed {seed})"
            assert hist[-1]["train_loss"] < hist[0]["train_loss"], \
                f"seed {seed} loss did not decrease"
            pipe = RiskPipeline(model, mu, sd, **ALERT)
            pipe.save(seed_dir)
            print(f"seed {seed}: saved -> {seed_dir}")

        with torch.no_grad():
            out = pipe.model(torch.from_numpy(Xva), torch.from_numpy(Lva))
        auc = {H: roc_auc_score(Yva[H], torch.sigmoid(out["risk"][H]).numpy())
               for H in HORIZONS}
        val_aucs[str(seed)] = {str(H): float(a) for H, a in auc.items()}
        print(f"seed {seed} val AUROC: "
              + "  ".join(f"{H}h={a:.3f}" for H, a in auc.items()))

    # canonical copy: seed 42 (headline tables + SDDB zero-shot)
    canonical = EWS_MODEL_DIR
    if (canonical / "model.pt").exists():
        (canonical / "model.pt").unlink()
        (canonical / "stats.json").unlink()
    shutil.copy(EWS_MODEL_DIR / f"seed{args.seeds[0]}" / "model.pt",
                canonical / "model.pt")
    shutil.copy(EWS_MODEL_DIR / f"seed{args.seeds[0]}" / "stats.json",
                canonical / "stats.json")
    (canonical / "val_aucs.json").write_text(json.dumps(val_aucs, indent=1))
    print(f"canonical checkpoint (seed {args.seeds[0]}) -> {canonical}")
    print("PASS: FEAN trained for all seeds; val AUROCs exported.")


if __name__ == "__main__":
    main()
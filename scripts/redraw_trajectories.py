"""Redraw the SDDB risk-trajectory figures with the span-split fix.

The first v2 figures plotted windows in manifest order, which lists pre
windows before far windows, so the x axis (hours to onset) was not monotonic
and the curve retraced backwards across itself (broken-looking lines). This
script recomputes per-fold predictions from the saved fold models (same logic
as fix_sddb_v2_cis.py) and redraws both trajectory figures with pre and far
spans as separate segments.

Usage: python scripts/redraw_trajectories.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from analyze_sddb import (EMBED_DIR, MODEL_DIR, OUTPUT_DIR, SEED, HORIZONS,  # noqa: E402
                          assemble, make_sequences)
from ews_risk import FEAN, mask_blocks  # noqa: E402

N_IDS = {"plain": 0, "adv": 20, "adv_pcgrad": 20, "gate": 0, "attn": 0,
         "weight": 0, "no_emb": 0, "no_eng": 0}
MASK = {"no_emb": dict(use_eng=True, use_emb=False),
        "no_eng": dict(use_eng=False, use_emb=True)}


def fold_splits(n_records):
    rs = np.random.default_rng(SEED)
    order = rs.permutation(n_records)
    return np.array_split(order, 5)


def compute_preds(name):
    """Per-window 6 h probabilities for one variant from its saved fold models."""
    meta = pd.read_csv(EMBED_DIR / "sddb_meta.csv")
    emb = np.load(EMBED_DIR / "sddb_sddb_emb.npy")
    n = len(meta)
    emb = emb.reshape(n, -1, 768).mean(axis=1)
    feats, masks, labels, tte, ids = assemble(meta, emb)
    kind_codes = pd.factorize(meta["kind"])[0]
    rec_codes = pd.factorize(meta["record"])[0]
    records = np.unique(rec_codes)
    folds = fold_splits(len(records))
    f = mask_blocks(feats, **MASK[name]) if name in MASK else feats
    fusion = name if name in ("gate", "attn", "weight") else "concat"
    pred = np.zeros(n)
    for k, test_idx in enumerate(folds):
        te = np.isin(rec_codes, records[test_idx])
        tr = ~te
        mu, sd = f[tr].mean(0), f[tr].std(0)
        z = (f - mu) / (sd + 1e-8)
        Xte, Lte, Yte, Tte, idte, Mte = make_sequences(
            rec_codes[te], kind_codes[te], z[te],
            {H: labels[H][te] for H in HORIZONS}, tte[te], ids[te], masks[te])
        model = FEAN(n_ids=N_IDS[name], fusion=fusion)
        ckpt = torch.load(MODEL_DIR / f"{name}_fold{k}.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(Xte), torch.from_numpy(Lte),
                        mask=None if fusion == "concat" else torch.from_numpy(Mte))
        pred[te] = torch.sigmoid(out["risk"][6]).numpy()
    return meta, pred


def draw(meta, preds, out_path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"plain": "#2a78d6", "adv": "#d67a2a", "adv_pcgrad": "#2a9d5b",
              "gate": "#2a78d6", "attn": "#d67a2a", "weight": "#2a9d5b",
              "no_emb": "#7a5bd6", "no_eng": "#d63b8f"}
    rec = meta.loc[meta["kind"] == "pre", "record"].value_counts().index[0]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, pred in preds.items():
        mk = (meta["record"] == rec).to_numpy() & (meta["kind"] == "pre").to_numpy()
        t = (meta.loc[mk, "t_start_s"] - meta.loc[mk, "onset_s"]).to_numpy() / 3600.0
        ax.plot(t, pred[mk], linewidth=1.6, color=colors.get(name), label=name)
    ax.axvline(0, color="#d03b3b", linestyle="--", linewidth=1.2)
    ax.set_xlabel("hours relative to VF onset")
    ax.set_ylabel("predicted risk (arrest within 6 h)")
    ax.set_title(f"SDDB record {rec}: pre-arrest CV risk trajectories"
                 f"{title_suffix}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="plain,adv,adv_pcgrad,no_emb,no_eng")
    args = ap.parse_args()

    # run 1 figure: adversarial variants + ablations
    names = [n for n in args.variants.split(",") if (MODEL_DIR / f"{n}_fold0.pt").exists()]
    print(f"variants with saved fold models: {names}")
    meta = None
    preds = {}
    for name in names:
        meta, pred = compute_preds(name)
        preds[name] = pred
    draw(meta, preds, OUTPUT_DIR / "fig_sddb_v2_risk_trajectory.png")

    # fusion figure: gate / attn / weight
    fusion = [n for n in ("gate", "attn", "weight")
              if (MODEL_DIR / f"{n}_fold0.pt").exists()]
    meta = None
    preds = {}
    for name in fusion:
        meta, pred = compute_preds(name)
        preds[name] = pred
    if preds:
        draw(meta, preds, OUTPUT_DIR / "fig_sddb_v2_fusion_risk_trajectory.png")
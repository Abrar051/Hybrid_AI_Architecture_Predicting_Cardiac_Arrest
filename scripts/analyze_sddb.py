"""Real-data stress test v2: SDDB (Sudden Cardiac Death Holter Database) vs
our EWS model, with section 7 v2 additions.

Compared with v1 (which reported chance-level CV AUROC on 20 records), v2 adds:
1. Identity-adversarial FEAN variants (gradient reversal against a patient-ID
   head, plan section 7 v2) - plain / adv / adv_pcgrad (PCGrad across tasks).
2. Feature-block ablations on real data (no_emb / no_eng).
3. Diagnostics: per-epoch learning curves, attention salience by lead-time
   bin, time-averaged AUC across lead-time bins, linear identity probe.

Protocol changes vs v1 (both affect all variants equally): torch.manual_seed(7)
before model init; training sequences respect record and span boundaries (v1
let histories cross record and pre/far-span boundaries).

Honest expectations (plan rule 8): distribution shift (Holter 1980s vs ICU
monitors), missing modalities, 20 patients -> wide confidence intervals.
Research prototype only.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from ews_risk import (FEAN, RiskPipeline, SEQ_LEN, ENG_COLS,  # noqa: E402
                      build_features_masked, mask_blocks, train_fean)
from sddb_utils import (VF_ONSET, FLOOR_S, SPAN_S, NEG_OFFSET_S,  # noqa: E402
                        FS_TARGET, onset_seconds, record_duration, load_span,
                        ecg_features, require_sddb)
from synth_signals import zscore_segments  # noqa: E402

EMBED_DIR = PROJECT / "cache/embeddings"
MODEL_DIR = PROJECT / "cache/models/sddb_v2"
OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ECG_DIM = 768
HORIZONS = (1, 6, 24)
SEED = 7

VARIANTS = {
    "plain":      dict(n_ids=0, adv_lam=0.0, use_pcgrad=False, batch=None, epochs=15),
    "adv":        dict(n_ids=20, adv_lam=1.0, use_pcgrad=False, batch=None, epochs=15),
    "adv_pcgrad": dict(n_ids=20, adv_lam=1.0, use_pcgrad=True, batch=256, epochs=15),
    # fusion comparison (reviewer point): mask-aware fusion modes, no
    # adversarial head; PPG is absent on SDDB -> mask bit 0 + learned
    # missing embedding instead of zero-fill
    "gate":   dict(n_ids=0, adv_lam=0.0, use_pcgrad=False, batch=None, epochs=15,
                   fusion="gate"),
    "attn":   dict(n_ids=0, adv_lam=0.0, use_pcgrad=False, batch=None, epochs=15,
                   fusion="attn"),
    "weight": dict(n_ids=0, adv_lam=0.0, use_pcgrad=False, batch=None, epochs=15,
                   fusion="weight"),
}
ABLATIONS = {                       # plain architecture, masked feature blocks
    "no_emb": dict(use_eng=True, use_emb=False),
    "no_eng": dict(use_eng=False, use_emb=True),
}
ABLATION_EPOCHS = 10


def build_manifest():
    """Windows of 5 min: positives = last 6 h before VF onset, negatives = an
    early 6 h span of the record. Segments = 30 s at the window middle."""
    rows, segs = [], []
    n_skipped = 0
    for rec in VF_ONSET:
        onset = onset_seconds(rec)
        total = record_duration(rec)
        spans = []
        lo = max(0.0, onset - SPAN_S)
        if onset - FLOOR_S - lo >= 300:
            spans.append(("pre", lo, onset))
        far_end = onset - SPAN_S
        if far_end - NEG_OFFSET_S >= 300:
            spans.append(("far", NEG_OFFSET_S, min(NEG_OFFSET_S + SPAN_S, far_end)))
        for kind, s0, s1 in spans:
            sig, meta = load_span(rec, s0, s1)
            win = 5 * 60 * int(FS_TARGET)             # 5 min at 125 Hz
            n_win = len(sig) // win
            for w in range(n_win):
                t_start = s0 + w * 300                # seconds
                t_end = t_start + 300
                if kind == "pre" and t_end > onset - FLOOR_S:
                    continue
                i0 = w * win + (win - 30 * int(FS_TARGET)) // 2
                seg = sig[i0:i0 + 30 * int(FS_TARGET)]
                if not np.isfinite(seg).all():        # digitized-tape NaN gaps
                    n_skipped += 1
                    continue
                ef = ecg_features(seg[:, 0])
                rows.append(dict(
                    record=rec, kind=kind, t_start_s=t_start, t_end_s=t_end,
                    onset_s=onset, hr=ef["hr"], rrsd=ef["rrsd"],
                    qf_pass=1.0, rr_mean_s=ef["rr_mean_s"], rr_sdnn_s=ef["rr_sdnn_s"],
                    rr_rmssd_s=ef["rr_rmssd_s"], ectopy_burden=ef["ectopy_burden"],
                    abp_mean=0.0, abp_std=0.0, ptt_est_ms=0.0,
                    hr_trend=0.0, sbp_trend=0.0))
                # 12-lead tensor for ECG-FM: lead1 -> II, lead2 -> V5, rest zeros
                leads = np.zeros((12, len(seg)), np.float32)
                leads[1] = seg[:, 0]                  # II
                leads[10] = seg[:, 1]                 # V5
                segs.append(leads)
    print(f"skipped {n_skipped} windows with NaN gaps (digitized-tape corruption)")
    return pd.DataFrame(rows), np.stack(segs)


def embed_ecg(segs, tag):
    """ECG-FM embeddings via the ecgfm_env extractor (cached)."""
    seg_in = EMBED_DIR / f"sddb_{tag}_seg.npz"
    seg_out = EMBED_DIR / f"sddb_{tag}_emb.npy"
    if not seg_out.exists():
        x500 = np.stack([np.vstack([
            np.interp(np.arange(len(l) * 4) / 4, np.arange(len(l)), l) for l in s])
            for s in segs])                          # 125 -> 500 Hz, linear
        x500 = x500[:, :, :2500 * (x500.shape[2] // 2500)]
        chunks = x500.reshape(len(segs), 12, -1, 2500).transpose(0, 2, 1, 3)[:, ::2]
        chunks = zscore_segments(chunks.reshape(-1, 12, 2500))   # 3 chunks per window
        np.savez(seg_in, ecg=chunks)
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [str(Path.home() / "anaconda3/envs/ecgfm_env/bin/python"),
             str(PROJECT / "scripts/extract_ecgfm.py"),
             "--input", str(seg_in), "--output", str(seg_out),
             "--checkpoint", str(PROJECT / "weights/mimic_iv_ecg_physionet_pretrained.pt"),
             "--batch-size", "16"],
            check=True, capture_output=True, text=True, env=env, timeout=7200)
    emb = np.load(seg_out)                           # (n_chunks, 768)
    return emb.reshape(len(segs), -1, ECG_DIM).mean(axis=1)   # mean over chunks


def assemble(meta, emb):
    """Window features + per-horizon labels + tte + record/kind codes +
    per-window modality masks ([vitals, eng, ecg, ppg] presence bits)."""
    rows = [build_features_masked(meta.iloc[i],
                                  dict(zip(ENG_COLS, meta.iloc[i][ENG_COLS].astype(float))),
                                  emb[i], None)
            for i in range(len(meta))]
    feats = np.stack([r[0] for r in rows]).astype(np.float32)
    masks = np.stack([r[1] for r in rows]).astype(np.float32)
    tte = (meta["onset_s"] - meta["t_end_s"]).to_numpy() / 3600.0     # hours
    labels = {H: ((tte > 0.25) & (tte <= H)).astype(int) for H in HORIZONS}
    ids = pd.factorize(meta["record"])[0]
    return feats, masks, labels, tte, ids


def make_sequences(rec_codes, kind_codes, feats, labels, tte, ids, masks=None,
                   step=1):
    """Sequences of up to SEQ_LEN windows within one (record, span): histories
    never cross records or pre/far span boundaries. With masks, sequences
    carry per-window modality bits (padded history positions get mask 0)."""
    X, L, Y, T, ID, M = [], [], [], [], [], []
    for key in np.unique(np.stack([rec_codes, kind_codes], 1), axis=0):
        pos = np.where((rec_codes == key[0]) & (kind_codes == key[1]))[0]
        for j in range(0, len(pos), step):
            lo = max(0, j - SEQ_LEN + 1)
            n_hist = j - lo + 1
            X.append(np.pad(feats[pos[lo:j + 1]], ((SEQ_LEN - n_hist, 0), (0, 0))))
            L.append(n_hist)
            Y.append([labels[H][pos[j]] for H in HORIZONS])
            T.append(tte[pos[j]])
            ID.append(ids[pos[j]])
            if masks is not None:
                M.append(np.pad(masks[pos[lo:j + 1]], ((SEQ_LEN - n_hist, 0), (0, 0))))
    out = (np.stack(X).astype(np.float32), np.array(L, np.int64),
           {H: np.array([y[k] for y in Y], np.float32) for k, H in enumerate(HORIZONS)},
           np.array(T, np.float32), np.array(ID, np.int64))
    if masks is not None:
        out = out + (np.stack(M).astype(np.float32),)
    return out


def identity_probe(model, Xtr, Ltr, idtr, Mtr=None, val_frac=0.2):
    """Linear probe on train contexts: LR on 80% of windows, classify the rest.
    Chance = 1 / n_train_records."""
    with torch.no_grad():
        ctx = model.encode(torch.from_numpy(Xtr), torch.from_numpy(Ltr),
                           mask=None if Mtr is None else torch.from_numpy(Mtr)).numpy()
    rs = np.random.default_rng(SEED)
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


def run_cv(meta, feats, masks, labels, tte, ids, kind_codes, variant, spec,
           model_name, epochs):
    """5-fold CV grouped by record; returns (preds, history, attention-by-bin,
    id_probe_acc). preds[H]: (n_windows,) probabilities; history: per-fold list
    of per-epoch dicts; attn: {bin: (mean, n)} aggregated over test windows."""
    rec_codes = pd.factorize(meta["record"])[0]
    records = np.unique(rec_codes)
    rs = np.random.default_rng(SEED)
    order = rs.permutation(len(records))
    folds = np.array_split(order, 5)
    n = len(meta)
    fusion = spec.get("fusion", "concat")
    pred = {H: np.zeros(n) for H in HORIZONS}
    history = []
    probes, attn_bins = [], {"(0,1]": [], "(1,6]": [], "(6,24]": []}
    for k, test_idx in enumerate(folds):
        te = np.isin(rec_codes, records[test_idx])
        tr = ~te
        assert not (set(meta.loc[tr, "record"]) & set(meta.loc[te, "record"]))
        if variant == "no_eng" or variant == "no_emb":
            f = mask_blocks(feats, use_eng=spec["use_eng"], use_emb=spec["use_emb"])
        else:
            f = feats
        mu, sd = f[tr].mean(0), f[tr].std(0)
        z = (f - mu) / (sd + 1e-8)
        Xtr, Ltr, Ytr, Ttr, idtr, Mtr = make_sequences(
            rec_codes[tr], kind_codes[tr], z[tr],
            {H: labels[H][tr] for H in HORIZONS}, tte[tr], ids[tr], masks[tr])
        Xte, Lte, Yte, Tte, idte, Mte = make_sequences(
            rec_codes[te], kind_codes[te], z[te],
            {H: labels[H][te] for H in HORIZONS}, tte[te], ids[te], masks[te])
        torch.manual_seed(SEED + k)
        model = FEAN(n_ids=spec["n_ids"], fusion=fusion)
        mask_kw = dict(masks_tr=Mtr, masks_va=Mte) if fusion != "concat" else {}
        model, hist = train_fean(
            model, Xtr, Ltr, Ytr, ids_tr=idtr if spec["n_ids"] else None,
            Xva=Xte, Lva=Lte, Yva=Yte, ids_va=idte if spec["n_ids"] else None,
            epochs=epochs, batch_size=spec["batch"],
            adv_lam=spec["adv_lam"], adv_ramp_epochs=5,
            use_pcgrad=spec["use_pcgrad"], seed=SEED + k, verbose=0, **mask_kw)
        history.append(hist)
        probes.append(identity_probe(model, Xtr, Ltr, idtr,
                                     Mtr if fusion != "concat" else None))
        fwd_kw = dict(mask=torch.from_numpy(Mte)) if fusion != "concat" else {}
        with torch.no_grad():
            out = model(torch.from_numpy(Xte), torch.from_numpy(Lte),
                        return_attn=True, **fwd_kw)
        for H in HORIZONS:
            pred[H][te] = torch.sigmoid(out["risk"][H]).numpy()
        # attention salience by lead-time bin (mean over valid history positions)
        attn = out["attn"].numpy()                   # (B, 72)
        for bname, lo, hi in (("(0,1]", 0.0, 1.0), ("(1,6]", 1.0, 6.0),
                              ("(6,24]", 6.0, 24.0)):
            m = (Tte > lo) & (Tte <= hi)
            if m.sum() == 0:
                continue
            Lm = Lte[m]
            wmean = np.zeros(SEQ_LEN)
            wcnt = np.zeros(SEQ_LEN)
            for i, l in enumerate(Lm):
                wmean[SEQ_LEN - l:] += attn[m][i][SEQ_LEN - l:]
                wcnt[SEQ_LEN - l:] += 1
            if wcnt.sum() > 0:
                attn_bins[bname].append(wmean / np.maximum(wcnt, 1))
        torch.save({"model": model.state_dict()}, MODEL_DIR / f"{model_name}_fold{k}.pt")
        print(f"  fold {k}: test records {sorted(int(r) for r in records[test_idx])} "
              f"(n={int(te.sum())}) done | val AUC6h last epoch "
              f"{hist[-1].get('val_auc_6h')}")
    return pred, history, attn_bins, float(np.mean(probes))


def bootstrap_metric(metric, y, p, rec_codes, n_iter=1000):
    """Patient (record) level bootstrap CI (plan rule 8)."""
    rec_ids = np.unique(rec_codes)
    s = []
    for _ in range(n_iter):
        idx = resample(np.arange(len(rec_ids)))
        m = np.isin(rec_codes, rec_ids[idx])
        if y[m].sum() > 0 and (y[m] == 0).sum() > 0:
            s.append(metric(y[m], p[m]))
    if not s:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(np.mean(s)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="plain,adv,adv_pcgrad",
                    help="comma list of: plain, adv, adv_pcgrad, gate, attn, "
                         "weight")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs for all variants (incl. ablations)")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "sddb_v2_metrics.json"),
                    help="path for the metrics JSON")
    ap.add_argument("--fig-prefix", default="sddb_v2",
                    help="prefix for figure and profile filenames under outputs/")
    args = ap.parse_args()

    require_sddb()
    print("building SDDB manifest...")
    meta, segs = build_manifest()
    n = len(meta)
    print(f"windows: {n} ({meta['kind'].value_counts().to_dict()})")
    meta.to_csv(EMBED_DIR / "sddb_meta.csv", index=False)

    emb = embed_ecg(segs, "sddb")
    assert len(emb) == n, (emb.shape, n)
    feats, masks, labels, tte, ids = assemble(meta, emb)
    kind_codes = pd.factorize(meta["kind"])[0]
    rec_codes = pd.factorize(meta["record"])[0]
    records = np.unique(rec_codes)
    print("label sums:", {H: int(labels[H].sum()) for H in HORIZONS})
    print(f"(24 h labels are all-positive on SDDB: no record extends far enough "
          f"before onset to supply 24 h negatives)")
    print(f"modality masks: {dict(zip(('vitals', 'eng', 'ecg', 'ppg'), masks.mean(0)))}")

    results = {"zero_shot": {}, "cv": {}, "ablations": {}, "attention": {},
               "identity_probe": {}, "learning_curves_summary": {}}

    # ---------- (a) zero-shot: synthetic-trained model on SDDB --------------
    print("\n(a) zero-shot (synthetic-trained model):")
    pipe = RiskPipeline.load(PROJECT / "cache/models/ews_v1")
    z = (feats - pipe.feat_mean) / (pipe.feat_std + 1e-8)
    Xz, Lz, _, _, _ = make_sequences(rec_codes, kind_codes, z, labels, tte, ids)
    with torch.no_grad():
        out = pipe.model(torch.from_numpy(Xz), torch.from_numpy(Lz))
    probs_zs = {H: torch.sigmoid(out["risk"][H]).numpy() for H in HORIZONS}
    for H in HORIZONS:
        auc = roc_auc_score(labels[H], probs_zs[H])
        ap = average_precision_score(labels[H], probs_zs[H])
        if H == 24:
            note = " (degenerate: no negatives on SDDB)"
        else:
            note = ""
        results["zero_shot"][f"{H}h"] = {"auroc": auc, "auprc": ap, "note": note}
        print(f"  {H:>2}h: AUROC {auc:.3f} | AUPRC {ap:.3f}{note}")

    # ---------- (b) 5-fold grouped-by-record CV, v2 variants -----------------
    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {list(VARIANTS)}")
    print(f"\n(b) 5-fold grouped-by-record CV: {variants}")
    all_hist = {}
    preds = {}
    attn_profiles = {}
    for name in variants:
        spec = dict(VARIANTS[name])
        epochs = args.epochs or spec.pop("epochs")
        print(f"\n--- variant {name} ({epochs} epochs) ---")
        pred, hist, attn, probe = run_cv(meta, feats, masks, labels, tte, ids,
                                         kind_codes, name, spec, name, epochs)
        all_hist[name] = hist
        preds[name] = pred
        attn_profiles[name] = attn
        results["identity_probe"][name] = probe
        results["attention"][name] = {b: float(np.mean([w[71] for w in lst]))
                                      for b, lst in attn.items() if lst}
        results["cv"][name] = {}
        for H in HORIZONS:
            if labels[H].sum() == 0 or (labels[H] == 0).sum() == 0:
                results["cv"][name][f"{H}h"] = {"auroc": None, "note": "degenerate"}
                continue
            auc, alo, ahi = bootstrap_metric(roc_auc_score, labels[H], pred[H], rec_codes)
            ap, aplo, aphi = bootstrap_metric(average_precision_score, labels[H],
                                              pred[H], rec_codes)
            results["cv"][name][f"{H}h"] = {
                "auroc": [auc, alo, ahi], "auprc": [ap, aplo, aphi],
                "n_pos": int(labels[H].sum())}
            print(f"  {H:>2}h: AUROC {auc:.3f} [{alo:.3f}, {ahi:.3f}] | "
                  f"AUPRC {ap:.3f} [{aplo:.3f}, {aphi:.3f}]")
        # time-averaged AUC across lead-time bins (24 h head, per notebook S8;
        # the 24 h head has no negatives on SDDB, so treat as descriptive)
        bins = [(0, 1), (1, 6), (6, 24)]
        tavg = []
        for lo, hi in bins:
            pos = (tte > lo) & (tte <= hi)
            if pos.sum() == 0 or (~pos).sum() == 0:
                continue
            tavg.append(roc_auc_score(pos.astype(int), pred[24]))
        results["cv"][name]["time_avg_auc_leadtime"] = float(np.mean(tavg)) if tavg else None
        print(f"  time-averaged AUC across lead-time bins: "
              f"{results['cv'][name]['time_avg_auc_leadtime']}")
        print(f"  identity probe acc (train windows): {probe:.3f} "
              f"(chance 1/{len(records)} = {1 / len(records):.3f})")

    # ---------- (c) feature-block ablations on real data ---------------------
    if not args.skip_ablations:
        print("\n(c) feature-block ablations (plain architecture, "
              f"{ABLATION_EPOCHS} epochs):")
        for name, mask in ABLATIONS.items():
            spec = dict(VARIANTS["plain"])
            epochs = args.epochs or ABLATION_EPOCHS
            pred, hist, attn, probe = run_cv(meta, feats, masks, labels, tte, ids,
                                             kind_codes, name,
                                             dict(spec, **mask), name, epochs)
            preds[name] = pred
            results["ablations"][name] = {}
            for H in HORIZONS:
                if labels[H].sum() == 0 or (labels[H] == 0).sum() == 0:
                    results["ablations"][name][f"{H}h"] = None
                    continue
                auc, alo, ahi = bootstrap_metric(roc_auc_score, labels[H], pred[H],
                                                 rec_codes)
                results["ablations"][name][f"{H}h"] = {"auroc": [auc, alo, ahi]}
                print(f"  {name:<7} {H:>2}h: AUROC {auc:.3f} [{alo:.3f}, {ahi:.3f}]")

    # learning-curve summary (mean over folds per epoch) for all CV variants
    for name, hist in all_hist.items():
        E = min(len(h) for h in hist)
        lc = {}
        for key in ("train_loss", "val_loss"):
            lc[key] = [float(np.mean([h[e][key] for h in hist]))
                       for e in range(E)]
        lc["val_auc_6h"] = [float(np.mean([h[e]["val_auc_6h"] for h in hist]))
                            for e in range(E)]
        results["learning_curves_summary"][name] = lc

    results["n_windows"] = n
    results["n_records"] = len(records)
    results["seed"] = SEED
    results["protocol_note"] = ("sequences respect record and span boundaries; "
                                "torch seed 7 at init and per fold")
    results["disclaimer"] = ("research prototype; Holter data, 20 patients, "
                             "no PPG/ABP; not clinical validation")
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    fig_p = args.fig_prefix

    # ---------- figures ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (i) learning curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = {"plain": "#2a78d6", "adv": "#d67a2a", "adv_pcgrad": "#2a9d5b"}
    for name, lc in results["learning_curves_summary"].items():
        c = colors.get(name, None)
        axes[0].plot(lc["train_loss"], color=c, alpha=0.35, label=f"{name} train")
        axes[0].plot(lc["val_loss"], color=c, label=f"{name} val (held-out)")
        if "val_auc_6h" in lc:
            axes[1].plot(lc["val_auc_6h"], color=c, label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (mean over folds)")
    axes[0].set_title("train vs held-out loss")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.5, color="#999999", linestyle=":", linewidth=1)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("held-out 6 h AUROC")
    axes[1].set_title("held-out 6 h AUROC per epoch")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"fig_{fig_p}_learning_curves.png", dpi=150)
    plt.close(fig)

    # (ii) risk trajectory for the record with the longest pre-arrest span.
    # Pre span only: the manifest lists pre windows before far windows, and
    # mixing spans makes the line retrace or break at the span boundary. Far
    # span windows (6 to 8 h before onset) stay at baseline risk.
    rec = meta.loc[meta["kind"] == "pre", "record"].value_counts().index[0]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, p in preds.items():
        mk = (meta["record"] == rec).to_numpy() & (meta["kind"] == "pre").to_numpy()
        t = (meta.loc[mk, "t_start_s"] - meta.loc[mk, "onset_s"]).to_numpy() / 3600.0
        ax.plot(t, p[6][mk], linewidth=1.6, color=colors.get(name), label=name)
    ax.axvline(0, color="#d03b3b", linestyle="--", linewidth=1.2)
    ax.set_xlabel("hours relative to VF onset")
    ax.set_ylabel("predicted risk (arrest within 6 h)")
    ax.set_title(f"SDDB record {rec}: CV risk trajectories (v2, real pre-arrest ECG)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"fig_{fig_p}_risk_trajectory.png", dpi=150)
    plt.close(fig)

    # (iii) attention salience by lead-time bin (weight vs hours back)
    np.savez(OUTPUT_DIR / f"{fig_p}_attention_profiles.npz",
             **{f"{name}_{bin}": np.mean(lst, axis=0)
                for name, bins in attn_profiles.items()
                for bin, lst in bins.items() if lst})
    print("\nattention salience (last-history-position mean weight per bin):")
    for name, bins in results["attention"].items():
        print(f"  {name:<11} " + " | ".join(f"{b} {w:.3f}" for b, w in bins.items()))
    n_variants = len(attn_profiles)
    fig, axes = plt.subplots(1, n_variants, figsize=(4.6 * n_variants, 3.4),
                             squeeze=False)
    hours_back = np.arange(SEQ_LEN)[::-1] * 5 / 60.0          # position -> hours
    for ax, (name, bins) in zip(axes[0], attn_profiles.items()):
        for b, lst in bins.items():
            if lst:
                ax.plot(hours_back, np.mean(lst, axis=0), label=b)
        ax.axvline(0, color="#999999", linestyle=":", linewidth=1)
        ax.set_xlabel("hours back from current window")
        ax.set_ylabel("mean attention weight")
        ax.set_title(f"{name}")
        ax.legend(fontsize=7)
    fig.suptitle("SDDB v2: attention vs lead time (mean over folds, held-out windows)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"fig_{fig_p}_attention.png", dpi=150)
    plt.close(fig)

    print(f"\nsaved -> {args.out}")
    print("(honest caveat: Holter 1980s data, 20 patients, no PPG/ABP -> "
          "a real-data smoke test, not a clinical validation)")


if __name__ == "__main__":
    main()
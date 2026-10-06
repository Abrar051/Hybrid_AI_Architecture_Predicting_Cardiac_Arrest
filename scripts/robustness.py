"""Robustness evaluation (reviewer point): noisy signals, missing PPG, ECG
artifacts, delayed measurements, sensor failure.

Rebuild (240-patient cohort): models trained on the dev-train split
(cache/models/robustness_v2), evaluated on the held-out TEST set (48
patients, step=1 windows). Feature-level perturbations + clean carry
patient-bootstrap CIs (valid at 48 patients); waveform-level perturbations
re-embed 200 sampled windows through the real encoders and stay point
estimates (stated in the disclaimer). Waveform segments are regenerated
deterministically via synth_signals.cut_segment (no npz dependency).

Usage: python scripts/robustness.py [--skip-embed] [--epochs 15]
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
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from ews_risk import FEAN, train_fean  # noqa: E402
from eval_utils import (load_manifest_features, load_splits,  # noqa: E402
                        make_sequences, make_sequences_memmap,
                        patient_bootstrap)
from embed_windows import build_segments  # noqa: E402
from synth_signals import zscore_segments  # noqa: E402

EMBED_DIR = PROJECT / "cache/embeddings"
MODEL_DIR = PROJECT / "cache/models/robustness_v2"
OUTPUT_DIR = PROJECT / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZONS = (1, 6, 24)
SEED = 7
PPG_DIM = 512
N_WAVEFORM_WINDOWS = 200          # subset re-embedded for waveform tags


# ---------- model training (dev-train split, cached) ------------------------

def train_models(epochs, batch_size):
    """Train concat/gate/weight on the dev-train subset (cached)."""
    meta, feats, masks, labels, tte, ids = load_manifest_features("dev")
    splits = load_splits()
    tr = meta.patient_id.isin(set(splits["train"])).to_numpy()
    rs = np.random.default_rng(SEED)
    missing = np.zeros(len(meta), bool)
    missing[np.where(tr)[0][rs.choice(int(tr.sum()), int(0.1 * tr.sum()),
                                      replace=False)]] = True
    z = (feats - feats[tr].mean(0)) / (feats[tr].std(0) + 1e-8)
    z[missing, -PPG_DIM:] = 0.0
    masks_tr = masks.copy()
    masks_tr[missing, 3] = 0.0
    Xtr, Ltr, Ytr, Ttr, Mtr = make_sequences_memmap(
        meta.loc[tr], z[tr], {H: labels[H][tr] for H in HORIZONS},
        tte[tr], cache_f=MODEL_DIR / "seqs_train.npy", masks=masks_tr[tr])
    out = {}
    for mode in ("concat", "gate", "weight"):
        ckpt = MODEL_DIR / f"{mode}.pt"
        stats = MODEL_DIR / f"{mode}_stats.npz"
        if not (ckpt.exists() and stats.exists()):
            torch.manual_seed(SEED)
            m = FEAN(fusion=mode)
            if mode == "concat":
                m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr, epochs=epochs,
                                     batch_size=batch_size, seed=SEED, verbose=0)
            else:
                m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr, masks_tr=Mtr,
                                     epochs=epochs, batch_size=batch_size,
                                     seed=SEED, verbose=0)
            assert np.isfinite(hist[-1]["train_loss"])
            torch.save({"model": m.state_dict()}, ckpt)
            np.savez(stats, mean=feats[tr].mean(0), std=feats[tr].std(0))
        out[mode] = (ckpt, stats)
    return out


# ---------- per-patient streaming prediction --------------------------------

def _load_model(mode, ckpt):
    m = FEAN(fusion=mode)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    m.eval()
    return m


def predict(mode, feats, masks, labels, tte, meta, ckpt, stats,
            horizons=(1, 6, 24)):
    """Stream predictions per patient (RAM-safe) and return
    {H: (y, p, patient_codes)} for finite labels."""
    st = np.load(stats)
    z = (feats - st["mean"]) / (st["std"] + 1e-8)
    m = _load_model(mode, ckpt)
    ys = {H: [] for H in horizons}
    ps = {H: [] for H in horizons}
    pats = []
    for pid, g in meta.groupby("patient_id"):
        idx = g.index.to_numpy()
        X, L, Y, T, M = make_sequences(meta.loc[idx], z[idx],
                                       {H: labels[H][idx] for H in horizons},
                                       tte[idx], step=1, masks=masks[idx])
        with torch.no_grad():
            out = m(torch.from_numpy(X), torch.from_numpy(L),
                    mask=None if mode == "concat" else torch.from_numpy(M))
        for H in horizons:
            fin = np.isfinite(Y[H])
            ys[H].append(Y[H][fin])
            ps[H].append(torch.sigmoid(out["risk"][H]).numpy()[fin])
            if H == horizons[0]:
                pats.append(np.repeat(int(pid), int(fin.sum())))
    out = {}
    for H in horizons:
        y = np.concatenate(ys[H])
        p = np.concatenate(ps[H])
        pa = np.concatenate(pats)
        assert y.sum() > 0 and (y == 0).sum() > 0, f"single class at {H}h"
        out[H] = (y, p, pa)
    return out


def auc_with_ci(y, p, pa, n_iter=1000):
    point, lo, hi, nv = patient_bootstrap(roc_auc_score, y, p, pa,
                                          n_iter=n_iter, seed=SEED)
    return {"auc": point, "ci_lo": lo, "ci_hi": hi, "n_valid_draws": nv}


# ---------- waveform re-embedding -------------------------------------------

def embed_ecg_tag(tag, chunks):
    """Re-embed perturbed ECG chunks via ecgfm_env (cached by tag). Chunks are
    re-z-scored as in the original protocol (the perturbation is applied to
    the z-scored clean segments, then standardized again)."""
    emb_out = EMBED_DIR / f"rob_{tag}_ecg.npy"
    if not emb_out.exists():
        seg_in = EMBED_DIR / f"rob_{tag}_ecg.npz"
        np.savez(seg_in, ecg=zscore_segments(chunks))
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["OMP_NUM_THREADS"] = "4"
        subprocess.run(
            [str(Path.home() / "anaconda3/envs/ecgfm_env/bin/python"),
             str(PROJECT / "scripts/extract_ecgfm.py"),
             "--input", str(seg_in), "--output", str(emb_out),
             "--checkpoint", str(PROJECT / "weights/mimic_iv_ecg_physionet_pretrained.pt"),
             "--batch-size", "16"],
            check=True, capture_output=True, text=True, env=env, timeout=7200)
        (EMBED_DIR / f"rob_{tag}_ecg.npz").unlink(missing_ok=True)
    return np.load(emb_out)                          # (n_chunks, 768)


def embed_ppg_tag(tag, segments):
    emb_out = EMBED_DIR / f"rob_{tag}_ppg.npy"
    if not emb_out.exists():
        seg_in = EMBED_DIR / f"rob_{tag}_ppg.npz"
        zs = np.stack([(s - s.mean()) / (s.std() + 1e-8) for s in segments])
        np.savez(seg_in, ppg=zs.astype(np.float32))
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["OMP_NUM_THREADS"] = "4"
        subprocess.run(
            [str(Path.home() / "anaconda3/envs/papagei_env/bin/python"),
             str(PROJECT / "scripts/extract_papagei.py"),
             "--input", str(seg_in), "--output", str(emb_out),
             "--weights", str(PROJECT / "weights/papagei_s.pt"),
             "--batch-size", "32"],
            check=True, capture_output=True, text=True, env=env, timeout=7200)
        (EMBED_DIR / f"rob_{tag}_ppg.npz").unlink(missing_ok=True)
    return np.load(emb_out)                          # (n_chunks, 512)


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    print("loading synthetic test set...")
    meta, feats, masks, labels, tte, ids = load_manifest_features("test")
    n = len(meta)
    n_patients = meta.patient_id.nunique()
    print(f"test windows {n}, patients {n_patients}")

    print("training (or loading cached) concat/gate/weight...")
    models = train_models(args.epochs, args.batch_size)
    print(f"  loaded: {list(models)}")

    results = {"clean": {}, "perturbations": {}}
    for mode, (ckpt, stats) in models.items():
        pred = predict(mode, feats, masks, labels, tte, meta, ckpt, stats)
        results["clean"][mode] = {str(H): auc_with_ci(*pred[H])
                                  for H in (1, 6, 24)}
        print(f"  clean {mode}: "
              f"{ {H: round(results['clean'][mode][str(H)]['auc'], 3) for H in (1, 6, 24)} }")

    def feature_perturbation(tag, f, m):
        results["perturbations"].setdefault(tag, {})
        for mode, (ckpt, stats) in models.items():
            pred = predict(mode, f, m, labels, tte, meta, ckpt, stats,
                           horizons=(6,))
            results["perturbations"][tag][mode] = auc_with_ci(*pred[6])
        print(f"  {tag} done")

    # P2 missing PPG
    f = feats.copy()
    f[:, -PPG_DIM:] = 0.0
    m = masks.copy()
    m[:, 3] = 0.0
    feature_perturbation("missing_ppg", f, m)

    # P3 delayed measurements (whole vector shifted k windows)
    for k in (1, 2, 3):
        f = feats.copy()
        for pid in np.unique(meta["patient_id"]):
            idx = np.where(meta["patient_id"] == pid)[0]
            f[idx[k:]] = feats[idx[:-k]]
        feature_perturbation(f"delay_{5 * k}min", f, masks)

    # P4 sensor failure: PPG dies 2 h before the case arrest, at 10 h for control
    f = feats.copy()
    m = masks.copy()
    for pid in np.unique(meta["patient_id"]):
        idx = np.where(meta["patient_id"] == pid)[0]
        ah = meta.loc[meta["patient_id"] == pid, "arrest_h"].iloc[0]
        cutoff = ah - 2.0 if np.isfinite(ah) else 10.0
        drop = idx[meta.loc[idx, "t_start_h"] >= cutoff]
        f[drop, -PPG_DIM:] = 0.0
        m[drop, 3] = 0.0
    feature_perturbation("sensor_failure", f, m)

    # P1 vitals noise (0.5/1/2 x per column std)
    for k in (0.5, 1.0, 2.0):
        rs = np.random.default_rng(SEED + int(k * 10))
        f = feats.copy()
        f[:, :7] += k * feats[:, :7].std(0) * rs.standard_normal((n, 7)).astype(
            np.float32)
        feature_perturbation(f"vitals_noise_{k}x", f, masks)

    # ---------- waveform level perturbations (re-embedded) -------------------
    if not args.skip_embed:
        rs = np.random.default_rng(SEED)
        win_idx = rs.choice(n, N_WAVEFORM_WINDOWS, replace=False)
        sub_meta = meta.iloc[win_idx].reset_index(drop=True)
        ecg_sub, _ = build_segments(sub_meta, "ecg")   # (200*6, 12, 2500)
        ppg_sub, _ = build_segments(sub_meta, "ppg")   # (200*3, 1250)

        def replace_ecg_block(tag, chunks):
            emb = embed_ecg_tag(tag, chunks)
            f = feats.copy()
            f[win_idx, 17:17 + 768] = emb.reshape(len(win_idx), 6, 768).mean(1)
            results["perturbations"].setdefault(tag, {})
            for mode, (ckpt, stats) in models.items():
                pred = predict(mode, f, masks, labels, tte, meta, ckpt, stats,
                               horizons=(6,))
                results["perturbations"][tag][mode] = {"auc": float(
                    roc_auc_score(*pred[6][:2])),
                    "ci_lo": None, "ci_hi": None}
            summary = {mm: round(results["perturbations"][tag][mm]["auc"], 3)
                       for mm in models}
            print(f"  {tag} done: {summary}")

        base = ecg_sub.copy()
        p2p = (base ** 2).mean(-1, keepdims=True)

        for snr in (20, 10, 5):
            noise = rs.standard_normal(base.shape).astype(np.float32)
            scale = np.sqrt(p2p) / (10 ** (snr / 10))
            replace_ecg_block(f"ecg_noise_{snr}dB",
                              (base + scale * noise).astype(np.float32))

        t = np.arange(2500) / 500.0
        amp = 0.2 * np.sqrt(p2p).mean()
        replace_ecg_block("ecg_wander",
                          (base + amp * np.sin(2 * np.pi * 0.5 * t)[None, None, :]
                           ).astype(np.float32))
        replace_ecg_block("ecg_mains",
                          (base + 0.05 * np.sin(2 * np.pi * 50 * t)[None, None, :]
                           ).astype(np.float32))
        flat = base.copy()
        for i in range(len(flat)):
            lo = rs.integers(0, 1250)
            flat[i, :, lo:lo + 1250] = 0.0
        replace_ecg_block("ecg_flatline", flat)
        sat = np.clip(base, -0.5 * base.max(), 0.5 * base.max())
        replace_ecg_block("ecg_saturation", sat)

        ppg_base = ppg_sub.copy()
        p2p = (ppg_base ** 2).mean(-1, keepdims=True)
        noise = rs.standard_normal(ppg_base.shape).astype(np.float32)
        ppg_noisy = (ppg_base + np.sqrt(p2p) / (10 ** (5 / 20)) * noise).astype(
            np.float32)
        emb = embed_ppg_tag("ppg_noise_5dB", ppg_noisy)
        f = feats.copy()
        f[win_idx, -PPG_DIM:] = emb.reshape(len(win_idx), 3, PPG_DIM).mean(1)
        results["perturbations"]["ppg_noise_5dB"] = {}
        for mode, (ckpt, stats) in models.items():
            pred = predict(mode, f, masks, labels, tte, meta, ckpt, stats,
                           horizons=(6,))
            results["perturbations"]["ppg_noise_5dB"][mode] = {
                "auc": float(roc_auc_score(*pred[6][:2])),
                "ci_lo": None, "ci_hi": None}
        print("  ppg_noise_5dB done")

    # merge waveform results from a previous full run when skipping embeds
    saved_path = OUTPUT_DIR / "robustness_metrics.json"
    if args.skip_embed and saved_path.exists():
        saved = json.loads(saved_path.read_text())
        for k, v in saved.get("perturbations", {}).items():
            results["perturbations"].setdefault(k, v)

    results["n_windows"] = n
    results["n_test_patients"] = n_patients
    results["n_waveform_windows"] = N_WAVEFORM_WINDOWS
    results["seed"] = SEED
    results["disclaimer"] = ("feature-level perturbations + clean: patient-level "
                             "bootstrap CIs over the 48-patient test set; "
                             "waveform tags: point estimates on 200 re-embedded "
                             "windows. Pipeline demonstration, not clinical "
                             "evidence.")
    (OUTPUT_DIR / "robustness_metrics.json").write_text(
        json.dumps(results, indent=1, default=str))

    # ---------- figures -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"concat": "#2a78d6", "gate": "#d67a2a", "weight": "#2a9d5b"}

    fig, ax = plt.subplots(figsize=(7, 3.6))
    levels = ["clean", "ecg_noise_20dB", "ecg_noise_10dB", "ecg_noise_5dB"]
    xpos = np.arange(len(levels))
    for mode in models:
        y = [results["clean"][mode]["6"]["auc"] if l == "clean"
             else results["perturbations"][l][mode]["auc"] for l in levels]
        ax.plot(xpos, y, marker="o", color=colors[mode], label=mode)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["clean", "SNR 20", "SNR 10", "SNR 5"], fontsize=9)
    ax.set_ylabel("6 h AUROC (test)")
    ax.set_title(f"ECG waveform noise (synthetic test set, {n_patients} patients)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_robustness_noise.png", dpi=150)
    plt.close(fig)

    cats = ["clean", "missing_ppg", "sensor_failure", "delay_15min",
            "vitals_noise_2.0x", "ecg_flatline", "ecg_saturation", "ecg_wander",
            "ecg_mains", "ppg_noise_5dB"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    xpos = np.arange(len(cats))
    w = 0.26
    for i, mode in enumerate(models):
        y = [results["clean"][mode]["6"]["auc"] if c == "clean"
             else results["perturbations"][c][mode]["auc"] for c in cats]
        ax.bar(xpos + (i - 1) * w, y, w, color=colors[mode], label=mode)
    ax.axhline(results["clean"]["concat"]["6"]["auc"], color="#999999",
               linestyle=":", linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["clean", "no PPG", "PPG dies", "15 min delay",
                        "vitals 2x", "flatline", "saturate", "wander", "mains",
                        "PPG 5 dB"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("6 h AUROC (test)")
    ax.set_title(f"Robustness sweep (synthetic test set, {n_patients} patients; "
                 "dotted line = clean concat)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_robustness_failure.png", dpi=150)
    print(f"\nsaved -> {OUTPUT_DIR / 'robustness_metrics.json'}")
    print("(feature perturbations carry patient-bootstrap CIs; waveform tags "
          "are 200-window point estimates)")


if __name__ == "__main__":
    main()
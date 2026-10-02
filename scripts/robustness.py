"""Robustness evaluation (reviewer point): noisy signals, missing PPG, ECG
artifacts, delayed measurements, sensor failure.

Evaluated on the synthetic replay set (2 held out patients, step=1 windows)
with three models trained on the synthetic cohort (cached):
  concat - v1 architecture (legacy zero fill for missing streams)
  gate   - gated fusion with per modality presence masks
  weight - static modality weights with masks
The mask aware models were trained with 10% of windows' PPG masked, so their
missing stream handling is exercised (concat never saw missing PPG in
training).

Perturbations:
  waveform level (re embedded through the real encoders):
    ECG noise at SNR 20/10/5 dB, baseline wander, 50 Hz mains, flatline
    segments, saturation, PPG noise at SNR 5 dB
  feature level (replay windows):
    vitals noise (0.5/1/2 x per column std), missing PPG, delayed
    measurements (whole vector shifted 1/2/3 windows = 5/10/15 min),
    sensor failure (PPG drops mid stay)

Honest caveat (plan rule 8): 2 replay patients -> point estimates, no
bootstrap; a pipeline demonstration, not clinical evidence.

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

from ews_risk import FEAN, ENG_COLS, SEQ_LEN, build_features_masked, train_fean  # noqa: E402
from test_fean_v2 import load_synthetic  # noqa: E402
from test_fusion import make_sequences_masked  # noqa: E402
from synth_signals import zscore_segments  # noqa: E402

EMBED_DIR = PROJECT / "cache/embeddings"
MODEL_DIR = PROJECT / "cache/models/robustness"
OUTPUT_DIR = PROJECT / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZONS = (1, 6, 24)
SEED = 7
ECG_DIM, PPG_DIM = 768, 512
N_WAVEFORM_WINDOWS = 200          # subset re embedded for waveform tags


def load_replay():
    meta = pd.read_csv(EMBED_DIR / "replay_meta.csv")
    eng = pd.read_parquet(EMBED_DIR / "replay_engineered.parquet")
    ecg = np.load(EMBED_DIR / "replay_window_ecgfm.npy")
    ppg = np.load(EMBED_DIR / "replay_window_papagei.npy")
    assert len(meta) == len(ecg) == len(ppg)
    vt = meta.sort_values(["patient_id", "t_start_h"])
    vt["hr_trend"] = vt.groupby("patient_id")["hr"].transform(lambda s: s.diff(6))
    vt["sbp_trend"] = vt.groupby("patient_id")["sbp"].transform(lambda s: s.diff(6))
    trend_map = vt.set_index(["patient_id", "t_start_h"])[["hr_trend", "sbp_trend"]]
    eng = (eng.merge(meta[["patient_id", "t_start_h"]], on=["patient_id", "t_start_h"],
                     how="right", validate="1:1")
              .set_index(["patient_id", "t_start_h"]).join(trend_map, how="left")
              .loc[list(zip(meta["patient_id"], meta["t_start_h"]))])
    feats = np.stack([build_features_masked(meta.iloc[i],
                                            eng[ENG_COLS].iloc[i].astype(float),
                                            ecg[i], ppg[i])[0]
                      for i in range(len(meta))]).astype(np.float32)
    masks = np.ones((len(meta), 4), np.float32)
    tte = (meta["arrest_h"] - meta["t_end_h"]).to_numpy()
    labels = {H: np.where(np.isnan(tte), 0, ((tte > 0) & (tte <= H)).astype(int))
              for H in HORIZONS}
    ids = pd.factorize(meta["patient_id"])[0]
    return meta, feats, masks, labels, tte, ids


def train_models(epochs):
    """Train concat/gate/weight on the synthetic cohort (cached)."""
    meta, feats, labels, tte, ids = load_synthetic()
    masks = np.ones((len(meta), 4), np.float32)
    case_pids = sorted(set(meta.loc[meta["is_case"], "patient_id"]))
    ctrl_pids = sorted(set(meta.loc[~meta["is_case"], "patient_id"]))
    val_pids = {case_pids[-1], ctrl_pids[-1]}
    tr = ~meta.patient_id.isin(val_pids).to_numpy()
    rs = np.random.default_rng(SEED)
    missing = np.zeros(len(meta), bool)
    missing[np.where(tr)[0][rs.choice(int(tr.sum()), int(0.1 * tr.sum()),
                                      replace=False)]] = True
    z = (feats - feats[tr].mean(0)) / (feats[tr].std(0) + 1e-8)
    z[missing, -PPG_DIM:] = 0.0
    masks_tr = masks.copy()
    masks_tr[missing, 3] = 0.0
    Xtr, Ltr, Ytr, Ttr, idtr, Mtr = make_sequences_masked(
        meta.loc[tr], z[tr], {H: labels[H][tr] for H in HORIZONS},
        tte[tr], ids[tr], masks_tr[tr])
    out = {}
    for mode in ("concat", "gate", "weight"):
        ckpt = MODEL_DIR / f"{mode}.pt"
        stats = MODEL_DIR / f"{mode}_stats.npz"
        if not (ckpt.exists() and stats.exists()):
            torch.manual_seed(SEED)
            m = FEAN(fusion=mode)
            if mode == "concat":
                m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr, epochs=epochs,
                                     seed=SEED, verbose=0)
            else:
                m, hist = train_fean(m, Xtr, Ltr, Ytr, Ttr, masks_tr=Mtr,
                                     epochs=epochs, seed=SEED, verbose=0)
            assert np.isfinite(hist[-1]["train_loss"])
            torch.save({"model": m.state_dict()}, ckpt)
            np.savez(stats, mean=feats[tr].mean(0), std=feats[tr].std(0))
        out[mode] = (ckpt, stats)
    return out


def predict(mode, feats, masks, labels, tte, meta, ckpt, stats):
    """6 h / 1 h AUROC on the replay set for one model."""
    st = np.load(stats)
    z = (feats - st["mean"]) / (st["std"] + 1e-8)
    X, L, Y, T, ID, M = make_sequences_masked(meta, z, labels, tte, ids_replay,
                                              masks, step=1)
    m = FEAN(fusion=mode)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    m.eval()
    with torch.no_grad():
        out = m(torch.from_numpy(X), torch.from_numpy(L),
                mask=None if mode == "concat" else torch.from_numpy(M))
    return {H: roc_auc_score(Y[H], torch.sigmoid(out["risk"][H]).numpy())
            for H in HORIZONS if Y[H].sum() > 0 and (Y[H] == 0).sum() > 0}


def embed_ecg_tag(tag, chunk_ids, chunks):
    """Re embed perturbed ECG chunks via ecgfm_env (cached by tag)."""
    emb_out = EMBED_DIR / f"rob_{tag}_ecg.npy"
    if not emb_out.exists():
        seg_in = EMBED_DIR / f"rob_{tag}_ecg.npz"
        np.savez(seg_in, ecg=zscore_segments(chunks))
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [str(Path.home() / "anaconda3/envs/ecgfm_env/bin/python"),
             str(PROJECT / "scripts/extract_ecgfm.py"),
             "--input", str(seg_in), "--output", str(emb_out),
             "--checkpoint", str(PROJECT / "weights/mimic_iv_ecg_physionet_pretrained.pt"),
             "--batch-size", "16"],
            check=True, capture_output=True, text=True, env=env, timeout=7200)
    return np.load(emb_out)                          # (n_chunks, 768)


def embed_ppg_tag(tag, segments):
    emb_out = EMBED_DIR / f"rob_{tag}_ppg.npy"
    if not emb_out.exists():
        seg_in = EMBED_DIR / f"rob_{tag}_ppg.npz"
        zs = np.stack([(s - s.mean()) / (s.std() + 1e-8) for s in segments])
        np.savez(seg_in, ppg=zs.astype(np.float32))
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [str(Path.home() / "anaconda3/envs/papagei_env/bin/python"),
             str(PROJECT / "scripts/extract_papagei.py"),
             "--input", str(seg_in), "--output", str(emb_out),
             "--weights", str(PROJECT / "weights/papagei_s.pt"),
             "--batch-size", "32"],
            check=True, capture_output=True, text=True, env=env, timeout=7200)
    return np.load(emb_out)                          # (n_chunks, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    print("loading synthetic replay set...")
    global ids_replay
    meta, feats, masks, labels, tte, ids_replay = load_replay()
    n = len(meta)
    print(f"replay windows {n}, patients {meta.patient_id.unique()}")

    print("training (or loading cached) concat/gate/weight...")
    models = train_models(args.epochs)
    print(f"  loaded: {list(models)}")

    results = {"clean": {}, "perturbations": {}}
    for mode, (ckpt, stats) in models.items():
        results["clean"][mode] = predict(mode, feats, masks, labels, tte, meta,
                                         ckpt, stats)
        print(f"  clean {mode}: {results['clean'][mode]}")

    # ---------- feature level perturbations ---------------------------------
    # P2 missing PPG
    for mode, (ckpt, stats) in models.items():
        f = feats.copy()
        f[:, -PPG_DIM:] = 0.0
        m = masks.copy()
        m[:, 3] = 0.0
        results["perturbations"].setdefault("missing_ppg", {})[mode] = \
            predict(mode, f, m, labels, tte, meta, ckpt, stats)
        print(f"  missing_ppg {mode}: "
              f"{results['perturbations']['missing_ppg'][mode]}")

    # P3 delayed measurements (whole vector shifted k windows)
    for k in (1, 2, 3):
        f = feats.copy()
        for pid in np.unique(meta["patient_id"]):
            idx = np.where(meta["patient_id"] == pid)[0]
            f[idx[k:]] = feats[idx[:-k]]
        results["perturbations"].setdefault(f"delay_{5 * k}min", {})
        for mode, (ckpt, stats) in models.items():
            results["perturbations"][f"delay_{5 * k}min"][mode] = \
                predict(mode, f, masks, labels, tte, meta, ckpt, stats)
        print(f"  delay {5 * k} min done")

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
    results["perturbations"]["sensor_failure"] = {}
    for mode, (ckpt, stats) in models.items():
        results["perturbations"]["sensor_failure"][mode] = \
            predict(mode, f, m, labels, tte, meta, ckpt, stats)
        print(f"  sensor_failure {mode}: "
              f"{results['perturbations']['sensor_failure'][mode]}")

    # P1 vitals noise (0.5/1/2 x per column std)
    for k in (0.5, 1.0, 2.0):
        rs = np.random.default_rng(SEED + int(k * 10))
        f = feats.copy()
        f[:, :7] += k * feats[:, :7].std(0) * rs.standard_normal((n, 7)).astype(
            np.float32)
        results["perturbations"].setdefault(f"vitals_noise_{k}x", {})
        for mode, (ckpt, stats) in models.items():
            results["perturbations"][f"vitals_noise_{k}x"][mode] = \
                predict(mode, f, masks, labels, tte, meta, ckpt, stats)
        print(f"  vitals noise {k}x done")

    # ---------- waveform level perturbations (re embedded) --------------------
    if not args.skip_embed:
        ecg_segs = np.load(EMBED_DIR / "replay_ecg_seg.npz")["ecg"]   # (n*6, 12, 2500)
        ppg_segs = np.load(EMBED_DIR / "replay_ppg_seg.npz")["ppg"]   # (n*3, 1250)
        rs = np.random.default_rng(SEED)
        win_idx = rs.choice(n, N_WAVEFORM_WINDOWS, replace=False)
        ecg_ids = np.concatenate([np.arange(w * 6, w * 6 + 6) for w in win_idx])
        ppg_ids = np.concatenate([np.arange(w * 3, w * 3 + 3) for w in win_idx])

        def replace_ecg_block(tag, chunks):
            emb = embed_ecg_tag(tag, ecg_ids, chunks)
            f = feats.copy()
            f[win_idx, 17:17 + ECG_DIM] = emb.reshape(len(win_idx), 6, ECG_DIM).mean(1)
            results["perturbations"].setdefault(tag, {})
            for mode, (ckpt, stats) in models.items():
                results["perturbations"][tag][mode] = \
                    predict(mode, f, masks, labels, tte, meta, ckpt, stats)
            summary = {mm: results["perturbations"][tag][mm][6]
                       for mm in models}
            print(f"  {tag} done: "
                  + ", ".join(f"{mm}: {summary[mm]:.3f}" for mm in summary))

        base = ecg_segs[ecg_ids].copy()
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

        ppg_base = ppg_segs[ppg_ids].copy()
        p2p = (ppg_base ** 2).mean(-1, keepdims=True)
        noise = rs.standard_normal(ppg_base.shape).astype(np.float32)
        ppg_noisy = (ppg_base + np.sqrt(p2p) / (10 ** (5 / 20)) * noise).astype(np.float32)
        emb = embed_ppg_tag("ppg_noise_5dB", ppg_noisy)
        f = feats.copy()
        f[win_idx, -PPG_DIM:] = emb.reshape(len(win_idx), 3, PPG_DIM).mean(1)
        results["perturbations"]["ppg_noise_5dB"] = {}
        for mode, (ckpt, stats) in models.items():
            results["perturbations"]["ppg_noise_5dB"][mode] = \
                predict(mode, f, masks, labels, tte, meta, ckpt, stats)
        print(f"  ppg_noise_5dB done")

    # merge waveform results from a previous full run when skipping embeds
    saved_path = OUTPUT_DIR / "robustness_metrics.json"
    if args.skip_embed and saved_path.exists():
        saved = json.loads(saved_path.read_text())
        for k, v in saved.get("perturbations", {}).items():
            results["perturbations"].setdefault(k, v)

    results["n_windows"] = n
    results["n_replay_patients"] = 2
    results["seed"] = SEED
    results["disclaimer"] = ("2 replay patients -> point estimates, no CIs; "
                             "pipeline demonstration, not clinical evidence")
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
        y = [results["clean"][mode][6] if l == "clean"
             else results["perturbations"][l][mode][6] for l in levels]
        ax.plot(xpos, y, marker="o", color=colors[mode], label=mode)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["clean", "SNR 20", "SNR 10", "SNR 5"], fontsize=9)
    ax.set_ylabel("6 h AUROC (replay)")
    ax.set_title("ECG waveform noise (synthetic replay, 2 patients)")
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
        y = [results["clean"][mode][6] if c == "clean"
             else results["perturbations"][c][mode][6] for c in cats]
        ax.bar(xpos + (i - 1) * w, y, w, color=colors[mode], label=mode)
    ax.axhline(results["clean"]["concat"][6], color="#999999", linestyle=":",
               linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["clean", "no PPG", "PPG dies", "15 min delay",
                        "vitals 2x", "flatline", "saturate", "wander", "mains",
                        "PPG 5 dB"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("6 h AUROC (replay)")
    ax.set_title("Robustness sweep (synthetic replay, 2 patients; "
                 "dotted line = clean concat)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_robustness_failure.png", dpi=150)
    print(f"\nsaved -> {OUTPUT_DIR / 'robustness_metrics.json'}")
    print("(honest caveat: 2 replay patients, point estimates only)")


if __name__ == "__main__":
    main()
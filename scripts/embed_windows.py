"""Batch-embed 5-min windows with ECG-FM and PaPaGei (subprocess env design).

Run with the main env (pyprime):
    python scripts/embed_windows.py --windows cache/embeddings/train_windows.csv --tag train

The windows CSV is a subset of the synthetic vitals table (patient_id, t_start_h,
hr, sbp, dbp, amp_ppg, pt_ms, rrsd, has_ppg, qf_ecg, qf_ppg, qf_abp). Segments
are regenerated deterministically per (patient, window), degraded per the qf_*
flags, preprocessed (ECG -> 500 Hz, 12-lead, 5 s chunks; PPG -> 125 Hz, 10 s
chunks), and embedded by the extractor scripts in their own conda envs.

Outputs (skipped if already present): cache/embeddings/{tag}_window_ecgfm.npy
(W, 768), {tag}_window_papagei.npy (W, 512), {tag}_meta.csv (row order matches).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from synth_signals import cut_segment, lead2_to_12leads, zscore_segments  # noqa: E402

ENV_PY = {
    "ecg": Path.home() / "anaconda3/envs/ecgfm_env/bin/python",
    "ppg": Path.home() / "anaconda3/envs/papagei_env/bin/python",
}
WEIGHTS = {
    "ecg": PROJECT / "weights/mimic_iv_ecg_physionet_pretrained.pt",
    "ppg": PROJECT / "weights/papagei_s.pt",
}
ECG_CHUNKS, PPG_CHUNKS = 6, 3    # 30 s -> 6x5 s (ECG) / 3x10 s (PPG)


def resample(x, fs_in, fs_out):
    """Linear-interpolation resampling (upsampling path, as in the notebook)."""
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x)


def build_segments(df, kind):
    """Cut + preprocess all windows. Returns (segments array, window index per
    segment)."""
    segs, win_idx = [], []
    for i, row in df.iterrows():
        seg = cut_segment(row["patient_id"], row["t_start_h"], row)
        if kind == "ecg":
            ecg500 = resample(seg["ecg"], 125.0, 500.0)
            chunks = ecg500[:len(ecg500) // 2500 * 2500].reshape(-1, 2500)
            chunks = np.stack([lead2_to_12leads(c) for c in chunks])
            segs.append(zscore_segments(chunks))
        else:
            if not bool(row["has_ppg"]):
                segs.append(np.zeros((PPG_CHUNKS, 1250), np.float32))
            else:
                chunks = seg["ppg"][:len(seg["ppg"]) // 1250 * 1250].reshape(-1, 1250)
                segs.append(zscore_segments(chunks))
        win_idx.append(np.full(len(segs[-1]), i))
    return np.concatenate(segs), np.concatenate(win_idx)


def run_extractor(env_key, script, extra):
    cmd = [str(ENV_PY[env_key]), str(PROJECT / script)] + extra
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"   # keep ~/.local packages from shadowing env packages
    print(f"$ {Path(cmd[1]).name} {' '.join(extra[:2])} ...")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"{script} failed with exit code {res.returncode}")
    print(res.stdout.strip())


def pool_to_windows(seg_emb, win_idx, n_windows):
    """Mean-pool segment embeddings per window."""
    out = np.zeros((n_windows, seg_emb.shape[1]), np.float32)
    for w in range(n_windows):
        m = win_idx == w
        if m.any():
            out[w] = seg_emb[m].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    emb_dir = PROJECT / "cache/embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_ecg = emb_dir / f"{args.tag}_window_ecgfm.npy"
    out_ppg = emb_dir / f"{args.tag}_window_papagei.npy"
    out_meta = emb_dir / f"{args.tag}_meta.csv"
    if out_ecg.exists() and out_ppg.exists() and out_meta.exists():
        print(f"[skip] {args.tag} embeddings already cached")
        return

    df = pd.read_csv(args.windows)
    print(f"[{args.tag}] {len(df)} windows")

    for kind in ("ecg", "ppg"):
        segs, win_idx = build_segments(df, kind)
        seg_in = emb_dir / f"{args.tag}_{kind}_seg.npz"
        np.savez(seg_in, win_idx=win_idx, **{kind: segs})   # extractors read key 'ecg'/'ppg'
        print(f"[{args.tag}] {kind} segments: {segs.shape}")
        seg_out = emb_dir / f"{args.tag}_{kind}_emb.npy"
        if kind == "ecg":
            run_extractor("ecg", "scripts/extract_ecgfm.py",
                          ["--input", str(seg_in), "--output", str(seg_out),
                           "--checkpoint", str(WEIGHTS["ecg"]), "--batch-size", "16"])
        else:
            run_extractor("ppg", "scripts/extract_papagei.py",
                          ["--input", str(seg_in), "--output", str(seg_out),
                           "--weights", str(WEIGHTS["ppg"]), "--batch-size", "32"])
        seg_emb = np.load(seg_out)
        pooled = pool_to_windows(seg_emb, win_idx, len(df))
        np.save(out_ecg if kind == "ecg" else out_ppg, pooled)
        print(f"[{args.tag}] {kind} window embeddings: {pooled.shape}")

    df.to_csv(out_meta, index=False)
    print(f"[{args.tag}] done -> {out_ecg.name}, {out_ppg.name}, {out_meta.name}")


if __name__ == "__main__":
    main()
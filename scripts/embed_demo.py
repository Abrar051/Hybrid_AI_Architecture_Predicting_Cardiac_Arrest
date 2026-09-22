"""End-to-end embedding demo: synthetic segments -> ECG-FM + PaPaGei embeddings.

Run with the main env (pyprime):
    python scripts/embed_demo.py

This mirrors the Section 6 design from the plan: the main environment generates
segments from the cached synthetic cohort, writes them under cache/embeddings/,
and calls the extractor scripts in their own conda envs via subprocess
(ecgfm_env for ECG-FM, papagei_env for PaPaGei). Embeddings land in
cache/embeddings/*.npy.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))

from synth_signals import generate_waveform_segment, lead2_to_12leads, zscore_segments  # noqa: E402

ENV_PY = {
    "ecg": Path.home() / "anaconda3/envs/ecgfm_env/bin/python",
    "ppg": Path.home() / "anaconda3/envs/papagei_env/bin/python",
}
WEIGHTS = {
    "ecg": PROJECT / "weights/mimic_iv_ecg_physionet_pretrained.pt",
    "ppg": PROJECT / "weights/papagei_s.pt",
}


def resample(x, fs_in, fs_out):
    """Linear-interpolation resampling (upsampling path, as in the notebook)."""
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x)


def pick_demo_windows(vitals, events):
    """Two clean windows of one case patient (far + near arrest) and two of one
    control patient."""
    case_pid = next(int(p) for p in events["patient_id"]
                    if bool(vitals.loc[vitals["patient_id"] == p, "has_ppg"].iloc[0]))
    arrest_h = float(vitals.loc[vitals["patient_id"] == case_pid, "arrest_h"].iloc[0])
    ctrl_pid = int(vitals.loc[~vitals["is_case"], "patient_id"].iloc[0])

    rows = []
    g = vitals[vitals["patient_id"] == case_pid]
    far = g[(g["t_start_h"] < arrest_h - 8.0) & (g["qf_ecg"] == "ok") &
            (g["qf_ppg"] == "ok")].iloc[0]
    near = g[(g["t_start_h"] < arrest_h - 1.0) & (g["t_start_h"] >= arrest_h - 3.0) &
             (g["qf_ecg"] == "ok") & (g["qf_ppg"] == "ok")].iloc[-1]
    rows += [far, near]
    g2 = vitals[vitals["patient_id"] == ctrl_pid]
    ctrl = g2[(g2["qf_ecg"] == "ok") & (g2["qf_ppg"] == "ok")]
    rows += [ctrl.iloc[len(ctrl) // 3], ctrl.iloc[2 * len(ctrl) // 3]]
    return rows


def build_ecg_segments(row):
    """One 30 s window -> six 5 s, 12-lead, 500 Hz z-scored segments (2500 each)."""
    seg = generate_waveform_segment(
        hr=row["hr"], sbp=row["sbp"], dbp=row["dbp"],
        amp_ecg=1.0, amp_ppg=row["amp_ppg"], pt_ms=row["pt_ms"],
        rr_sd=row["rrsd"],
        seed=int(row["patient_id"]) * 1_000_000 + int(round(row["t_start_h"] * 60.0)))
    ecg500 = resample(seg["ecg"], 125.0, 500.0)
    chunks = ecg500[:len(ecg500) // 2500 * 2500].reshape(-1, 2500)
    leads = np.stack([lead2_to_12leads(c) for c in chunks])   # (6, 12, 2500)
    return zscore_segments(leads)


def build_ppg_segments(row):
    """One 30 s window -> three 10 s, 125 Hz z-scored segments (1250 each)."""
    seg = generate_waveform_segment(
        hr=row["hr"], sbp=row["sbp"], dbp=row["dbp"],
        amp_ecg=1.0, amp_ppg=row["amp_ppg"], pt_ms=row["pt_ms"],
        rr_sd=row["rrsd"],
        seed=int(row["patient_id"]) * 1_000_000 + int(round(row["t_start_h"] * 60.0)))
    chunks = seg["ppg"][:len(seg["ppg"]) // 1250 * 1250].reshape(-1, 1250)
    return zscore_segments(chunks)


def run_extractor(env_key, script, extra):
    cmd = [str(ENV_PY[env_key]), str(PROJECT / script)] + extra
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"   # keep ~/.local packages from shadowing env packages
    print(f"$ {' '.join(cmd[-2:])} {' '.join(extra[:2])} ...")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"{script} failed with exit code {res.returncode}")
    print(res.stdout.strip())


def main():
    emb_dir = PROJECT / "cache/embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    vitals = pd.read_parquet(PROJECT / "cache/synthetic_vitals.parquet")
    events = pd.read_csv(PROJECT / "cache/synthetic_events.csv")
    rows = pick_demo_windows(vitals, events)
    print("demo windows:")
    for r in rows:
        print(f"  patient {int(r['patient_id'])}  t={r['t_start_h']:.1f} h  "
              f"hr={r['hr']:.0f}  sbp={r['sbp']:.0f}")

    ecg = np.concatenate([build_ecg_segments(r) for r in rows])   # (24, 12, 2500)
    ppg = np.concatenate([build_ppg_segments(r) for r in rows])   # (12, 1250)
    ecg_in = emb_dir / "demo_ecg.npz"
    ppg_in = emb_dir / "demo_ppg.npz"
    np.savez(ecg_in, ecg=ecg)
    np.savez(ppg_in, ppg=ppg)
    print(f"segments: ECG {ecg.shape} -> {ecg_in.name}, PPG {ppg.shape} -> {ppg_in.name}")

    ecg_out = emb_dir / "demo_ecgfm.npy"
    run_extractor("ecg", "scripts/extract_ecgfm.py",
                  ["--input", str(ecg_in), "--output", str(ecg_out),
                   "--checkpoint", str(WEIGHTS["ecg"])])
    ppg_out = emb_dir / "demo_papagei.npy"
    run_extractor("ppg", "scripts/extract_papagei.py",
                  ["--input", str(ppg_in), "--output", str(ppg_out),
                   "--weights", str(WEIGHTS["ppg"])])

    e_ecg = np.load(ecg_out)
    e_ppg = np.load(ppg_out)
    assert e_ecg.shape == (len(ecg), 768), e_ecg.shape
    assert e_ppg.shape == (len(ppg), 512), e_ppg.shape
    print(f"\nembeddings: ECG-FM {e_ecg.shape} (mean {e_ecg.mean():.3f}, "
          f"std {e_ecg.std():.3f}) | PaPaGei-S {e_ppg.shape} "
          f"(mean {e_ppg.mean():.3f}, std {e_ppg.std():.3f})")

    # sanity: same-window segments should be more similar than different-window ones
    def within_vs_across(emb, segs_per_window):
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sim = emb @ emb.T
        n = len(emb)
        within = np.mean([sim[i, j] for i in range(n) for j in range(n)
                          if i != j and i // segs_per_window == j // segs_per_window])
        across = np.mean([sim[i, j] for i in range(n) for j in range(n)
                          if i // segs_per_window != j // segs_per_window])
        return within, across
    w, a = within_vs_across(e_ecg, 6)
    print(f"ECG-FM mean cosine sim: within-window {w:.3f} vs across-window {a:.3f}")
    w, a = within_vs_across(e_ppg, 3)
    print(f"PaPaGei mean cosine sim: within-window {w:.3f} vs across-window {a:.3f}")
    print("\nembedding demo OK")


if __name__ == "__main__":
    main()
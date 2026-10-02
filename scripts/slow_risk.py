"""Section 9 (optional): slow-risk layer - QT, T-wave alternans and long-term
HRV markers aggregated over 3 h blocks, kept separate from the short-horizon
FEAN model (per the plan).

Plan note: "Use ECG-SCD code on NTUH data, or a daily-aggregated risk score."
The NTUH ECG-SCD route needs credentialed access that is not available yet, so
this builds the marker-extraction half now and evaluates it as a standalone
block-level model on SDDB (open access), which at least answers whether slow
ECG markers alone separate pre-arrest blocks from far blocks.

Markers per 30 s segment (lead 1, 125 Hz), all honest proxies on a single-lead
Holter without fiducial libraries:
  - QTc (Bazett): R peaks -> Q onset -> T end (simplified tangent method)
  - T-wave alternans index: power at 0.5 cycles/beat of the beat-level T
    amplitude series
  - QRS width estimate (Q onset to S end)
  - HR, ectopy burden, RMSSD from RR intervals (as in ecg_features)
Blocks of 3 h: median + IQR of segment markers, SDNN/SDANN proxies from
segment mean RR, HR trend slope, max TWA. Labels: "arrest within H hours of
block end" with a 0.5 h margin rule (blocks whose tte falls in (H, H+0.5] are
dropped for that horizon to avoid boundary ambiguity).

Model: logistic regression, 5-fold CV grouped by record, patient-level
bootstrap CIs (plan rule 8).

Rule 7: --selftest runs the marker extraction on cached synthetic ECG
segments and sanity-checks the marker ranges before touching real data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from sddb_utils import (VF_ONSET, FLOOR_S, NEG_OFFSET_S, FS_TARGET,  # noqa: E402
                        onset_seconds, record_duration, load_span, ecg_features,
                        require_sddb)

OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BLOCK_H = 3.0
LABEL_MARGIN_H = 0.5
SEED = 7
MARKER_COLS = ["hr", "rr_mean_s", "rr_rmssd_s", "ectopy_burden", "qtc_s",
               "twa_index", "qrs_width_ms"]


def segment_markers(ecg30):
    """QT/TWA/QRS markers from one 30 s single-lead ECG at FS_TARGET Hz."""
    x = ecg30 - np.nanmedian(ecg30)
    base = np.nanmedian(ecg30)
    fs = FS_TARGET
    peaks, _ = find_peaks(x, distance=int(0.3 * fs),
                          height=max(0.6 * np.nanstd(x), 0.3))
    rr = np.diff(peaks) / fs
    if len(rr) < 4:
        return {c: np.nan for c in MARKER_COLS}
    qts, twa_amps, qrs_w = [], [], []
    dx = np.abs(np.diff(x))
    for k in range(1, len(peaks) - 1):
        r = peaks[k]
        rr_i = (peaks[k] - peaks[k - 1]) / fs
        # QRS onset by derivative: scan back from R; onset = where |dx| drops
        # below 20% of the max |dx| in the 150 ms window (flat before the Q
        # downstroke). Robust to baseline wander, unlike zero crossings.
        q_win = dx[max(0, r - int(0.15 * fs)): r]
        if len(q_win) < 3:
            continue
        thr = 0.2 * q_win.max()
        below = np.where(q_win[::-1] <= thr)[0]
        q_on = r - len(q_win) + (len(q_win) - below[0] - 1) if len(below) else \
            r - int(0.04 * fs)
        # T end: T peak, then simplified tangent: extrapolate the steepest
        # descending slope after the peak to the baseline
        t_hi = min(r + int(0.44 * fs), peaks[k + 1] - int(0.08 * fs))
        t_win = x[r + int(0.10 * fs): t_hi]
        if len(t_win) < 10:
            continue
        t_peak = r + int(0.10 * fs) + int(np.argmax(np.abs(t_win)))
        t_amp = x[t_peak]
        after = x[t_peak: t_hi]
        if len(after) < 5 or abs(t_amp) < 0.05 * np.nanstd(x):
            continue
        slope = np.diff(after)
        t_end = t_peak + int(np.argmax(slope < 0)) if (slope < 0).any() else \
            t_peak + int(0.10 * fs)
        # tangent from the steepest descending point to baseline
        steep = t_peak + int(np.argmin(slope))
        if abs(x[steep]) > 1e-6:
            tan = x[steep] + slope[np.argmin(slope)] * np.arange(len(x) - steep)
            cross = np.where(np.abs(tan) <= max(0.02 * abs(t_amp), 1e-3))[0]
            if len(cross):
                t_end = min(steep + int(cross[0]), t_hi)
        # S end for QRS width
        s_win = x[r: r + int(0.10 * fs)]
        if len(s_win) > 2:
            s_end = r + int(np.argmin(s_win))
            for i in range(s_end, min(r + int(0.14 * fs), len(x))):
                if x[i] >= 0:
                    s_end = i
                    break
        qts.append((t_end - q_on) / fs)
        twa_amps.append(t_amp)
        qrs_w.append((s_end - q_on) * 1000 / fs)

    out = ecg_features(ecg30)
    if len(qts) < 3:
        return {c: np.nan for c in MARKER_COLS}
    rr_med = float(np.median(rr))
    # T-wave alternans index: power at 0.5 cycles/beat of the beat-level T
    # amplitude series, normalized by the mean |T amplitude|.
    twa = 0.0
    a = np.asarray(twa_amps)
    if len(a) >= 8:
        a = a - a.mean()
        spec = np.abs(np.fft.rfft(a))
        if len(a) % 2 == 0:
            twa = float(2 * spec[len(a) // 2] / len(a) /
                        max(np.mean(np.abs(a)), 1e-9))
    return dict(hr=out["hr"], rr_mean_s=out["rr_mean_s"], rr_rmssd_s=out["rr_rmssd_s"],
                ectopy_burden=out["ectopy_burden"],
                qtc_s=float(np.median(qts) / np.sqrt(max(rr_med, 0.2))),
                twa_index=twa, qrs_width_ms=float(np.median(qrs_w)))


def block_rows(rec, t0, t1):
    """Marker rows for the 3 h block [t0, t1): one row per 5 min window."""
    rows = []
    sig, _ = load_span(rec, t0, t1)
    win = 5 * 60 * int(FS_TARGET)
    for w in range(len(sig) // win):
        seg = sig[w * win + (win - 30 * int(FS_TARGET)) // 2:
                  w * win + (win + 30 * int(FS_TARGET)) // 2]
        if not np.isfinite(seg).all():
            continue
        rows.append(segment_markers(seg[:, 0]))
    return rows


def build_blocks():
    """3 h blocks from NEG_OFFSET_S to onset - FLOOR_S per record."""
    recs, starts, ends = [], [], []
    for rec in VF_ONSET:
        onset = onset_seconds(rec)
        total = record_duration(rec)
        s = NEG_OFFSET_S
        while s + BLOCK_H * 3600 <= onset - FLOOR_S:
            recs.append(rec)
            starts.append(s)
            ends.append(s + BLOCK_H * 3600)
            s += BLOCK_H * 3600
    return pd.DataFrame({"record": recs, "t0_s": starts, "t1_s": ends})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run marker extraction on cached synthetic ECG only")
    args = ap.parse_args()

    if args.selftest:
        # Rule 7: sanity-check markers on synthetic segments before real data.
        segs = np.load(PROJECT / "cache/embeddings/train_ecg_seg.npz")["ecg"]
        # chunks are (n, 12, 2500) at 500 Hz; take lead II, decimate to 125 Hz
        lead2 = segs[::50, 1, :]                      # ~500 segments
        x125 = lead2[:, ::4].astype(np.float32)
        all_m = [segment_markers(s) for s in x125]
        df = pd.DataFrame(all_m)
        print(f"synthetic marker sanity ({len(df)} segments):")
        for c in MARKER_COLS:
            v = df[c].dropna()
            print(f"  {c:<14} n={len(v):>3} median {v.median():.3f} "
                  f"min {v.min():.3f} max {v.max():.3f}")
        qtc_ok = 0.25 < df["qtc_s"].dropna().median() < 0.55
        assert qtc_ok, "QTc outside plausible range on synthetic data"
        assert (df["qrs_width_ms"].dropna() > 20).all(), "QRS width implausible"
        print("selftest PASS: markers within plausible ranges on synthetic ECG")
        return

    require_sddb()
    print("building 3 h blocks...")
    blocks = build_blocks()
    feats, meta_rows = [], []
    n_blocks = 0
    for _, b in blocks.iterrows():
        rows = block_rows(b["record"], float(b["t0_s"]), float(b["t1_s"]))
        if len(rows) < 6:                             # need enough segments
            continue
        df = pd.DataFrame(rows)
        onset = onset_seconds(b["record"])
        agg = {}
        for c in MARKER_COLS:
            v = df[c].dropna()
            if len(v) < 3:
                agg[c + "_med"] = np.nan
                agg[c + "_iqr"] = np.nan
            else:
                agg[c + "_med"] = v.median()
                agg[c + "_iqr"] = np.subtract(*np.percentile(v, [75, 25]))
        rr = df["rr_mean_s"].dropna()
        agg["sdn_proxy_s"] = rr.std() if len(rr) > 3 else np.nan    # SDANN proxy
        agg["hr_slope_per_h"] = (np.polyfit(np.arange(len(df)), df["hr"].fillna(
            df["hr"].median()), 1)[0] * (60 / 5) if len(df) > 3 else np.nan)
        agg["twa_max"] = df["twa_index"].max() if df["twa_index"].notna().any() else np.nan
        agg["record"] = b["record"]
        agg["t1_s"] = b["t1_s"]
        agg["onset_s"] = onset
        meta_rows.append(agg)
        n_blocks += 1
    meta = pd.DataFrame(meta_rows)
    meta["tte_h"] = (meta["onset_s"] - meta["t1_s"]) / 3600.0
    print(f"blocks: {n_blocks} across {meta['record'].nunique()} records")
    feats = meta[[c for c in meta.columns if c not in
                  ("record", "t1_s", "onset_s", "tte_h")]].astype(float)
    feats = feats.fillna(feats.median())

    results = {}
    rec_codes = pd.factorize(meta["record"])[0]
    for H in (1, 6):
        pos = (meta["tte_h"] > 0.25) & (meta["tte_h"] <= H)
        amb = (meta["tte_h"] > H) & (meta["tte_h"] <= H + LABEL_MARGIN_H)
        use = pos | ~amb                              # drop boundary blocks
        y = pos[use].astype(int).to_numpy()
        print(f"\n{H} h horizon: {y.sum()} positives, {len(y) - y.sum()} negatives, "
              f"{int(amb.sum())} ambiguous blocks dropped")
        if y.sum() == 0 or (y == 0).sum() == 0:
            results[f"{H}h"] = {"error": "no valid split"}
            continue
        X = feats[use].to_numpy()
        rs = np.random.default_rng(SEED)
        order = rs.permutation(len(np.unique(rec_codes)))
        folds = np.array_split(order, 5)
        pred = np.zeros(len(y))
        for test_idx in folds:
            te = np.isin(rec_codes[use], np.unique(rec_codes)[test_idx])
            tr = ~te
            mu, sd = X[tr].mean(0), X[tr].std(0)
            z = (X - mu) / (sd + 1e-8)
            clf = LogisticRegression(max_iter=5000)
            clf.fit(z[tr], y[tr])
            pred[te] = clf.predict_proba(z[te])[:, 1]

        def boot(metric):
            s = []
            recs = np.unique(rec_codes[use])
            for _ in range(1000):
                idx = resample(np.arange(len(recs)))
                m = np.isin(rec_codes[use], recs[idx])
                if y[m].sum() > 0 and (y[m] == 0).sum() > 0:
                    s.append(metric(y[m], pred[m]))
            lo, hi = np.percentile(s, [2.5, 97.5])
            return float(np.mean(s)), float(lo), float(hi)

        auc, alo, ahi = boot(roc_auc_score)
        ap, aplo, aphi = boot(average_precision_score)
        results[f"{H}h"] = {"auroc": [auc, alo, ahi], "auprc": [ap, aplo, aphi],
                            "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                            "n_ambiguous_dropped": int(amb.sum())}
        print(f"  slow-marker LR: AUROC {auc:.3f} [{alo:.3f}, {ahi:.3f}] | "
              f"AUPRC {ap:.3f} [{aplo:.3f}, {aphi:.3f}]")

    results["n_blocks"] = n_blocks
    results["n_records"] = meta["record"].nunique()
    results["seed"] = SEED
    results["disclaimer"] = ("slow-marker layer on Holter data, 20 patients; "
                             "research prototype, not clinical validation")
    (OUTPUT_DIR / "slow_risk_metrics.json").write_text(
        json.dumps(results, indent=1, default=str))

    # trajectory figure: LR score per block for the longest pre-arrest record
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec = meta.groupby("record")["tte_h"].max().idxmax()
    m = meta["record"] == rec
    t = (meta.loc[m, "t1_s"] - meta.loc[m, "onset_s"]).to_numpy() / 3600.0
    Xall = feats.to_numpy()
    zall = (Xall - Xall.mean(0)) / (Xall.std(0) + 1e-8)
    y6 = ((meta["tte_h"] > 0.25) & (meta["tte_h"] <= 6)).astype(int).to_numpy()
    clf = LogisticRegression(max_iter=5000).fit(zall, y6)
    score = clf.predict_proba(zall[m.to_numpy()])[:, 1]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, score, color="#7a5bd6", linewidth=1.6, marker="o", markersize=4)
    ax.axvline(0, color="#d03b3b", linestyle="--", linewidth=1.2)
    ax.set_xlabel("hours relative to VF onset (block end)")
    ax.set_ylabel("slow-marker 6 h risk score (LR)")
    ax.set_title(f"SDDB record {rec}: slow-risk trajectory (3 h blocks, "
                 "fitted on all records)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_slow_risk_trajectory.png", dpi=150)
    print(f"\nsaved -> {OUTPUT_DIR / 'slow_risk_metrics.json'}")
    print("(honest caveat: 20 Holter patients, single lead, marker proxies; "
          "the slow-risk layer is separate from the short-horizon FEAN model)")


if __name__ == "__main__":
    main()
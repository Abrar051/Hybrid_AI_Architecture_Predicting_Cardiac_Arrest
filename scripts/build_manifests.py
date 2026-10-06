"""Build the synthetic cohort, splits, and dev/test window manifests.

Sections 1/2/4 of the notebook in one script (240-patient rebuild):
  - cohort vitals + arrest events (cached parquet/csv; force with --force)
  - window labels (kept for the summary + per-horizon prevalence print)
  - splits.json: test = latest-admitted 20% (temporal), val = 20% of dev
    stratified by case/control, train = dev - val
  - dev manifest: PPG-having dev patients, 12 h spans (cases end at the
    pre-arrest floor, controls centered mid-stay)
  - test manifest: ALL test patients, full stays (cases end at the floor)
  - engineered features per tag (cached parquet) + 30-min trend join

Usage: python scripts/build_manifests.py [--force]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from config import *  # noqa: E402,F403
from synth_signals import cut_segment  # noqa: E402
from ews_risk import ENG_COLS  # noqa: E402


def _draw_qf(rs):
    """Random ground-truth quality flag for one stream-window."""
    r = rs.random()
    if r < 0.02:
        return "flat"
    if r < 0.03:
        return "sat"
    if r < 0.05:
        return "noise"
    return "ok"


def build_synthetic_cohort(seed=RANDOM_SEED, force=False):
    """Generate vitals trajectories + arrest events. Returns (vitals, events)."""
    if not force and CACHE_VITALS.exists() and CACHE_EVENTS.exists():
        vitals = pd.read_parquet(CACHE_VITALS)
        events = pd.read_csv(CACHE_EVENTS)
        print(f"loaded cached cohort: {len(vitals):,} windows, {len(events)} events "
              f"(delete cache/ or pass --force to rebuild)")
        return vitals, events

    rs = np.random.default_rng(seed)
    rows, event_rows = [], []
    win_h = WINDOW_MIN / 60.0

    for pid in range(N_PATIENTS):
        stay_h = rs.uniform(STAY_MIN_H, STAY_MAX_H)
        admit_h = rs.uniform(0.0, 365.0 * 24.0)     # admission time within a year
        is_case = bool(rs.random() < P_ARREST)
        has_ppg = bool(rs.random() >= P_MISSING_PPG)
        arrest_h = None
        if is_case:
            arrest_h = rs.uniform(ARREST_MIN_H, min(ARREST_MAX_H, stay_h - 0.5))

        hr0 = rs.normal(78, 6)
        sbp0 = rs.normal(126, 8)
        dbp0 = rs.normal(76, 6)
        spo0 = rs.normal(97.5, 1.0)
        phase = rs.uniform(0.0, 2.0 * np.pi)

        t = 0.0
        while t < stay_h:
            hr = hr0 + 3.0 * np.sin(2 * np.pi * t / 8.0 + phase) + rs.normal(0, 1.5)
            sbp = sbp0 + 6.0 * np.sin(2 * np.pi * t / 12.0 + phase) + rs.normal(0, 3.0)
            dbp = dbp0 + 3.0 * np.sin(2 * np.pi * t / 12.0 + phase) + rs.normal(0, 2.0)
            spo2 = spo0 + rs.normal(0, 0.4)
            amp_ppg = 1.0 if has_ppg else np.nan
            pt_ms, rrsd = 250.0, 0.04

            if is_case and (arrest_h - t) <= DRIFT_H and t < arrest_h:
                p = (1.0 - (arrest_h - t) / DRIFT_H) ** 2   # 0 -> 1 approaching arrest
                hr += 55.0 * p
                sbp -= 45.0 * p
                dbp -= 25.0 * p
                spo2 -= 12.0 * p
                amp_ppg *= 1.0 - 0.6 * p
                pt_ms += 70.0 * p
                rrsd = max(0.008, rrsd - 0.025 * p)

            if is_case and t >= arrest_h:
                # post-arrest: disorganized signal, excluded downstream (plan rule 4)
                hr, sbp = rs.uniform(30, 170), rs.uniform(60, 110)
                dbp, spo2 = rs.uniform(30, 70), rs.uniform(70, 95)
                amp_ppg = 0.1 if has_ppg else np.nan
                rrsd = rs.uniform(0.02, 0.15)

            rows.append(dict(
                patient_id=pid, is_case=is_case, has_ppg=has_ppg,
                stay_h=stay_h, admit_h=admit_h, arrest_h=arrest_h,
                t_start_h=round(t, 4), t_end_h=round(t + win_h, 4),
                hr=hr, sbp=sbp, dbp=dbp, spo2=spo2,
                amp_ppg=amp_ppg, pt_ms=pt_ms, rrsd=rrsd,
                qf_ecg=_draw_qf(rs), qf_ppg=_draw_qf(rs) if has_ppg else "missing",
                qf_abp=_draw_qf(rs),
            ))
            t += win_h

        if is_case:
            event_rows.append(dict(patient_id=pid, arrest_time_h=round(arrest_h, 4),
                                   stay_start_epoch_h=round(admit_h, 4),
                                   source="synthetic"))

    vitals = pd.DataFrame(rows)
    events = pd.DataFrame(event_rows)
    vitals.to_parquet(CACHE_VITALS, index=False)
    events.to_csv(CACHE_EVENTS, index=False)
    print(f"built cohort: {len(vitals):,} windows, {len(events)} events -> cache/")
    return vitals, events


def label_summary(vitals):
    """Per-horizon label stats over the full cohort (Section 2 summary, with
    the pre-arrest floor + post-arrest exclusion)."""
    floor_h = EXCLUDE_BEFORE_ARREST_MIN / 60.0
    out = {}
    case = vitals[vitals["is_case"]]
    ctrl = vitals[~vitals["is_case"]]
    for H in HORIZONS:
        tte = (case["arrest_h"] - case["t_end_h"]).to_numpy()
        pos = int(((tte > floor_h) & (tte <= H)).sum())
        excl = int(((tte > 0) & (tte <= floor_h)).sum() + (tte <= 0).sum())
        neg = len(case) - pos - excl + len(ctrl)
        out[H] = {"positive": pos, "negative": neg, "excluded": excl}
        print(f"  horizon {H:>2}h: positive {pos:>6}  negative {neg:>6}  "
              f"excluded {excl:>6}  prevalence {pos / (pos + neg):.4f}")
    return out


def make_splits(vitals):
    """Test = latest-admitted 20%; val = stratified 20% of dev; persisted."""
    admit = vitals.groupby("patient_id")["admit_h"].first().sort_values()
    n_test = int(np.ceil(TEMPORAL_HOLDOUT_FRAC * len(admit)))
    test_pids = sorted(int(p) for p in admit.index[-n_test:])
    dev_pids = sorted(int(p) for p in admit.index[:-n_test])
    assert not (set(test_pids) & set(dev_pids)), "temporal hold-out overlaps dev"

    case_map = vitals.groupby("patient_id")["is_case"].first()
    rs = np.random.default_rng(RANDOM_SEED)
    val_pids = []
    for cls in (True, False):
        pool = [p for p in dev_pids if bool(case_map.loc[p]) == cls]
        n_val = max(1, int(np.ceil(VAL_FRAC * len(pool))))
        val_pids += [int(p) for p in rs.choice(np.array(sorted(pool)),
                                               size=n_val, replace=False)]
    val_pids = sorted(val_pids)
    train_pids = [p for p in dev_pids if p not in val_pids]
    splits = {"test": test_pids, "dev": dev_pids, "val": val_pids,
              "train": train_pids, "seed": RANDOM_SEED}
    SPLITS_FILE.write_text(json.dumps(splits, indent=1))

    def cnt(pids):
        c = int(sum(bool(case_map.loc[p]) for p in pids))
        return len(pids), c
    for k in ("train", "val", "test"):
        n, c = cnt(splits[k])
        print(f"  split {k:<5}: {n:>3} patients ({c:>3} cases)")
    return splits


def build_manifests(vitals, events, splits):
    """dev: PPG-having dev patients, 12 h spans. test: ALL test patients, full
    stays. Case windows end at the pre-arrest floor (plan rule 4)."""
    floor_h = EXCLUDE_BEFORE_ARREST_MIN / 60.0
    has_ppg = set(vitals.loc[vitals["has_ppg"], "patient_id"].unique())
    dev_ppg = [p for p in splits["dev"] if p in has_ppg]

    def case_windows(pid, span_h=None):
        g = vitals[vitals["patient_id"] == pid]
        ah = float(g["arrest_h"].iloc[0])
        sel = g[g["t_end_h"] <= ah - floor_h]
        if span_h:
            sel = sel[sel["t_start_h"] >= ah - span_h]
        return sel

    def ctrl_windows(pid, span_h=None):
        g = vitals[vitals["patient_id"] == pid]
        if span_h:
            mid = float(g["stay_h"].iloc[0]) / 2.0
            return g[(g["t_start_h"] >= mid - span_h / 2.0)
                     & (g["t_end_h"] <= mid + span_h / 2.0)]
        return g

    dev_parts = [case_windows(p, 12.0) if bool(vitals.loc[vitals["patient_id"] == p,
                                                         "is_case"].iloc[0])
                 else ctrl_windows(p, 12.0) for p in dev_ppg]
    dev_rows = pd.concat(dev_parts)[MANIFEST_COLS].reset_index(drop=True)

    test_parts = []
    for p in splits["test"]:
        if bool(vitals.loc[vitals["patient_id"] == p, "is_case"].iloc[0]):
            test_parts.append(case_windows(p))
        else:
            test_parts.append(ctrl_windows(p))
    test_rows = pd.concat(test_parts)[MANIFEST_COLS].reset_index(drop=True)

    assert not (set(dev_rows.patient_id) & set(test_rows.patient_id)), \
        "dev/test patient overlap"
    dev_rows.to_csv(EMBED_DIR / "dev_windows.csv", index=False)
    test_rows.to_csv(EMBED_DIR / "test_windows.csv", index=False)
    print(f"manifests: dev {len(dev_rows)} windows ({dev_rows.patient_id.nunique()} "
          f"patients) | test {len(test_rows)} windows "
          f"({test_rows.patient_id.nunique()} patients)")
    return dev_rows, test_rows


# ---------- engineered features (notebook cell 18, unchanged math) -----------

def _band_power(x, fs, lo, hi):
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    p = np.abs(np.fft.rfft(x - np.nanmean(x))) ** 2
    m = (f >= lo) & (f < hi)
    return p[m].sum() if m.any() else 0.0


def quality_gate(x, fs, sat_frac=0.05, snr_thresh=2.0):
    """Rule-based per-stream quality gate (notebook cell 13)."""
    flags = {"flatline": False, "saturation": False, "snr_low": False, "missing": False}
    x = np.asarray(x, dtype=np.float64)
    if np.isnan(x).all():
        flags["missing"] = True
        return False, flags
    if float(np.nanstd(x)) < 1e-4:
        flags["flatline"] = True
        return False, flags
    xmax, xmin = np.nanmax(x), np.nanmin(x)
    pinned = np.mean((np.abs(x - xmax) < 1e-6) | (np.abs(x - xmin) < 1e-6))
    flags["saturation"] = bool(pinned > sat_frac)
    p_sig = _band_power(x, fs, 0.5, 12.0)
    p_noise = _band_power(x, fs, 30.0, min(60.0, fs / 2.0))
    flags["snr_low"] = bool(p_sig / (p_noise + 1e-12) < snr_thresh)
    return not (flags["saturation"] or flags["snr_low"]), flags


def engineered_features(row):
    """Per-window engineered features from the 30 s waveforms (cell 18 math)."""
    seg = cut_segment(row["patient_id"], row["t_start_h"], row)
    has_ppg = bool(row["has_ppg"])
    ecg_ok, _ = quality_gate(seg["ecg"], MONITOR_FS)
    abp_ok, _ = quality_gate(seg["abp"], MONITOR_FS)
    ppg_ok = quality_gate(seg["ppg"], MONITOR_FS)[0] if has_ppg else False
    f = {"qf_pass": float(ecg_ok and abp_ok and (ppg_ok or not has_ppg))}

    ecg = seg["ecg"] - np.nanmedian(seg["ecg"])
    peaks, _ = find_peaks(ecg, distance=int(0.3 * MONITOR_FS),
                          height=max(0.6 * np.nanstd(ecg), 0.3))
    rr = np.diff(peaks) / MONITOR_FS
    if len(rr) >= 4:
        f["rr_mean_s"] = float(rr.mean())
        f["rr_sdnn_s"] = float(rr.std())
        f["rr_rmssd_s"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        f["ectopy_burden"] = float(np.mean(rr < 0.7 * np.median(rr)))
    else:
        f["rr_mean_s"] = f["rr_sdnn_s"] = f["rr_rmssd_s"] = np.nan
        f["ectopy_burden"] = np.nan

    abp = seg["abp"]
    f["abp_mean"] = float(np.nanmean(abp))
    f["abp_std"] = float(np.nanstd(abp))

    f["ptt_est_ms"] = np.nan
    if has_ppg and len(peaks) > 2:
        # per-beat PTT: for each R-peak, the PPG foot = steepest upstroke in
        # the physiological window after the peak; median over beats.
        ppg = seg["ppg"] - np.nanmedian(seg["ppg"])
        delays = []
        for p in peaks[:-1]:                          # last peak may be truncated
            lo, hi = p + int(0.03 * MONITOR_FS), p + int(0.55 * MONITOR_FS)
            if hi >= len(ppg):
                continue
            d = np.diff(ppg[lo:hi])
            if len(d) < 2:
                continue
            delays.append((int(np.argmax(d)) + 1) / MONITOR_FS * 1000.0)
        if delays:
            f["ptt_est_ms"] = float(np.median(delays))
    return f


def compute_engineered(manifest, tag, chunk=500):
    """Engineered features for every manifest window, cached to parquet
    (chunked prints so long runs stay visible)."""
    cache_f = EMBED_DIR / f"{tag}_engineered.parquet"
    if cache_f.exists():
        print(f"[{tag}] engineered features already cached ({cache_f.name})")
        return pd.read_parquet(cache_f)
    rows = []
    for i in range(0, len(manifest), chunk):
        sub = manifest.iloc[i:i + chunk]
        for _, r in sub.iterrows():
            rows.append({"patient_id": r["patient_id"], "t_start_h": r["t_start_h"],
                         **engineered_features(r)})
        print(f"[{tag}] engineered {min(i + chunk, len(manifest)):,}/{len(manifest):,}")
    out = pd.DataFrame(rows)
    out.to_parquet(cache_f, index=False)
    return out


def join_trends(vitals, eng, meta):
    """30-min HR/SBP trends joined onto the engineered table (cell 18).
    Idempotent: drops trend columns left over from a previous run."""
    vt = vitals.sort_values(["patient_id", "t_start_h"])
    lag_w = int(30 / WINDOW_MIN)                      # 30 min = 6 windows
    vt["hr_trend"] = vt.groupby("patient_id")["hr"].transform(lambda s: s.diff(lag_w))
    vt["sbp_trend"] = vt.groupby("patient_id")["sbp"].transform(lambda s: s.diff(lag_w))
    trend_map = vt.set_index(["patient_id", "t_start_h"])[["hr_trend", "sbp_trend"]]
    eng = eng.drop(columns=[c for c in ("hr_trend", "sbp_trend")
                            if c in eng.columns])
    return (eng.set_index(["patient_id", "t_start_h"]).join(trend_map, how="left")
              .reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild cohort parquet/csv")
    ap.add_argument("--skip-engineered", action="store_true")
    args = ap.parse_args()

    for d in (DATA_DIR, CACHE_DIR, EMBED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    vitals, events = build_synthetic_cohort(force=args.force)
    n_cases = int(vitals.groupby("patient_id")["is_case"].first().sum())
    print(f"cohort: {vitals.patient_id.nunique()} patients, {n_cases} cases | "
          f"{len(events)} events | seed {RANDOM_SEED}")

    print("\nper-horizon labels over the full cohort (floor + post-arrest excluded):")
    label_stats = label_summary(vitals)

    print("\npatient-level splits:")
    splits = make_splits(vitals)
    print("  ok: test/dev/val/train partitions (asserted, no overlap)")

    dev_rows, test_rows = build_manifests(vitals, events, splits)

    if not args.skip_engineered:
        eng_dev = compute_engineered(dev_rows, "dev")
        eng_test = compute_engineered(test_rows, "test")
        eng_dev = join_trends(vitals, eng_dev, dev_rows)
        eng_test = join_trends(vitals, eng_test, test_rows)
        eng_dev.to_parquet(EMBED_DIR / "dev_engineered.parquet", index=False)
        eng_test.to_parquet(EMBED_DIR / "test_engineered.parquet", index=False)
        print(f"engineered features: dev {eng_dev.shape}, test {eng_test.shape}")
        m = (eng_dev["qf_pass"] == 1) & (dev_rows["has_ppg"] == 1)
        print(f"  PTT estimate vs ground truth (clean dev windows): corr = "
              f"{np.corrcoef(eng_dev.loc[m, 'ptt_est_ms'],
                             dev_rows.loc[m, 'pt_ms'])[0, 1]:.3f}")

    print("\nPASS: cohort, splits, manifests, engineered features built.")


if __name__ == "__main__":
    main()
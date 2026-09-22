"""Synthetic ECG/PPG/ABP waveform synthesis.

Shared by the notebook (Section 1) and the embedding-extraction scripts
(Section 6) so both sides generate identical signals. Beat-constructed and
loosely coupled as in vivo: each ECG R-peak is followed by a PPG pulse and an
ABP pulse delayed by the pulse transit time (PTT).

Research prototype only - nothing here is validated for clinical use.
"""

import numpy as np

ECG_FM_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

# Approximate lead-II projection gains used by lead2_to_12leads (placeholder:
# real multi-lead synthesis would need a heart-vector model).
_LEAD2_GAINS = np.array([0.5, 1.0, 0.5, -1.5, -1.0, 1.5,
                         0.3, 0.6, 0.9, 1.2, 1.0, 0.8])


def _ecg_beat(t):
    """One cardiac cycle of a synthetic single-lead ECG (P, QRS, T as Gaussian bumps).

    t: time relative to the R-peak, in seconds.
    """
    def bump(center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    return (bump(-0.25, 0.020, 0.15)    # P wave
            + bump(-0.05, 0.006, -0.10)    # Q
            + bump(0.00, 0.004, 1.00)    # R
            + bump(0.05, 0.006, -0.25)    # S
            + bump(0.28, 0.030, 0.30))    # T wave


def _ppg_pulse(t, beat_period):
    """One PPG pulse: systolic upstroke, exponential decay, dicrotic notch.

    t: time relative to the pulse foot (R-peak + PTT), in seconds.
    """
    tau = beat_period / 4.0
    rise = 0.85 * np.exp(-0.5 * ((t - 0.10) / 0.06) ** 2)
    decay = 0.50 * np.exp(-t / tau)
    notch = 0.15 * np.exp(-0.5 * ((t - 0.30) / 0.025) ** 2)
    return rise + decay + notch


def _abp_pulse(t, sbp, dbp, beat_period):
    """One ABP pulse: fast rise to SBP, exponential decay toward DBP.

    t: time relative to the pulse onset, in seconds.
    """
    tau = beat_period / 5.0
    decay = dbp + (sbp - dbp) * np.exp(-t / tau)
    overshoot = 0.08 * (sbp - dbp) * np.exp(-0.5 * ((t - 0.03) / 0.02) ** 2)
    return decay + overshoot


def generate_waveform_segment(hr, sbp, dbp, amp_ecg, amp_ppg, pt_ms,
                              fs=125.0, dur_s=30.0,
                              rr_sd=0.04, noise=0.015, mains=0.008, seed=0):
    """Generate one synchronized ECG/PPG/ABP segment from window vitals.

    Deterministic per `seed` (use stable (patient, window) keys). Returns a
    dict of float32 arrays at `fs`; missing streams are NaN-filled.
    """
    rs = np.random.default_rng(seed)
    n = int(round(fs * dur_s))
    t = np.arange(n) / fs
    out = {k: np.full(n, np.nan) for k in ("ecg", "ppg", "abp")}

    has_ppg = amp_ppg is not None and not np.isnan(amp_ppg)
    beat_period = 60.0 / hr

    # --- ECG: beat train with sinus-arrhythmia jitter -------------------------
    ecg = np.zeros(n)
    r_time = -beat_period          # start one beat early so the segment opens mid-cycle
    beat_times = []
    while r_time < dur_s + beat_period:
        rr = beat_period * (1 + rs.normal(0, rr_sd))
        r_time += rr
        beat_times.append(r_time)
        lo, hi = r_time - 0.45, r_time + 0.55
        mask = (t >= lo) & (t < hi)
        if mask.any():
            ecg[mask] += amp_ecg * _ecg_beat(t[mask] - r_time)
    ecg += 0.06 * np.sin(2 * np.pi * 0.25 * t + rs.uniform(0, 2 * np.pi))   # baseline wander
    if mains > 0:
        ecg += mains * np.sin(2 * np.pi * 60.0 * t)                        # mains interference
    ecg += noise * rs.standard_normal(n)
    out["ecg"] = ecg

    # --- PPG / ABP: one pulse per beat, delayed by PTT -------------------------
    ppg = np.zeros(n)
    abp = np.zeros(n)
    for r in beat_times:
        onset = r + pt_ms / 1000.0
        mask = (t >= onset) & (t < onset + 1.5 * beat_period)
        if not mask.any():
            continue
        tl = t[mask] - onset
        if has_ppg:
            ppg[mask] += amp_ppg * _ppg_pulse(tl, beat_period)
        abp[mask] += _abp_pulse(tl, sbp, dbp, beat_period)
    if has_ppg:
        ppg += 0.03 * np.sin(2 * np.pi * 0.25 * t + rs.uniform(0, 2 * np.pi))
        ppg += noise * rs.standard_normal(n)
        out["ppg"] = ppg
    abp += 0.02 * np.sin(2 * np.pi * 0.25 * t + rs.uniform(0, 2 * np.pi))
    abp += noise * rs.standard_normal(n)
    out["abp"] = abp

    return {k: v.astype(np.float32) for k, v in out.items()}


def lead2_to_12leads(lead2):
    """Approximate 12-lead ECG from a lead-II signal (gain projections + noise).

    Placeholder for the pipeline: ECG-FM expects 12 leads at 500 Hz, bedside
    monitors typically provide 1-3. Documented limitation - see plan section 8.
    """
    lead2 = np.asarray(lead2, dtype=np.float64)
    rng = np.random.default_rng(7)  # fixed so the transform is deterministic
    leads = [_LEAD2_GAINS[i] * lead2 + 0.01 * rng.standard_normal(len(lead2))
             for i in range(12)]
    return np.stack(leads).astype(np.float32)


def zscore_segments(x):
    """Z-score each segment independently (ECG-FM / PaPaGei preprocessing)."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=tuple(range(1, x.ndim)), keepdims=True)
    sd = x.std(axis=tuple(range(1, x.ndim)), keepdims=True)
    return ((x - mu) / (sd + 1e-9)).astype(np.float32)


def inject_artifact(x, qf, rs):
    """Degrade a segment according to its ground-truth quality flag."""
    x = x.copy()
    if qf == "flat":
        x[:] = np.nanmean(x) + 1e-5 * rs.standard_normal(len(x))
    elif qf == "sat":
        lo, hi = np.nanpercentile(x, 3), np.nanpercentile(x, 97)
        x = np.clip(x, lo, hi)
    elif qf == "noise":
        x = x + 0.4 * rs.standard_normal(len(x))
    return x


def cut_segment(patient_id, t_start_h, row, inject=True, fs=125.0, dur_s=30.0):
    """Cut one 30 s ECG/PPG/ABP segment for a 5-min window.

    Deterministic per (patient, window): the seed is a stable integer, so
    segments can be regenerated (and their embeddings cached) without drift.
    `row` is a vitals-table row: hr, sbp, dbp, amp_ppg, pt_ms, rrsd, has_ppg,
    qf_ecg, qf_ppg, qf_abp.
    """
    seed = int(patient_id) * 1_000_000 + int(round(t_start_h * 60.0))
    rs = np.random.default_rng(seed)
    seg = generate_waveform_segment(
        hr=row["hr"], sbp=row["sbp"], dbp=row["dbp"],
        amp_ecg=1.0,
        amp_ppg=(None if not bool(row["has_ppg"]) else row["amp_ppg"]),
        pt_ms=row["pt_ms"], rr_sd=row["rrsd"], seed=seed, fs=fs, dur_s=dur_s)
    if inject:
        seg["ecg"] = inject_artifact(seg["ecg"], row["qf_ecg"], rs)
        seg["abp"] = inject_artifact(seg["abp"], row["qf_abp"], rs)
        if bool(row["has_ppg"]):
            seg["ppg"] = inject_artifact(seg["ppg"], row["qf_ppg"], rs)
    return seg
"""Shared Sudden Cardiac Death Holter Database (SDDB) utilities.

Open-access PhysioNet database (no credentials): 23 Holter recordings of
patients who arrested during recording; 20 have documented VF-onset times.
2 leads, 250 Hz, 4-25 h per record.

Used by analyze_sddb.py (short-horizon stress test) and slow_risk.py
(section 9 slow-risk layer). Research prototype only.
"""
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import find_peaks, resample_poly

PROJECT = Path(__file__).resolve().parent.parent
SDDB_DIR = PROJECT / "data/sddb"
FS_TARGET = 125.0

# VF onset (elapsed since recording start) per record; 40/42/49 have no VF
VF_ONSET = {
    "30": "07:54:33", "31": "13:42:24", "32": "16:45:18", "33": "04:46:19",
    "34": "06:35:44", "35": "24:34:56", "36": "18:59:01", "37": "01:31:13",
    "38": "08:01:54", "39": "04:37:51", "41": "02:59:24", "43": "15:37:11",
    "44": "19:38:45", "45": "18:09:17", "46": "03:41:47", "47": "06:13:01",
    "48": "02:29:40", "50": "11:45:43", "51": "22:58:23", "52": "02:32:40",
}
FLOOR_S = 15 * 60          # "already responding" floor before arrest
SPAN_S = 6 * 3600          # 6 h span of windows
NEG_OFFSET_S = 10 * 60     # negatives start 10 min into the record


def onset_seconds(rec):
    h, m, s = map(int, VF_ONSET[rec].split(":"))
    return h * 3600 + m * 60 + s


def record_duration(rec):
    """Recording duration in seconds, from the .hea header."""
    head = (SDDB_DIR / f"{rec}.hea").read_text().splitlines()[0].split()
    return int(head[3]) / int(head[2])          # nsamples / fs


def load_span(rec, start_s, end_s):
    """Read one span of a record at 125 Hz via wfdb (250 -> 125 Hz)."""
    rec_path = SDDB_DIR / rec
    sig, meta = wfdb.rdsamp(str(rec_path), sampfrom=int(start_s * 250),
                            sampto=int(end_s * 250))
    sig = resample_poly(sig, 1, 2, axis=0).astype(np.float32)
    return sig, meta


def ecg_features(ecg30):
    """HR, RR variability and ectopy burden from a 30 s ECG at 125 Hz."""
    x = ecg30 - np.nanmedian(ecg30)
    peaks, _ = find_peaks(x, distance=int(0.3 * FS_TARGET),
                          height=max(0.6 * np.nanstd(x), 0.3))
    rr = np.diff(peaks) / FS_TARGET
    if len(rr) < 4:
        return dict(rr_mean_s=np.nan, rr_sdnn_s=np.nan, rr_rmssd_s=np.nan,
                    ectopy_burden=np.nan, hr=np.nan, rrsd=np.nan)
    hr = 60.0 / rr.mean()
    return dict(rr_mean_s=float(rr.mean()), rr_sdnn_s=float(rr.std()),
                rr_rmssd_s=float(np.sqrt(np.mean(np.diff(rr) ** 2))),
                ectopy_burden=float(np.mean(rr < 0.7 * np.median(rr))),
                hr=float(hr), rrsd=float(rr.std() / rr.mean()))


def require_sddb():
    """Plan rule 6: fail loudly with a clear message when the data is missing."""
    if not (SDDB_DIR / "30.hea").exists():
        raise FileNotFoundError(
            f"SDDB not found at {SDDB_DIR}. Download it from PhysioNet "
            "(sudden-cardiac-death-holter-database, open access) with: "
            "wget -r -N -c -np https://physionet.org/files/sddb/1.0.0/ "
            f"-P {SDDB_DIR}")
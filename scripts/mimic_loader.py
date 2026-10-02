"""MIMIC-III Ext-CA loader - credentialed ICU evaluation (pending access).

Prepares the loading code for the definitive evaluation (plan section 6 /
manuscript "next step"): the MIMIC-III Ext-CA annotation set (36 arrest
episodes in 31 unique patients) plus the waveform matched subset of
mimic3wdb. Per the PhysioNet description, `dataset.csv` has columns:

    row_id, subject_id, hadm_id, file,
    cardiac_arrest_start, cardiac_arrest_end   (YYYY-MM-DD HH:MM:SS, shifted)

`file` names the wfdb record in the matched subset (e.g.
"p000020-2117-05-26-17-48"). Records must carry a PPG signal (PLETH*) plus
either ECG (II) or continuous ABP (ABP/ART). If a record ends before the
arrest end timestamp, the record end is the arrest end.

Plan rules enforced in code:
  rule 2: split by patient (split_patients asserts disjointness)
  rule 3: fixed seeds, class/event counts printed
  rule 4: windows overlapping [arrest_start, arrest_end] (resuscitation /
          post-arrest) and the 15 min response floor are excluded + asserted
  rule 6: fail loudly with clear messages when data is missing

Usage:
  python scripts/mimic_loader.py --check      # inventory; exit 1 if missing
  python scripts/mimic_loader.py --selftest   # window/label logic on fake events
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
EXT_CA_CANDIDATES = [
    DATA_DIR / "mimic-iii-ext-ca" / "dataset.csv",
    DATA_DIR / "mimiciii_ext_ca" / "dataset.csv",
    DATA_DIR / "mimic-iii-ext-ca-1.0" / "dataset.csv",
]
WAVE_ROOT_CANDIDATES = [
    DATA_DIR / "mimic3wdb" / "matched",
    DATA_DIR / "mimic3wdb-matched",
]
REQUIRED_EVENT_COLS = ["row_id", "subject_id", "hadm_id", "file",
                       "cardiac_arrest_start", "cardiac_arrest_end"]
WINDOW_S = 5 * 60
FLOOR_S = 15 * 60          # response-phase floor before arrest
PRE_SPAN_S = 6 * 3600
NEG_OFFSET_S = 10 * 60     # negatives start 10 min into the record
SEED = 7


def find_ext_ca():
    for p in EXT_CA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "MIMIC-III Ext-CA dataset.csv not found. Expected one of:\n  " +
        "\n  ".join(str(p) for p in EXT_CA_CANDIDATES) +
        "\nPhysioNet access is credentialed (CITI training + signed DUA). "
        "Download from https://physionet.org/content/mimic-iii-ext-ca/ and "
        "place dataset.csv under data/. (Rule 6: failing loudly.)")


def find_wave_root():
    for p in WAVE_ROOT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "MIMIC-III Waveform Database Matched Subset not found. Expected one of:\n  " +
        "\n  ".join(str(p) for p in WAVE_ROOT_CANDIDATES) +
        "\nDownload the matched subset from "
        "https://physionet.org/content/mimic3wdb-matched/ and place it under "
        "data/. (Rule 6: failing loudly.)")


def load_events(path=None):
    """Read the Ext-CA event table; validate columns; parse timestamps."""
    p = Path(path) if path else find_ext_ca()
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_EVENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} is missing required columns {missing}; "
                         f"found {list(df.columns)}")
    df["cardiac_arrest_start"] = pd.to_datetime(df["cardiac_arrest_start"])
    df["cardiac_arrest_end"] = pd.to_datetime(df["cardiac_arrest_end"])
    bad = df["cardiac_arrest_end"] < df["cardiac_arrest_start"]
    if bad.any():
        raise ValueError(f"{int(bad.sum())} events have end < start")
    df = df.sort_values(["subject_id", "cardiac_arrest_start"]).reset_index(drop=True)
    print(f"events: {len(df)} in {df['subject_id'].nunique()} patients | "
          f"seed {SEED} (rule 3)")
    print(f"  event count per patient: "
          f"{df.groupby('subject_id').size().describe().to_dict()}")
    return df


def record_layout(wave_root, file):
    """Parse the layout header of a matched-subset record.

    Returns dict(fs, n_samples, sig_names). The layout header lives next to
    the numbered segment files, e.g. <file>_layout.hea."""
    rec_dir = wave_root / file[:5] / file            # p00xxx / record base
    layout = rec_dir.parent / (file + "_layout.hea")
    if not layout.exists():
        # some records use a short layout header name; fall back to searching
        hits = sorted(rec_dir.parent.glob(file + "*_layout.hea"))
        if not hits:
            raise FileNotFoundError(
                f"no layout header for record {file} under {wave_root}")
        layout = hits[0]
    head = layout.read_text().splitlines()[0].split()
    sig_names = [ln.split()[0] for ln in layout.read_text().splitlines()[1:]]
    return dict(fs=int(head[2]), n_samples=int(head[3]), sig_names=sig_names)


def record_span_s(wave_root, file):
    """[start_s, end_s] of the record, in seconds since the record start."""
    lay = record_layout(wave_root, file)
    return 0.0, lay["n_samples"] / lay["fs"]


def required_signal_check(wave_root, file):
    """The Ext-CA inclusion criteria: PPG present, plus ECG-II or ABP/ART.
    Returns (ok, present, missing)."""
    sigs = record_layout(wave_root, file)["sig_names"]
    has_ppg = any(s.upper().startswith("PLETH") for s in sigs)
    has_ecg = any(s.upper() in ("II", "II+", "ECG") for s in sigs)
    has_abp = any(s.upper() in ("ABP", "ART") for s in sigs)
    ok = has_ppg and (has_ecg or has_abp)
    missing = []
    if not has_ppg:
        missing.append("PPG")
    if not (has_ecg or has_abp):
        missing.append("ECG-II-or-ABP")
    return ok, sorted(set(sigs)), missing


def record_start_dt(file):
    """Record start datetime from the wfdb record name, e.g.
    'p000020-2117-05-26-17-48' -> 2117-05-26 17:48:00. The Ext-CA timestamps
    live in the same (shifted) timeline as these names."""
    try:
        return pd.to_datetime(file[8:], format="%Y-%m-%d-%H-%M")
    except ValueError:
        raise ValueError(f"cannot parse record start from filename {file}; "
                         f"expected pNNNNNN-YYYY-MM-DD-HH-MM")


def build_manifest(events, wave_root):
    """5 min windows per event: pre-arrest positives (last 6 h) + far
    negatives (early 6 h), with exclusion reasons (plan rule 4)."""
    rows = []
    for _, ev in events.iterrows():
        file, subj = ev["file"], ev["subject_id"]
        t0 = record_start_dt(file)
        _, rec_e = record_span_s(wave_root, file)
        a0 = (ev["cardiac_arrest_start"] - t0).total_seconds()
        # If the record ends before the arrest end timestamp, the record end
        # is the arrest end (documented in the Ext-CA data description).
        a1 = min((ev["cardiac_arrest_end"] - t0).total_seconds(), rec_e)
        assert a1 >= a0, f"event {ev['row_id']}: arrest window invalid after clamping"
        spans = []
        lo = max(0.0, a0 - PRE_SPAN_S)
        if a0 - FLOOR_S - lo >= WINDOW_S:
            spans.append(("pre", lo, a0))
        far_end = a0 - PRE_SPAN_S
        if far_end - NEG_OFFSET_S >= WINDOW_S:
            spans.append(("far", NEG_OFFSET_S, min(NEG_OFFSET_S + PRE_SPAN_S, far_end)))
        for kind, s0, s1 in spans:
            for t in np.arange(s0, s1, WINDOW_S):
                t1 = t + WINDOW_S
                if t1 > a0 - FLOOR_S:
                    reason = "response_floor"
                elif t1 >= a1:                    # windows after arrest (incl.
                    reason = "post_arrest"        # resuscitation [a0, a1])
                else:
                    reason = "ok"
                row = dict(subject_id=subj, file=file, kind=kind,
                           t_start_s=t, t_end_s=t1,
                           arrest_start_s=a0, arrest_end_s=a1,
                           excluded=reason)
                if reason == "ok":
                    row["tte_h"] = (a0 - t1) / 3600.0
                rows.append(row)
    manifest = pd.DataFrame(rows)
    # plan rule 4 assert: no evaluable window overlaps the arrest window
    ev_ok = manifest[manifest["excluded"] == "ok"]
    overlap = (ev_ok["t_start_s"] < ev_ok["arrest_end_s"]) & \
              (ev_ok["t_end_s"] > ev_ok["arrest_start_s"])
    assert not overlap.any(), "evaluable windows overlap the arrest window"
    assert (ev_ok["t_end_s"] <= ev_ok["arrest_start_s"] - FLOOR_S).all(), \
        "evaluable windows violate the response floor"
    print(f"manifest: {len(manifest)} windows "
          f"({manifest['excluded'].value_counts().to_dict()})")
    for H in (1, 6, 24):
        y = ((ev_ok["tte_h"] > 0.25) & (ev_ok["tte_h"] <= H)).astype(int)
        print(f"  {H:>2}h positives: {int(y.sum())} (rule 3)")
    return manifest


def split_patients(events, test_frac=0.2, seed=SEED):
    """Patient-level split of the event table (plan rule 2)."""
    rs = np.random.default_rng(seed)
    subjects = events["subject_id"].unique()
    rs.shuffle(subjects)
    k = max(1, int(round(test_frac * len(subjects))))
    test_subjects = set(subjects[:k])
    train_subjects = set(subjects[k:])
    assert not (test_subjects & train_subjects), "patient overlap in split"
    return train_subjects, test_subjects


def load_span(wave_root, file, start_s, end_s, fs_target=125.0):
    """Read one time span of a matched-subset record at fs_target Hz."""
    import wfdb
    from scipy.signal import resample_poly
    lay = record_layout(wave_root, file)
    rec_dir = wave_root / file[:5] / file
    sig, meta = wfdb.rdsamp(str(rec_dir),
                            sampfrom=int(start_s * lay["fs"]),
                            sampto=int(end_s * lay["fs"]))
    if lay["fs"] != fs_target:
        sig = resample_poly(sig, fs_target, lay["fs"], axis=0)
    return sig.astype(np.float32), meta, lay


def check():
    """Inventory + fail-loudly check (rule 6)."""
    events = load_events()
    wave_root = find_wave_root()
    missing_files = []
    for _, ev in events.iterrows():
        f = ev["file"]
        try:
            ok, sigs, miss = required_signal_check(wave_root, f)
        except FileNotFoundError as e:
            missing_files.append((f, str(e)))
            continue
        mark = "OK " if ok else "MISSING: " + ",".join(miss)
        print(f"  {f:<32} {mark}")
        if not ok:
            missing_files.append((f, ",".join(miss)))
    if missing_files:
        raise SystemExit(f"check failed: {len(missing_files)} records missing "
                         f"or lacking PPG / ECG-II / ABP signals")
    print("check PASS: all event records present with required signals")


def selftest():
    """Window/label/exclusion logic on a fake event table (rule 7: no real
    data needed for the label machinery)."""
    fake = pd.DataFrame({
        "row_id": [1, 2, 3],
        "subject_id": [101, 102, 103],
        "hadm_id": [1, 2, 3],
        "file": ["f101", "f102", "f103"],
        "cardiac_arrest_start": pd.to_datetime(["2100-01-01 12:00:00",
                                                "2100-01-01 08:00:00",
                                                "2100-01-01 20:00:00"]),
        "cardiac_arrest_end": pd.to_datetime(["2100-01-01 12:30:00",
                                              "2100-01-01 08:20:00",
                                              "2100-01-01 20:15:00"]),
    })
    # fake records: 30 h each, all starting 2100-01-01 00:00:00
    mod = sys.modules[__name__]
    orig = (mod.record_layout, mod.record_span_s, mod.record_start_dt)
    mod.record_layout = lambda wave_root, file: dict(
        fs=125.0, n_samples=int(30 * 3600 * 125), sig_names=["PLETH", "II"])
    mod.record_span_s = lambda wave_root, file: (0.0, 30 * 3600.0)
    mod.record_start_dt = lambda file: pd.Timestamp("2100-01-01 00:00:00")

    try:
        manifest = build_manifest(fake, None)
        ev_ok = manifest[manifest["excluded"] == "ok"]
        assert len(ev_ok) > 0
        # no evaluable window in the last 15 min before arrest
        assert (ev_ok["t_end_s"] <= ev_ok["arrest_start_s"] - FLOOR_S).all()
        # no evaluable window inside the arrest window
        assert not ((ev_ok["t_start_s"] < ev_ok["arrest_end_s"]) &
                    (ev_ok["t_end_s"] > ev_ok["arrest_start_s"])).any()
        # exclusion reasons only from the documented set
        assert set(manifest["excluded"].unique()) <= {"ok", "response_floor",
                                                      "post_arrest"}
        # patient split disjointness (rule 2)
        tr, te = split_patients(fake)
        assert not (tr & te)
        # every patient contributes at least one pre-arrest positive (rule 3)
        for s in fake["subject_id"]:
            y6 = ((ev_ok["tte_h"] > 0.25) & (ev_ok["tte_h"] <= 6) &
                  (ev_ok["subject_id"] == s)).sum()
            assert y6 > 0, f"patient {s} has no 6 h positives"
        # missing-data behavior (rule 6): unknown path fails loudly
        try:
            load_events(path=PROJECT / "data/does_not_exist.csv")
            raise AssertionError("load_events did not fail on missing file")
        except FileNotFoundError:
            pass
        print("selftest PASS: manifest/label/exclusion logic correct on fake "
              "events (rule 7)")
    finally:
        (mod.record_layout, mod.record_span_s, mod.record_start_dt) = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.check:
        check()
    else:
        check()


if __name__ == "__main__":
    main()
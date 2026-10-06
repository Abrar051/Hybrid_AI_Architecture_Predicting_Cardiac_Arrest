"""Rule-based evidence branch + hybrid replay (reviewer point 1, Phase B).

Implements the paper's Section 3.9/3.10 as executable code:
  - extends the synthetic cohort with the manual-observation channels the rule
    branch needs (respiratory rate, temperature, ACVPU, PPG-derived RR),
    generated ADDITIVELY from the cached Phase A vitals so every Phase A
    artifact (cohort, splits, embeddings, models) stays bit-identical;
  - computes NEWS2 and the six-component deterioration score S_rule per window;
  - runs the four deterministic trigger rules with per-rule 180 s cooldowns;
  - replays the learned branch (cache/models/ews_v2, 60 min refractory) and the
    hybrid decision agent (Algorithm 8: one event per cycle combining every
    eligible trigger) chronologically over the held-out test patients;
  - compares learned-only vs rule-only vs hybrid: alerts per patient-day,
    sensitivity, PPV, lead time (within-horizon true-alert definition);
  - runs the generator honesty check on the NEW channels (case/control must
    differ only inside the drift window).

Outputs: cache/rule_channels.parquet, outputs/rule_branch_metrics.json

Usage: python scripts/rule_branch.py [--skip-honesty]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, ttest_ind

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from config import (CACHE_VITALS, CACHE_DIR, EWS_MODEL_DIR, OUTPUT_DIR,  # noqa: E402
                    DRIFT_H, HORIZONS, ALERT, WINDOW_MIN)
from eval_utils import (load_manifest_features, make_sequences,  # noqa: E402
                        iter_test_patients, alert_from_risks)
from ews_risk import RiskPipeline  # noqa: E402

RULE_SEED = 1007                 # separate stream; vitals stay untouched
CHANNEL_CACHE = CACHE_DIR / "rule_channels.parquet"
OUT_JSON = OUTPUT_DIR / "rule_branch_metrics.json"
REFRACTORY_WIN = int(ALERT["refractory_min"] / 5)      # 60 min = 12 windows
COOLDOWN_S = 180


# ---------- extended channels (additive, drift-coupled) --------------------

def build_rule_channels(force=False):
    """RR / temp / ACVPU / PPG-derived RR for every cohort window, keyed to the
    cached Phase A vitals. Case/control differences enter ONLY through the same
    drift ramp p used by the original generator, so pre-drift windows remain
    statistically indistinguishable between cases and controls."""
    if not force and CHANNEL_CACHE.exists():
        out = pd.read_parquet(CHANNEL_CACHE)
        print(f"rule channels cached: {out.shape} -> {CHANNEL_CACHE.name}")
        return out
    vitals = pd.read_parquet(CACHE_VITALS)
    rs = np.random.default_rng(RULE_SEED)

    rows = []
    for pid, g in vitals.groupby("patient_id"):
        is_case = bool(g["is_case"].iloc[0])
        ah = g["arrest_h"].iloc[0]
        rr0 = rs.normal(15.0, 2.0)
        temp0 = rs.normal(36.7, 0.25)
        acvpu_base_p = 0.003        # identical for cases and controls (honesty)
        t = g["t_start_h"].to_numpy()
        for i, (_, r) in enumerate(g.iterrows()):
            p = 0.0
            if is_case and (ah - r["t_start_h"]) <= DRIFT_H and r["t_start_h"] < ah:
                p = (1.0 - (ah - r["t_start_h"]) / DRIFT_H) ** 2
            rr = rr0 + 14.0 * p + rs.normal(0, 1.0)
            temp = temp0 + 1.5 * p + rs.normal(0, 0.08)
            # consciousness: alert at baseline, voice/pain responses become more
            # likely as the arrest approaches (cases only, via the drift ramp)
            p_vp = min(0.9, acvpu_base_p + 0.35 * p * p)
            acvpu = "A" if rs.random() >= p_vp else ("P" if rs.random() < 0.5 else "V")
            # PPG-derived RR estimate: tracks measured RR with noise + a lag
            # error that grows with drift -> disagreement component activates
            ppg_rr = rr + 2.0 * p + rs.normal(0, 1.2)
            rows.append(dict(patient_id=pid, t_start_h=r["t_start_h"],
                             rr=round(float(rr), 2), temp=round(float(temp), 2),
                             acvpu=acvpu, ppg_rr=round(float(ppg_rr), 2)))
    out = pd.DataFrame(rows)
    out.to_parquet(CHANNEL_CACHE, index=False)
    print(f"rule channels built: {out.shape} -> {CHANNEL_CACHE.name}")
    return out


def honesty_check(channels, vitals):
    """Case/control equivalence OUTSIDE the drift window for the new channels
    (same protocol as generator_honesty.py, Bonferroni across 4 x 2 tests)."""
    m = vitals.merge(channels, on=["patient_id", "t_start_h"])
    case = m[m["is_case"]]
    pre = case[(case["arrest_h"] - case["t_start_h"]) > DRIFT_H]
    ctrl = m[~m["is_case"]]
    out = {"drift_h": DRIFT_H}
    tests = {}
    for ch in ("rr", "temp"):
        a = pre.groupby("patient_id")[ch].mean()
        b = ctrl.groupby("patient_id")[ch].mean()
        tests[f"{ch}_ks"] = float(ks_2samp(a, b).pvalue)
        tests[f"{ch}_welch"] = float(ttest_ind(a, b, equal_var=False).pvalue)
    for ch in ("acvpu",):
        # pooled proportion z-test: per-patient KS on a rare binary is
        # hypersensitive to heterogeneous window counts (ties at 0)
        na = float((pre[ch] != "A").sum()); nc = float((ctrl[ch] != "A").sum())
        pa, pc = na / len(pre), nc / len(ctrl)
        p_pool = (na + nc) / (len(pre) + len(ctrl))
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / len(pre) + 1 / len(ctrl)))
        z = (pa - pc) / se
        from scipy.stats import norm
        tests[f"{ch}_z_pooled"] = float(2 * (1 - norm.cdf(abs(z))))
        a = pre.groupby("patient_id")[ch].apply(lambda s: (s != "A").mean())
        b = ctrl.groupby("patient_id")[ch].apply(lambda s: (s != "A").mean())
        tests[f"{ch}_welch"] = float(ttest_ind(a, b, equal_var=False).pvalue)
    out["tests"] = tests
    alpha = 0.05 / len(tests)
    bad = [k for k, v in tests.items() if v < alpha]
    out["verdict"] = ("clean: no detectable case/control difference outside the "
                      "drift window after Bonferroni" if not bad else
                      f"WARNING: channels differ pre-drift: {bad}")
    print(f"rule-channel honesty ({len(tests)} tests, alpha={alpha:.4f}): "
          f"{out['verdict']}")
    for k, v in tests.items():
        print(f"  {k:12s} p={v:.4f}")
    return out


# ---------- NEWS2 + six-component score ------------------------------------

def news2_score(rr, spo2, temp, acvpu, hr, sbp):
    s = 0
    s += 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3
    s += 0 if spo2 >= 96 else 1 if spo2 >= 94 else 2 if spo2 >= 92 else 3
    s += 3 if temp <= 35.0 else 1 if temp <= 36.0 else 0 if temp <= 38.0 else \
        1 if temp <= 39.0 else 2
    s += 0 if acvpu == "A" else 3
    s += 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else \
        2 if hr <= 130 else 3
    s += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else \
        0 if sbp <= 219 else 3
    return s


def components(df):
    """Six deterioration components + S_rule per window (paper Table
    tab:rulecomponents; trailing trends over 10 min = 2 windows)."""
    d = df.copy()
    for col in ("rrsd", "amp_ppg", "spo2", "news2", "qf_score"):
        d[f"{col}_prev"] = d.groupby("patient_id")[col].shift(2)
    pct = lambda cur, prev: (cur - prev) / prev.abs().replace(0, np.nan) * 100

    c_news2 = np.clip(200.0 * (d["news2"] - d["news2_prev"]) / 10.0, 0, 100)
    c_rmssd = np.where(pct(d["rrsd"], d["rrsd_prev"]) < -5,
                       np.clip(2.5 * pct(d["rrsd"], d["rrsd_prev"]).abs(), 0, 100), 0.0)
    c_pi = np.where(pct(d["amp_ppg"], d["amp_ppg_prev"]) < -3,
                    np.clip(5.0 * pct(d["amp_ppg"], d["amp_ppg_prev"]).abs(), 0, 100), 0.0)
    c_spo2 = np.where((d["spo2"] - d["spo2_prev"]) < 0,
                      np.clip(100.0 * (d["spo2"] - d["spo2_prev"]).abs() / 10.0, 0, 100), 0.0)
    delta = (d["ppg_rr"] - d["rr"]).abs()
    c_resp = np.where(delta > 4, np.clip(5.0 * delta, 0, 100), 0.0)
    c_qf = np.where(pct(d["qf_score"], d["qf_score_prev"]) < -10,
                    np.clip(1.5 * pct(d["qf_score"], d["qf_score_prev"]).abs(), 0, 100), 0.0)
    s_rule = np.minimum(100.0, np.round(
        0.25 * c_news2.fillna(0) + 0.25 * c_rmssd + 0.20 * c_pi +
        0.15 * c_spo2 + 0.10 * c_resp + 0.05 * c_qf))
    for name, c in (("c_news2", c_news2), ("c_rmssd", c_rmssd), ("c_pi", c_pi),
                    ("c_spo2", c_spo2), ("c_resp", c_resp), ("c_qf", c_qf)):
        d[name] = np.nan_to_num(c, nan=0.0)
    d["s_rule"] = s_rule
    return d


# ---------- replay ----------------------------------------------------------

def rule_replay(df):
    """Chronological replay of the four deterministic rules with per-rule
    180 s cooldowns + the NEWS2 tracker. Returns an alert mask over windows."""
    alerts = np.zeros(len(df), dtype=bool)
    per_patient = []
    for pid, g in df.groupby("patient_id"):
        g = g.reset_index(drop=True)
        a = np.zeros(len(g), dtype=bool)
        last_fire = {k: -np.inf for k in ("high", "sustained", "coupled", "news2")}
        news2_crossed, news2_ref = False, 0.0
        for i, r in g.iterrows():
            t_min = (g.loc[i, "t_start_h"] - g.loc[0, "t_start_h"]) * 60.0
            elig = {k: (t_min - last_fire[k]) >= COOLDOWN_S for k in last_fire}
            fired = []
            if elig["high"] and r["s_rule"] >= 75:
                fired.append("high")
            if elig["sustained"] and i >= 2 and (g.loc[i - 2:i + 1, "s_rule"] >= 50).all():
                fired.append("sustained")
            if elig["coupled"] and r["c_rmssd"] >= 70 and r["c_spo2"] >= 50:
                fired.append("coupled")
            if elig["news2"]:
                if (not news2_crossed and r["news2"] >= 7) or \
                   (news2_crossed and r["news2"] - news2_ref >= 2):
                    news2_crossed = True
                    news2_ref = float(r["news2"])
                    fired.append("news2")
            for k in fired:
                last_fire[k] = t_min
                a[i] = True
        per_patient.append(a)
    return np.concatenate(per_patient)


def learned_risks(meta):
    """Learned 6 h risks for the test manifest using the Phase A seed-42
    checkpoint (same features, scaling, and step=1 sequences as the Phase A
    evaluation; streams per patient to stay RAM-safe)."""
    meta_full, feats, masks, labels, tte, _ = load_manifest_features("test")
    pipe = RiskPipeline.load(EWS_MODEL_DIR)
    mu, sd = pipe.feat_mean, pipe.feat_std
    z = (feats - mu) / (sd + 1e-8)
    risks = np.full(len(meta_full), np.nan)
    for pid, is_case, arrest_h, idx, X, L, Y, T, M in iter_test_patients(
            meta_full, z, masks, labels, tte):
        with torch.no_grad():
            out = pipe.model(torch.from_numpy(X), torch.from_numpy(L))
        risks[idx] = torch.sigmoid(out["risk"][6]).numpy().ravel()
    return meta_full, risks


def true_alert_mask(meta, alert, horizon=6):
    out = np.zeros(len(meta), dtype=bool)
    for pid in np.unique(meta["patient_id"]):
        g = meta["patient_id"] == pid
        ah = meta.loc[g, "arrest_h"].iloc[0]
        if not np.isfinite(ah):
            continue
        t = meta.loc[g, "t_start_h"].to_numpy()
        a = alert[g.to_numpy()]
        out[g.to_numpy()] = a & (ah - t > 0) & (ah - t <= horizon)
    return out


def variant_stats(meta, alert):
    """Alerts per patient-day, sensitivity, PPV, lead time for one variant.
    Patient-day denominator = replayed span (first to last window), matching
    the Phase A evaluation."""
    out = {}
    g = meta.assign(alert=alert)
    per = g.groupby("patient_id").agg(
        n_alert=("alert", "sum"), t_first=("t_start_h", "min"),
        t_last=("t_start_h", "max"), is_case=("is_case", "first"),
        arrest_h=("arrest_h", "first"))
    span_h = (per["t_last"] - per["t_first"]) + (WINDOW_MIN / 60.0)
    per["alerts_per_day"] = per["n_alert"] / (span_h / 24.0)
    cases = per[per["is_case"]]
    ctrls = per[~per["is_case"]]
    true = true_alert_mask(meta, alert)
    out["alerts_per_day_case"] = [float(x) for x in cases["alerts_per_day"].round(2)]
    out["alerts_per_day_ctrl"] = [float(x) for x in ctrls["alerts_per_day"].round(2)]
    n_cases = int(cases.shape[0])
    # sensitivity = fraction of case patients with at least one within-horizon alert
    case_ids = set(meta.loc[meta["is_case"], "patient_id"])
    alerted_ids = set(meta.loc[true, "patient_id"])
    out["n_case_alerted"] = len(alerted_ids & case_ids)
    out["sensitivity"] = float(len(alerted_ids & case_ids) / n_cases) if n_cases else np.nan
    # true alert = alert with arrest inside the 6 h horizon (any patient)
    out["n_alerts"] = int(alert.sum())
    out["n_true_alerts"] = int(true.sum())
    out["ppv"] = float(true.sum() / alert.sum()) if alert.sum() else np.nan
    leads = []
    for pid in np.unique(meta.loc[true, "patient_id"]):
        g = (meta["patient_id"] == pid) & true
        leads.append(float(meta.loc[g, "arrest_h"].iloc[0] -
                           meta.loc[g, "t_start_h"].min()))
    if leads:
        out["first_true_lead_h"] = [float(np.min(leads)), float(np.median(leads)),
                                    float(np.max(leads))]
    return out


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-honesty", action="store_true")
    ap.add_argument("--force-channels", action="store_true")
    args = ap.parse_args()

    vitals = pd.read_parquet(CACHE_VITALS)
    channels = build_rule_channels(force=args.force_channels)
    honesty = None if args.skip_honesty else honesty_check(channels, vitals)
    if honesty and honesty["verdict"].startswith("WARNING"):
        raise RuntimeError("rule channels fail the honesty check — inspect "
                           "cache/rule_channels.parquet")

    # ---- rule branch over the held-out test patients ----------------------
    test_meta = pd.read_csv(PROJECT / "cache/embeddings/test_windows.csv")
    df = test_meta.merge(channels, on=["patient_id", "t_start_h"], how="left")
    assert df["rr"].notna().all(), "missing rule channels on test windows"
    df["qf_score"] = (df["qf_ecg"] == "ok").astype(int) + \
                     (df["qf_ppg"] == "ok").astype(int) + \
                     (df["qf_abp"] == "ok").astype(int)
    df["news2"] = df.apply(lambda r: news2_score(r["rr"], r["spo2"], r["temp"],
                                                r["acvpu"], r["hr"], r["sbp"]),
                           axis=1)
    df = components(df)

    rule_alert = rule_replay(df)

    # ---- learned branch (Phase A checkpoint) ------------------------------
    lmeta, risks = learned_risks(test_meta)
    df = df.merge(lmeta[["patient_id", "t_start_h"]].assign(risk6=risks),
                  on=["patient_id", "t_start_h"], how="left")
    learned_alert = np.zeros(len(df), dtype=bool)
    for pid in np.unique(df["patient_id"]):
        g = df["patient_id"] == pid
        idx = g.to_numpy()
        ah = df.loc[g, "arrest_h"].iloc[0]
        t = df.loc[g, "t_start_h"].to_numpy()
        r = df.loc[g, "risk6"].to_numpy()
        if np.isnan(r).all():
            continue
        a, _, _ = alert_from_risks(t, r, ah, threshold=ALERT["threshold"],
                                   refractory_h=ALERT["refractory_min"] / 60.0)
        learned_alert[idx] = a

    # ---- hybrid (Algorithm 8): one event per cycle, independent cooldowns --
    hybrid_alert = learned_alert | rule_alert

    results = {
        "config": {"patients": 240, "test_patients": 48,
                   "rule_seed": RULE_SEED, "cooldown_s": COOLDOWN_S,
                   "learned_threshold": ALERT["threshold"],
                   "learned_refractory_min": ALERT["refractory_min"]},
        "honesty": honesty,
        "variants": {
            "learned_only": variant_stats(df, learned_alert),
            "rule_only": variant_stats(df, rule_alert),
            "hybrid": variant_stats(df, hybrid_alert),
        },
        "disclaimer": "synthetic pipeline demonstration - not clinically validated",
    }
    OUT_JSON.write_text(json.dumps(results, indent=1))
    print(f"\nexported -> {OUT_JSON}")
    for name, s in results["variants"].items():
        print(f"  {name:12s} alerts {s['n_alerts']:>4} | true {s['n_true_alerts']:>3} "
              f"| sens {s['sensitivity']:.2f} | PPV {s['ppv']:.2f} | "
              f"alerts/day median case "
              f"{np.median(s['alerts_per_day_case']):.1f} / ctrl "
              f"{np.median(s['alerts_per_day_ctrl']):.1f}")
    print("PASS: rule branch + hybrid replay complete.")


if __name__ == "__main__":
    main()

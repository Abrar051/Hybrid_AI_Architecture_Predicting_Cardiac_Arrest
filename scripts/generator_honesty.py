"""Generator honesty test (reviewer point 2): do case and control patients
differ outside the drift window?

Compares pre-drift case windows (hours-to-arrest > DRIFT_H) with control
windows on hr/sbp/dbp/spo2, aggregated to PATIENT-LEVEL means first (windows
within a patient are autocorrelated; pooling them would make the p-values
meaninglessly small). KS + Welch t-test per channel -> outputs/generator_honesty.json.

Usage: python scripts/generator_honesty.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from config import CACHE_EVENTS, CACHE_VITALS, DRIFT_H, OUTPUT_DIR  # noqa: E402

CHANNELS = ["hr", "sbp", "dbp", "spo2"]


def main():
    vitals = pd.read_parquet(CACHE_VITALS)
    events = pd.read_csv(CACHE_EVENTS)
    case_ids = set(events["patient_id"])
    ctrl_ids = set(vitals.loc[~vitals["is_case"], "patient_id"].unique())

    # pre-drift case windows: more than DRIFT_H before arrest
    case = vitals[vitals["patient_id"].isin(case_ids)]
    tte = (case["arrest_h"] - case["t_end_h"]).to_numpy()
    pre_drift = case[tte > DRIFT_H]
    ctrl = vitals[vitals["patient_id"].isin(ctrl_ids)]

    out = {"drift_h": DRIFT_H, "n_case_patients": len(case_ids),
           "n_ctrl_patients": len(ctrl_ids),
           "n_case_windows_pre_drift": int(len(pre_drift)),
           "n_ctrl_windows": int(len(ctrl)),
           "tests": {}}
    print(f"pre-drift case windows: {len(pre_drift):,} from {len(case_ids)} cases | "
          f"control windows: {len(ctrl):,} from {len(ctrl_ids)} controls")

    for ch in CHANNELS:
        case_mean = pre_drift.groupby("patient_id")[ch].mean()
        ctrl_mean = ctrl.groupby("patient_id")[ch].mean()
        ks = stats.ks_2samp(case_mean, ctrl_mean)
        welch = stats.ttest_ind(case_mean, ctrl_mean, equal_var=False)
        out["tests"][ch] = {
            "case_patient_mean": float(case_mean.mean()),
            "ctrl_patient_mean": float(ctrl_mean.mean()),
            "ks_p": float(ks.pvalue),
            "welch_p": float(welch.pvalue)}
        print(f"  {ch:<5} case mean {case_mean.mean():7.2f} | ctrl mean "
              f"{ctrl_mean.mean():7.2f} | KS p={ks.pvalue:.3f} | "
              f"Welch p={welch.pvalue:.3f}")

    n_tests = len(CHANNELS) * 2          # KS + Welch per channel
    alpha_adj = 0.05 / n_tests           # Bonferroni across channels x tests
    out["n_tests"] = n_tests
    out["alpha_adj"] = alpha_adj
    (OUTPUT_DIR / "generator_honesty.json").write_text(
        json.dumps(out, indent=1))
    sig = any(t["ks_p"] < alpha_adj or t["welch_p"] < alpha_adj
              for t in out["tests"].values())
    verdict = ("no detectable case/control difference outside the drift window "
               f"(all p > Bonferroni-adjusted {alpha_adj:.4f} across "
               f"{n_tests} tests)")
    if sig:
        verdict = ("WARNING: a channel differs outside the drift window after "
                   "multiple-testing correction - inspect before reporting")
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
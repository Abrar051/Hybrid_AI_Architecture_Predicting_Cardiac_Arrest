# Response to Reviewer — Round 2

Separate document accompanying the revised manuscript
(`../MDPI_Hybrid_Revision/main.tex`). Point-by-point reply to the ten reviewer
comments in `REVIEWER_LETTER_ROUND2.md`.

**Status:** Phases A (points 2+3), B (point 1), and C (point 4) complete with
numbers below. MIMIC-III credentialed evaluation remains impossible without
access to the protected database (stated plainly in the revision).

---

## Point 1 — Only the learned branch is tested

**We agree.** The rule-based evidence branch, the four trigger states, the
per-rule 180 s cooldowns, and the hybrid decision agent were implemented and
evaluated in the revision (`scripts/rule_branch.py`).

What changed:

- The synthetic generator now produces the three manual-observation channels
  the rules require — respiratory rate, temperature, and the ACVPU
  consciousness level — plus a PPG-derived respiratory rate estimate for the
  respiratory-disagreement component. The new channels couple to the same
  drift ramp as the original signals, and a generator honesty check confirmed
  no case/control difference outside the drift window after Bonferroni
  correction.
- NEW section "Rule branch and hybrid replay" (Section 4.2) replays all three
  configurations — learned only, rule only, and the Algorithm 8 hybrid — over
  the 48 held-out test patients.

Key numbers (synthetic, 48 held-out patients):

| Configuration | Alerts | True alerts | Sensitivity | PPV | Alerts/patient-day (case; control) |
|---|---|---|---|---|---|
| Learned only | 755 | 136 | 1.00 | 0.18 | 8.5; 6.3 |
| Rule only | 29 | 29 | 1.00 | 1.00 | 0.6; 0.0 |
| Hybrid | 782 | 163 | 1.00 | 0.21 | 9.2; 6.3 |

The rule branch fires rarely but precisely (all 29 alerts true), later than the
learned branch (lead 1.2–3.0 h vs 3.0–5.9 h), because its components only
activate in the final phase of the synthetic drift. We state plainly that the
near-perfect rule precision is a property of the synthetic construction (the
components are defined on the drift channels), not a clinical estimate. The
alarm burden of the hybrid is dominated by the learned branch, which answers
the open question previously left in Section 3.10 — the sentence now cites the
measured burden.

The Algorithm 8 / Section 3.10 contradiction (all eligible triggers vs
highest-priority rule only) was resolved in the text: Algorithm 8 collects
every eligible trigger into a single evidence record, and Section 3.10 matches.

## Point 2 — Synthetic signal as a patient-level shortcut

**We agree.** The case/control difference was the drift term only. The
revision keeps this construction — and now proves it:

- 240 patients (seed 42), 122 events, 175,525 windows; 153/39/48
  train/val/test split by patient, temporal hold-out for test.
- Generator honesty checks (pre-drift case vs control): 8 tests for the
  original channels, 6 further tests for the new rule channels — all clean
  after Bonferroni correction. Only the drift differs.
- The 21.7 h "first true alert" false-alert claim is removed with the
  two-patient replay that produced it; true alerts now use the within-horizon
  definition (arrest inside the 6 h alert window).
- The 6 h Brier score remains honestly reported as worse than a constant
  forecast (0.136 vs 0.069), with the calibration figure.
- The abstract no longer headlines 0.891 / 21.7 h.

## Point 3 — Replay too small; numbers disagree

**We agree, and the evaluation was rebuilt** as requested ("a few hundred
synthetic patients, proper held-out test set, several seeds, regenerate every
table from one run"):

- Held-out test set: 48 patients (29 events), 26,635 windows — never seen in
  training.
- Every confidence interval is a patient-level bootstrap (1000 resamples of
  the 48 test patients), which cannot degenerate the way the two-patient
  interval did.
- Three model seeds (42, 1, 2); 6 h validation AUROC 0.949/0.944/0.937.
- All tables (baselines, held-out test, alerts, ablations, robustness,
  SDDB) are regenerated from this single run.

Held-out test results (seed 42):

| Horizon | Positives | AUROC (95% CI) | AUPRC | Brier (constant) |
|---|---|---|---|---|
| 1 h | 261 | 0.998 (0.997, 1.000) | 0.943 | 0.002 (0.010) |
| 6 h | 2,001 | 0.861 (0.838, 0.884) | 0.734 | 0.136 (0.069) |
| 24 h | 8,078 | 0.534 (0.490, 0.580) | 0.462 | 0.256 (0.211) |

All 29 test patients with an arrest were alerted within the 6 h horizon; first
true alert leads 3.0–5.9 h (median 4.5 h). PPV at sensitivity 0.8 is 0.977 at
1 h, 0.170 at 6 h, 0.294 at 24 h — reported without flattering framing.

## Point 4 — Holter denominators + inversion

**We agree on all four sub-points.** The revision:

- Adds the denominators to the Holter stress test (1798 five-minute windows,
  20 records): the 1 h task compares 176 positive windows from all 20 records
  against 1622 negatives from 20 records; the 6 h task compares 1107 positives
  from 20 records against 691 negatives from 13 records; the 24 h horizon has
  only 17 negatives and is not evaluated.
- Recomputes the identity probe under the requested protocol: a linear probe
  fitted on train-fold window contexts and scored on held-out fold windows,
  with chance defined as the majority-class share (0.356). No variant
  identified a single held-out window correctly (accuracy 0.000 for every
  variant, below the majority-class chance). Patient-specific patterns learned
  within the training distribution do not transfer to unseen patients; the
  earlier within-train probe accuracies (0.13–0.47) are now reported as
  within-cohort overfitting, and the text explains both.
- Adds logistic-regression and gradient-boosting baselines on the engineered
  window features with the same grouped folds and record-level bootstrap
  intervals: LR 1 h 0.574 (0.487, 0.652), 6 h 0.476 (0.394, 0.569); GBM
  1 h 0.483 (0.426, 0.544), 6 h 0.362 (0.306, 0.423). Neither exceeded chance,
  which localises the engineered-feature 1 h signal to the temporal
  aggregation of the FEAN architecture rather than instantaneous window
  statistics.
- Explains the below-chance 6 h CV results: several folds become confidently
  inverted (training loss decreases while held-out loss stays flat or rises),
  and pooling inverted folds yields sub-chance aggregates; fold-level curves
  are shown and only the aggregate with bootstrap intervals is reported.
- Adopts the ESC/NASPE 1996 standard: HRV features are now computed over the
  full 5-minute window instead of a 30 s segment, and the whole SDDB
  evaluation (zero-shot, CV, ablations, baselines, probes) was re-run on the
  new features. Re-run results (all with patient-level bootstrap CIs): direct
  transfer 1 h 0.544 / 6 h 0.428; CV variants 1 h 0.594–0.695 and 6 h
  0.407–0.504; engineered-features-only 1 h 0.753 (0.690, 0.829) remains the
  strongest single block. The honest negative conclusion is unchanged.
- MIMIC-III: the loader is prepared but credentialed access was not granted;
  the letter states this constraint plainly and the limitation remains in the
  paper.

## Point 5 — Clinical setting / rule branch

**We agree.** The revision states a single setting (a monitored general
ward/step-down unit), ACVPU is a manual nursing entry (now generated in the
synthetic cohort), all rule weights and thresholds are flagged as provisional
unvalidated placeholders, the multiplier was removed, and the fallback claim
was corrected (four of six components genuinely require ECG/PPG-derived
quantities; the text says so).

## Point 6 — Evidence record

**We agree.** The evidence record is now fully defined: leading drivers
(largest component contributions), a descriptive confidence value, and
time-to-event normalisation (tau/24 sigmoid); the censoring limitation of the
TTE head (trained on positive windows only) is stated; the LLM explanation
service was removed and replaced with templated summaries; "agents" are
defined as deterministic software services, not LLMs.

## Point 7 — Reproducibility

**We agree.** Seeds (42; model seeds 42/1/2), hyperparameters, and split
procedure are reported in the paper and in the code; the ESC/NASPE 5-minute
HRV standard is now used for the SDDB features with a full re-run
(Point 4); repository commit hashes of all third-party models are recorded.
A code DOI will be provided with the accepted version.
## Point 8 — Telegram

**We agree.** Telegram is presented as a prototype notification channel only,
with the encryption caveat stated.

## Point 9 — References

**We agree.** Added: NEWS2 (RCP 2017), SDDB/PhysioNet (Goldberger 2000,
Greenwald 1986), MIMIC-III (Johnson 2016), the Ext-CA annotation set, PCGrad
(Yu et al., NeurIPS 2020), McSharry 2003; [30] corrected to Ding et al.;
PaPaGei and Abbaspourazad updated to ICLR 2025/2024; the demographic-encoding
finding is cited to support the identity-probe results.

## Point 10 — Presentation

**We agree.** Hyphens restored in proper nouns; British spelling unified;
the stray paragraph moved to the Discussion; Figs 12–13 dropped; slow-risk and
robustness material retained; back-matter statements added.

---

## Items not delivered

- MIMIC-III credentialed evaluation — access not granted; stated plainly.
- MDPI template conversion — the revised manuscript retains the submission
  format per the editorial request (will be converted at acceptance).

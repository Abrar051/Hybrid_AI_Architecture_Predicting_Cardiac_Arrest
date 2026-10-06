# Reviewer comment verification — 2026-10-02

Verified against: executed `Pipeline.ipynb`, `cache/embeddings/`, `scripts/analyze_sddb.py`,
`scripts/ews_risk.py`, submitted paper `../MDPI_AI_RPM_AGENT.pdf` (source: extract of
`../MDPI_Submission.zip` → `../MDPI_Hybrid_Revision/`). The letter is saved verbatim in
`REVIEWER_LETTER_ROUND2.md`.

## Execution status (2026-10-06, Phase A COMPLETE 06:54)

- **Phase A (points 2+3) — DONE.** Embedding (27 groups), train_fean (3 seeds),
  ablations, notebook (executed, Pipeline.ipynb refreshed), fusion + fean v2
  tests, robustness (48-patient test set), SDDB zero-shot with ews_v2 models.
  Manuscript refreshed 2026-10-06 ~07:30: synthetic subsection rewritten (new
  Tables tab:baselines/test/alerts/ablation, fusion paragraph, figures 02-04
  replaced from fresh outputs), robustness table + intro/limitations text,
  SDDB zero-shot row (0.571/0.441). Compiles clean: 0 errors, 30 pages.
- Remaining: Phase B (rule branch + hybrid replay), Phase C (SDDB upgrades),
  response letter, MDPI template conversion.
- **Embedding crash + fix (2026-10-03):** the 2026-10-02 overnight run died at group 0  with `UnboundLocalError: cannot access local variable 'win_idx'` in
  `embed_windows.py` (the memory-optimization `del segs, win_idx` deleted the index
  needed for pooling) — both tags crashed at 16:07 right after ECG-FM chunk extraction;
  the machine then shut down at 23:53. FIXED: `del segs` only, plus per-kind chunk-emb
  reuse (`{tag}_{kind}_emb.npy` + `{kind}_seg.npz` cached -> skip extraction, reload
  win_idx from npz), so the already-extracted group-0 ECG embeddings are reused.
  Verified by 2-window smoke test (fresh + reuse paths). Relaunched with
  `python -u` + nohup (logs now stream per-group lines). Expected ~16-20 h wall with
  dev+test in parallel; the machine must stay ON. Done marker: `{dev,test}_window_*.npy`
  + `{dev,test}_meta.csv` all present.
- **Embedding OOM + mmap fix (2026-10-03, 10:42):** with two extractors loading
  simultaneously, the kernel OOM-killed dev's group-1 ECG extractor (exit -9; PyCharm
  ~2 GB also resident). Root cause: `np.load(npz)["ecg"].astype(np.float32)` in both
  extractors = full 1.44 GB copy + a second 1.44 GB astype copy + ~2 GB model per
  extractor (~5 GB peak each). FIXED: `np.load(..., mmap_mode="r")` + assert float32
  + per-batch `.copy()` (torch needs writable arrays); validated bit-identical
  (max diff 0.0) against the git-HEAD versions on both encoders. Dev relaunched
  (append-mode log); memory now ~7 GB headroom with both tags running.
- **Second OOM + overnight stop (2026-10-03 12:15 → 2026-10-04 09:23):** despite the
  mmap fix, the morning's OOM aftermath (0 GB available, 15/19 GB swap) killed dev's
  g1 extractor again (exit -9) and the test job died silently (last log line 11:42);
  machine shut down 01:14. Completed before the stop: dev g0; test g0-g2. Relaunched
  2026-10-04 09:53 after reboot (swap clean, PyCharm closed), tags staggered 3 min to
  desync segment-build memory spikes; 23 groups left, ETA ~midnight 2026-10-04.
  Lesson: extractor fixes are not enough while swap is full — keep the machine's
  desktop apps closed during embedding.
- **Embedding completed + train bug fixed (2026-10-06 ~01:50):** all 27 groups done
  and concatenated; the runner started train_fean. Attempt 1 failed on
  `TypeError: RiskPipeline.__init__() got an unexpected keyword argument
  'horizon_h'` — config's ALERT used `horizon_h` but the constructor takes
  `alert_horizon`. Fixed in config.py (key renamed, verified construct).
  Val AUCs from the failed attempt were 0.95 @ 6 h — training itself healthy.
  Runner retried with the fixed code (attempt 3).
- **test_fean_v2 identity bug (2026-10-06 04:49):** runner aborted after 3 identical
  `IndexError: Target 156 is out of bounds` in the adv phase — `ids` from
  `load_manifest_features` are factorized over the WHOLE dev set (0..191) but the adv
  head has `n_ids=n_train_patients` (153). Fixed: factorize within each subset
  (`sequence_ids(meta.loc[tr])` / `sequence_ids(meta.loc[va])`); removed a stray
  duplicate `make_sequences` val call. Verified idtr max 137 < 153, idva max 35.
  Runner restarted 04:52, resumed at test_fean_v2.
- **Writing (points 5, 6, 8, 9, 10 + parts of 1, 2) — DONE** in
  `../MDPI_Hybrid_Revision/main.tex` (compiles clean, 2 passes, 0 undefined refs).
  - Abstract rewritten (no 0.891/21.7 h headline); agent terminology defined; LLM
    explanation service removed (templated summaries); Telegram prototype-only; setting =
    monitored general ward/step-down; consciousness = manual ACVPU entry; multiplier
    removed; weights flagged provisional; fallback claim fixed; evidence record defined
    (drivers, confidence, TTE = tau/24 sigmoid); censoring caveat; 4.1 "true alert"
    mislabel corrected + Brier admission; Algorithm 8 collects all eligible rules;
    20 events/h alarm-burden limit acknowledged; 7 citations added ([30] -> Ding C,
    [28] ICLR 2025, [32] ICLR 2024 + demographic finding cited); hyphens restored in
    proper nouns; British spelling unified; 3.7 paragraph -> Discussion; Figs 12/13
    dropped; back-matter added; TODO(PHASE-A) markers on the superseded numbers.
  - LaTeX fixes: fontawesome5 not installed -> icons stripped from Figure 1;
    `\usepackage[expansion=false]{microtype}` (cm-super bitmap font issue).
- **Remaining:** Phase A numbers into Tables 3-8; Phase B hybrid replay; Phase C SDDB
  text; MDPI template conversion at final formatting.

## Phase A implementation state

**New cohort (built):** 240 patients, seed 42, P=0.5 -> 175,525 windows, 122 cases.
Splits (cache/splits.json): train 153 (74 cases) / val 39 (19) / test 48 (29) — temporal
hold-out 20% for test, stratified 20% of dev for val. Manifests: dev 24,630 windows
(174 PPG-having dev patients, 12 h spans with 0.25 h floor), test 26,635 (48 patients,
full stays). Engineered features cached; PTT corr 0.977.

**New scripts:** `config.py` (single config source), `eval_utils.py` (horizon_labels
with NaN exclusions, make_sequences[_memmap], sequence_ids, patient_bootstrap with
explicit patients, true_alert, alert_from_risks, load_manifest_features, load_splits),
`build_manifests.py`, `embed_all.py` (grouped/resumable embedding, concat + row-count
assert), `train_fean_models.py` (seeds {42,1,2} -> cache/models/ews_v2, canonical copy),
`train_ablations.py` (ablations_v2; NOTE: fixed old bug where val AUCs used unmasked
features), `generator_honesty.py` (outputs/generator_honesty.json — clean verdict after
Bonferroni across 8 tests), `test_eval_utils.py` (selftests pass).

**Modified:** `embed_windows.py` (OMP/MKL threads=4, del-segments gc), `test_fean_v2.py`
+ `test_fusion.py` (eval_utils loader, splits.json, batch 256, n_ids dynamic, fusion
exports outputs/fusion_metrics.json), `robustness.py` (v2 rewrite: dev/test caches,
robustness_v2 dir, per-patient streaming, patient-bootstrap CIs on clean + feature
perturbations, waveform tags = 200-window point estimates, segments regenerated via
cut_segment), `analyze_sddb.py` (`--model-dir`, `--zero-shot-only` merges into
sddb_v2_metrics.json), `Pipeline.ipynb` (cells 2/6/16/18/20/22/24/26/28/29/32 rewritten,
markdown 17/21/23/25/27/33 updated, ALL outputs cleared).

**Old artifacts:** outputs/metrics_60p_archive.json + robustness_metrics_60p_archive.json;
old model dirs (ews_v1, ablations, robustness, sddb_v2) untouched; old embedding caches
deleted (train_*/replay_*/rob_*), sddb_*/demo_* kept.

**Sequence after embedding completes** (embedding = the only blocker; now
automated by `scripts/phase_a_runner.py`, launched 2026-10-04 10:19 — supervises
embedding, retries each step 3×, resumes from cache/phase_a_state.json; after a
reboot re-run `bash scripts/phase_a_launch.sh`):
1. `python scripts/train_fean_models.py` (~1 h)  2. `python scripts/train_ablations.py`
(~1 h)  3. notebook: `jupyter nbconvert --to notebook --execute Pipeline.ipynb --output
/tmp/pipeline_check.ipynb --ExecutePreprocessor.timeout=7200` FROM REPO ROOT (cells hit
caches; cell 22 no-ops once embeddings exist), then copy over Pipeline.ipynb
4. `test_fusion.py` + `test_fean_v2.py`
5. `robustness.py` (~1 h)  6. `analyze_sddb.py --zero-shot-only --model-dir
cache/models/ews_v2`  7. number pass vs this file; refresh manuscript tables.

**Gotchas for the next session:**
- Embedding progress is NOT visible in logs/ (nohup buffers stdout) — check
  `cache/embeddings/` group files; both final concats exist -> done. Jobs are resumable
  (per-group skip). Do not restart the completion watcher unless memory is healthy.
- All training uses batch_size=256 + float32 memmap sequences (RAM: ~2.7 GB/seq set);
  never full-batch at 240-patient scale.
- The notebook must be pre-seeded: embedding + trained models BEFORE nbconvert, else
  cell 22 blocks for ~10 h inside the executor.
- `RiskPipeline` stats keys are `mean`/`std` — keep them in v2 stats.json.
- Workspace for the paper: `../MDPI_Hybrid_Revision/` (NOT in git). Repo manuscript/
  is the old version; do not edit it for this revision.

Verdicts: every numeric claim in the review checks out. Details below.

## 1. Only the learned branch is tested — CONFIRMED
- Rule branch (3.9), decision agent (3.10), delivery (3.11) are paper pseudocode only.
  No implementation of the 6-component score, trigger states, cooldowns, or decision
  agent exists in this repo or in the AIRPM backend project.
- Generator (Pipeline.ipynb cell 6) emits HR, SBP, DBP, SpO2 only. Missing for NEWS2:
  respiratory rate, temperature, consciousness (ACVPU). Fix: extend generator.
- Contradiction: 3.10 text says all active triggers combined into one event;
  Algorithm 8 takes only the highest-priority rule. Fix text or algorithm.
- 180 s cooldown = up to 20 alerts/hour per rule (3600/180). Undercuts 5.1 alarm-burden claim.

## 2. Synthetic signal as patient-level shortcut — CONFIRMED
- DRIFT_H = 6.0 h, quadratic ramp, only term that differs between case and control.
  Generator has NO case/control differences outside drift (same baseline distributions).
  Add an explicit test asserting this.
- Patient 37: 6 h risk >= 0.5 at t = 4.2 h, arrest at 25.8 h -> "21.7 h first true alert"
  is a FALSE alert for a 6 h horizon. Label bug: cell 26 counts any pre-arrest alert
  as true (`a["t_h"] < arrest_h`). Fix: require 0 < arrest_h - t_h <= 6 h.
- Replay 24 h AUROC 0.946 vs validation 0.667: confirmed from executed outputs.
- Patients 37 and 52 NOT in training: confirmed from cache/embeddings/*_meta.csv (no overlap).
- 6 h Brier 0.315 vs constant forecast ~0.11 at observed prevalence (72/604): worse. Confirmed.
- Remove 0.891 / 21.7 h from the abstract headline.

## 3. Replay too small, numbers disagree — CONFIRMED (all four)
- 6 h CI [0.825, 0.922] cannot come from cell 28's patient_bootstrap: resampling the 2
  replay patients with replacement, every valid draw gives the identical full-set AUROC
  -> degenerate single-value CI. Printed interval is a stale output from an earlier
  (window-level) bootstrap. Algorithm 11 cannot produce it.
- Validation concat spread 0.769-0.859 vs replay 0.891/0.929 > every effect in Table 8 (max 0.03).
- 247 case alerts: cell 29 variant_replay counts windows with risk >= 0.5 with NO
  refractory, unlike the main replay (simulate_patient applies 60 min). Max possible
  with refractory in 26 h is ~26.
- Positive counts 12/72/288: replay labels use tte > 0 with NO 15-min exclusion;
  SDDB labels use tte > 0.25 (analyze_sddb.py:154). Protocol inconsistency confirmed.
  With exclusion the 1 h count would be 9, not 12.
- Fix as reviewer says: a few hundred synthetic patients, proper held-out test set,
  several seeds, regenerate every table from one run.

## 4. Holter denominators + inversion — CONFIRMED
- Pre-onset span median 7.64 h. 6 h negatives come from only 12 of 20 records
  (the 690 "far" windows). Denominators from cache:
    1 h: 178 pos (20 rec) vs 948 + 690 neg
    6 h: 1126 pos (20 rec) vs 690 neg (12 rec)
- Below-0.5 6 h AUROCs with CIs excluding 0.5: real (Table 7). Cause = inverted folds +
  pooling; explain properly in the paper.
- Identity probe: fit on TRAIN-fold windows (80/20), chance = 1/n_records.
  Fix per reviewer: score on held-out test-fold windows, chance = majority-class rate.
  (analyze_sddb.py:185 identity_probe)
- No LR/GBM baselines on SDDB. Fix: add them on engineered features.
- MIMIC-III: loader ready (scripts/mimic_loader.py), blocked on credentialed access.

## 5. Clinical setting / rule branch — CONFIRMED
- Three settings at once (ICU waveforms, ward NEWS2, remote Telegram). Pick one.
- Consciousness level source unspecified; generator cannot produce it.
- Weights/thresholds/multiplier unsourced; multiplier raises one subgroup's alert rate by design.
- 4 of 6 components need ECG or PPG (RMSSD->ECG, perfusion->PPG, resp disagreement->PPG,
  signal quality->ECG/PPG). 3.9's "fallback for missing waveforms" claim is false as written.

## 6. Evidence record — CONFIRMED
- "Leading drivers", confidence values, TTE normalization: never defined.
- TTE trained only on positive windows (delta_i in eq. 7) -> censoring ignored, yet TTE
  appears in every warning.
- LM explanation service: described, never tested.
- "Agents" are deterministic services; terminology will read as LLM agents. Rename or define.

## 7. Reproducibility — CONFIRMED
- Repo has values (seed 42, config cell, hyperparameters) but paper does not report them.
  No code DOI.
- HRV from one 30 s segment; ESC/NASPE 1996 standard is the full 5-min window.
  Adopting this changes the SDDB engineered features (the only real-data signal)
  -> full SDDB re-run needed.

## 8. Telegram — CONFIRMED
- Cloud chats not end-to-end encrypted. Simplest fix: present Telegram as prototype channel only.

## 9. References — CONFIRMED
- Missing: NEWS2 (RCP 2017), SDDB/PhysioNet (Goldberger 2000), MIMIC-III (Johnson 2016),
  Ext-CA annotation set, PCGrad (Yu et al., NeurIPS 2020), synthetic beat model
  (McSharry 2003).
- [30] wrong: SiamQuality first author is Cheng Ding (PMB 45:085004, arXiv:2404.17667),
  not Hu.
- [28] PaPaGei -> ICLR 2025. [32] Abbaspourazad -> ICLR 2024.
- [32] embeddings encode demographics/health info: supports the 4.2 identity finding; cite it.

## 10. Presentation — CONFIRMED
- No-hyphen rule broke proper nouns: ECG-FM, PaPaGei-S, MIMIC-III, Martinez-Alanis,
  time-to-event, in-hospital. Restore hyphens in names/compounds only.
- British/American spelling mix (behaviour/multicentre vs generalize/labeling). Pick one.
- Move last paragraph of 3.7 to Discussion.
- Drop Figs 12-13 (duplicate Table 8); consider slow-risk + robustness as supplementary.
- Add MDPI template and back-matter statements.

## Fix priority (proposed)
1. Point 3 + 2: new synthetic cohort (hundreds of patients), fix label/exclusion bugs,
   fix bootstrap, proper test set, several seeds, regenerate all tables. Removes the
   0.891/21.7 h headline problem at the root.
2. Point 1: extend generator with RR/temp/consciousness + implement rule branch +
   hybrid replay (learned-only vs rule-only vs hybrid, alerts per patient-day).
3. Point 4: SDDB denominators table, held-out-window identity probe with majority-class
   chance, LR/GBM baselines, inversion explanation.
4. Points 5-6: setting rewrite, evidence-record definitions, agent terminology.
5. Points 7-10: HRV 5-min window (+ SDDB re-run), reproducibility section, Telegram
   framing, references, MDPI template, hyphen/spelling pass.
6. Blocked: MIMIC-III credentialed evaluation.
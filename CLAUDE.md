# Cardiac Arrest Detection — EWS experiment

Research prototype for short-horizon (1/6/24 h) cardiac-arrest early warning from ECG + PPG + BP + HR. Full plan: `cardiac_arrest_ews_plan.md`. Main deliverable: `Pipeline.ipynb` (one notebook, sections 0–10, all implemented and validated in synthetic-data mode; real-data work lives in `scripts/`).

## Rules (from the plan, section 7)

1. Never commit data, credentials or model weights. `.gitignore` covers `data/`, `cache/`, `weights/`, `third_party/`.
2. Split by patient at all times. Assert no patient appears in both train and test.
3. Fix seeds and log them. Print class counts and event counts at every split.
4. Exclude signal from resuscitation and post-arrest periods; assert this in code.
5. Cache every expensive step. The notebook must be re-runnable from any section.
6. Fail loudly with clear messages when a repo, weight file or dataset is missing.
7. Test each section on synthetic data before touching real data.
8. Do not report headline metrics without confidence intervals. With 31 events, say plainly that results are a pipeline demonstration.
9. Do not claim clinical validity anywhere in outputs or markdown cells.
10. Record versions and commit hashes of `ecg-fm`, `fairseq-signals`, `papagei`, `pulseppg`.

## Environment

- Local dev env: conda `pyprime` (Python 3.13; numpy/pandas/scipy/sklearn/matplotlib/wfdb/pyarrow/torch present; lifelines missing).
- Embedding envs (built, working): conda `ecgfm_env` (Python 3.10, torch 2.14.0+cpu, fairseq-signals editable install, numpy 1.23.5, transformers 4.36.2) and conda `papagei_env` (Python 3.10, torch 2.14.0+cpu, papagei requirements + pyPPG==1.0.41 + peakutils).
- Extraction runs via subprocess with `PYTHONNOUSERSITE=1` (a broken transformers in `~/.local` shadows env packages otherwise). See `scripts/extract_ecgfm.py` / `scripts/extract_papagei.py`.
- **ECG-FM inference API** (verified): `build_model_from_checkpoint(ckpt)` then `model.extract_features(src, None)` → `res["x"]` is (B, T, 768) batch-first; mean-pool over T for embeddings. Do NOT call `model(source=...)` on the pretrained checkpoint — `Wav2Vec2CMSCModel.forward` hardcodes `return_features=True` and runs the pretraining loss head.
- Weights (downloaded): `weights/mimic_iv_ecg_physionet_pretrained.pt` (1.1 GB), `weights/mimic_iv_ecg_finetuned.pt` (1.1 GB), `weights/papagei_s.pt` (23 MB).
- Repo commits: ecg-fm `9f926f19`, fairseq-signals `f8f0ff1c`, papagei `0c537dad`, pulseppg `716eaf9c`.
- Validate the notebook with: `jupyter nbconvert --to notebook --execute Pipeline.ipynb --output /tmp/pipeline_check.ipynb`
- Disk is tight (96% full): prefer `--no-cache-dir` for pip, CPU-only torch wheels.

## Status / next steps

- [x] Sections 0–10 implemented and validated end-to-end in `Pipeline.ipynb` (34 cells, all pass on synthetic data; full execution ~15 min on CPU, everything cached):
  - S0 config/env · S1 synthetic cohort · S2 labels · S3 preprocessing · S4 engineered features (HRV, ectopy, BP stats, per-beat PPG-foot PTT — corr 0.98 with ground truth) · S5 baselines (LR, GBM, MEWS) · S6 ECG-FM/PaPaGei embeddings (subprocess envs) · S7 FEAN temporal model · S8 full evaluation (bootstrap CIs, calibration, lead-time bins, failure tables) + ablations · S10 export
- [x] SDDB real-data stress test v1 (`scripts/analyze_sddb.py`): 20 records, 1829 windows, zero-shot + 5-fold grouped CV → honest negative result (zero-shot AUROC ~0.52; CV 6 h AUROC 0.39, below chance). Manuscript + journal summary written (`manuscript/main.tex`, `JOURNAL_SUMMARY.md`).
- [x] Section 7 v2 implemented in `scripts/ews_risk.py` (backward compatible with v1 checkpoints): `GradReverse` identity-adversarial head (FEAN `n_ids`), `pcgrad_step`, shared `train_fean` with per-epoch val history; `FEAN.encode()` exposes contexts for probes. Synthetic test passes (`scripts/test_fean_v2.py` — on synthetic the identity probe stays high by design: case/control IS a patient-level property). SDDB v2 CV re-run done → `outputs/sddb_v2_metrics.json` (corrected record-level bootstrap CIs; `fix_sddb_v2_cis.py`).
- [x] Multimodal fusion + missing masks (reviewer point): `ModalityFusion` (gate/attn/weight) + per-modality presence masks with learned missing embeddings in `ews_risk.py`; synthetic test `scripts/test_fusion.py`; SDDB comparison → `outputs/sddb_v2_fusion_metrics.json`.
- [x] Section 9 slow-risk layer (`scripts/slow_risk.py`): QT/QTc, TWA index, QRS width, long-term HRV per 30 s segment, aggregated over 3 h blocks, LR with grouped CV, 0.5 h label margin rule; synthetic selftest passes; SDDB eval → `outputs/slow_risk_metrics.json`.
- [x] MIMIC-III Ext-CA loader prep (`scripts/mimic_loader.py`): dataset.csv parsing (row_id/subject_id/hadm_id/file/cardiac_arrest_start/end), record-start datetime from wfdb filename, window manifest with exclusion reasons + asserts, patient split helper, `--check` fails loudly until credentialed data lands, `--selftest` passes.
- [ ] Real data: MIMIC-III / MIMIC-IV waveforms + MIMIC-III-Ext-CA annotations (credentialed PhysioNet access pending)
- [ ] GPU embedding of the full cohort (CPU-bound here)
- [x] Robustness evaluation (`scripts/robustness.py`, reviewer point): 5 perturbation families on the synthetic replay set (2 patients) with concat/gate/weight models — ECG noise SNR 20/10/5 dB + wander/mains/flatline/saturation/PPG noise re-embedded through the real encoders, vitals noise 0.5/1/2× std, missing PPG, delays 5/10/15 min, mid-stay PPG sensor failure. Result: every perturbation within 0.03 of clean 6 h AUROC (0.929-0.945); only delays degrade directionally (mild, 0.929→0.920 at 15 min). Caveats documented: synthetic signal lives in vitals/eng blocks; gates bypassed by design; point estimates. → `outputs/robustness_metrics.json` + 2 figures.
- Findings on synthetic data (not clinical evidence): vitals+engineered ≈ full model at 6 h (embeddings add little on synthetic drift — expected); alert-level PPV at sens 0.8 for 6 h is low (~0.3); the 1 h horizon is trivial on synthetic data. Fusion: gate/weight ≈ concat on synthetic (6 h AUC 0.77-0.86); attention mode degrades with PPG removed at eval.
- Findings on SDDB v2 (not clinical evidence; 20 records, patient-bootstrap CIs, protocol fixed so histories never cross record/span boundaries): zero-shot 1 h 0.34 / 6 h 0.45. CV 6 h: all deep variants at/below chance (plain 0.402, adv 0.436, adv_pcgrad 0.459, gate 0.386, attn 0.473, weight 0.394); CV 1 h: engineered features carry the signal (no_emb 0.767), embeddings dilute it (full 0.632). Identity probe stays 0.42-0.47 (chance 0.05) even with the adversarial head — identity leakage not removed. Slow-risk layer below chance (6 h 0.313). Attention salience: model spreads attention over history, ~zero on the current window. Fold-level metrics swing 0↔1; only aggregate + bootstrap numbers are meaningful.
- Manuscript: `manuscript/main.tex` updated with v2/fusion/slow-risk results; compiles (pdflatex, exit 0). Note: `algorithm.sty`/`algorithmicx` are installed in `~/texmf/` (no sudo); `JOURNAL_SUMMARY.md` + `.docx` (pandoc) updated.
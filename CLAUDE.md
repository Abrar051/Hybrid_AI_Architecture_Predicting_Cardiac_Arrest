# Cardiac Arrest Detection — EWS experiment

Research prototype for short-horizon (1/6/24 h) cardiac-arrest early warning from ECG + PPG + BP + HR. Full plan: `cardiac_arrest_ews_plan.md`. Main deliverable: `Pipeline.ipynb` (one notebook, sections 0–10; currently 0–3 implemented, synthetic-data mode).

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
- [ ] Real data: MIMIC-III / MIMIC-IV waveforms + MIMIC-III-Ext-CA annotations (credentialed PhysioNet access)
- [ ] GPU embedding of the full cohort (CPU-bound here)
- [ ] Section 7 v2: identity-adversarial head, PCGrad
- [ ] Optional Section 9 slow-risk layer (ECG-SCD / Nightingale NTUH)
- Findings on synthetic data (not clinical evidence): vitals+engineered ≈ full model at 6 h (embeddings add little on synthetic drift — expected); alert-level PPV at sens 0.8 for 6 h is low (~0.3); the 1 h horizon is trivial on synthetic data.
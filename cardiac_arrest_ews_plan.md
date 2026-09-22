# Cardiac arrest early-warning experiment: implementation plan for Claude Code

Deliverable: one Jupyter notebook, `ews_experiment.ipynb`, that runs the whole experiment from data loading to evaluation.

Status of this document: written from paper and repository reading. Items marked **VERIFY** were not confirmed and must be checked before relying on them.

Research prototype only. Nothing here is validated for clinical use.

---

## GitHub links index

Each URL below was checked and resolves (HTTP 200). Clone third-party repos into `third_party/`.

**Core repos (required)**

| Purpose | Link |
|---|---|
| ECG foundation model (ECG-FM) | https://github.com/bowang-lab/ecg-fm |
| ECG-FM inference quickstart notebook | https://github.com/bowang-lab/ECG-FM/blob/main/notebooks/infer_quickstart.ipynb |
| ECG-FM command-line inference notebook | https://github.com/bowang-lab/ECG-FM/blob/main/notebooks/infer_cli.ipynb |
| fairseq_signals (training/inference framework ECG-FM depends on) | https://github.com/Jwoo5/fairseq-signals |
| fairseq_signals ECG preprocessing scripts | https://github.com/Jwoo5/fairseq-signals/tree/master/scripts/preprocess/ecg |
| PaPaGei (PPG foundation model) | https://github.com/Nokia-Bell-Labs/papagei-foundation-model |
| PaPaGei example notebook | https://github.com/Nokia-Bell-Labs/papagei-foundation-model/blob/main/example_papagei.ipynb |
| Pulse-PPG (PPG foundation model, wearable field data) | https://github.com/maxxu05/pulseppg |

**Supporting repos**

| Purpose | Link |
|---|---|
| WFDB-Python (read PhysioNet/MIMIC waveform records in section 1) | https://github.com/MIT-LCP/wfdb-python |
| ECG-SCD (optional slow-risk layer, Nature sudden cardiac death paper) | https://github.com/alexmschubert/ECG-SCD |
| PulseDB (optional: curated ECG/PPG/ABP segments derived from VitalDB; VERIFY exact contents and licence) | https://github.com/pulselabteam/PulseDB |
| ECG-FM benchmarking (optional reference: compares ECG foundation models across public datasets) | https://github.com/AI4HealthUOL/ecg-fm-benchmarking |

**No public repository found** (searched GitHub and the web): Wav2Arrest / Wav2Arrest 2.0 and PPG-GPT. Wav2Arrest is reimplemented from the paper (arXiv 2509.21695); check the paper's code statement or contact the authors.

Model weights are not on GitHub: ECG-FM on Hugging Face (`wanglab/ecg-fm`), PaPaGei on Zenodo (record 13983110), Pulse-PPG on Zenodo (DOI 10.5281/zenodo.17270930). See section 3.

---

## 1. Goal and scope

Build a system that takes continuous ECG, PPG, blood pressure and heart rate and outputs a rising-risk score for cardiac arrest at short horizons (1 h, 6 h, 24 h).

Design reference: **Wav2Arrest 2.0** (arXiv 2509.21695) and its predecessor (arXiv 2502.08612). It predicts arrest up to 24 h ahead from ICU PPG using a foundation-model feature extractor, a BLSTM-attention aggregator, time-to-event targets and an identity-adversarial step. Our version extends this to multimodal input (ECG + PPG + BP + HR).

Non-goals for v1:
- No weeks-to-months risk layer (optional section 9 only).
- No real-time deployment, no clinical claims.

## 2. Decisions to confirm before coding

| Question | Why it matters |
|---|---|
| Do we have our own labeled arrest events with timestamps? | If yes, it is the primary dataset and public data is for pretraining/testing only. If no, we only build a pipeline prototype on public data. |
| How many arrest events, how many patients? | With fewer than a few hundred events, use frozen encoders and small heads only. |
| Signal source: ICU bedside monitor or wearable? | Sampling rates and lead count differ. Bedside ICU data is the only setting with published validation. |
| Sampling rates of each stream | Decides resampling for each encoder (see section 4). |

Add a `USE_SYNTHETIC` flag so the whole notebook runs end to end on generated data with no data access. This is how Claude Code should test each section.

---

## 3. Repositories and model assets

All are open source. Clone into `third_party/`. Check each licence before use beyond research.

### 3.1 ECG encoder: ECG-FM

- Repo: https://github.com/bowang-lab/ecg-fm (MIT licence)
- Depends on **fairseq_signals**: https://github.com/Jwoo5/fairseq-signals (install per its top-level README)
- Weights: Hugging Face, https://huggingface.co/wanglab/ecg-fm
  - `mimic_iv_ecg_physionet_pretrained.pt` (pretrained on MIMIC-IV-ECG v1.0 and PhysioNet 2021)
  - `mimic_iv_ecg_finetuned.pt` (fine-tuned on MIMIC-IV-ECG labels)
  - Config files for reproducing training are on the same Hugging Face page
- Reference notebooks in the repo: `notebooks/infer_quickstart.ipynb` (inference and embeddings), `notebooks/infer_cli.ipynb` (command line inference)
- Preprocessing pipeline: `fairseq-signals/scripts/preprocess/ecg`
- Model: wav2vec 2.0 style, 90.9M parameters. Input: 12-lead ECG, 500 Hz, 5 s segments, z-score normalized. Embedding: 768-dimensional.
- Important: the public checkpoint was pretrained on MIMIC-IV-ECG and PhysioNet 2021. Those datasets are therefore **not independent test data** for this encoder.

### 3.2 PPG encoders (use one, benchmark both)

**PaPaGei** (ICLR 2025)
- Repo: https://github.com/Nokia-Bell-Labs/papagei-foundation-model
- Setup per README: conda env with Python 3.10, `pip install -r requirements.txt`, `pip install pyPPG==1.0.41`
- Weights: Zenodo record 13983110 (https://zenodo.org/records/13983110). Save `papagei_s.pt` into `weights/`.
- Per the README example: resample to 125 Hz, 10 s segments, tensor shape `(n_segments, 1, 1250)`. PaPaGei-S returns a tuple, and the embeddings are the first element.
- Has an example notebook: `example_papagei.ipynb`

**Pulse-PPG** (UbiComp 2025, trained on field wearable data)
- Repo: https://github.com/maxxu05/pulseppg
- Setup: `conda env create -f env.yml`, `conda activate pulseppg`, `pip install -e .`
- Weights: Zenodo DOI 10.5281/zenodo.17270930, via `bash ./download_pulseppg.sh`
- Uses temporal pooling, so inputs are not tied to one length (README says 30 s inputs work). **VERIFY** the exact input sampling rate in the repo config before use.

**PPG-GPT** (used in Wav2Arrest): no public weights found. Do not plan around it.

### 3.3 Wav2Arrest

- No public repository found (searched GitHub and the web). **VERIFY** by checking the paper's code availability statement or contacting the authors.
- Plan: reimplement from the paper (section 6 of this document).

### 3.4 Optional slow-risk layer: ECG-SCD

- Repo: https://github.com/alexmschubert/ECG-SCD (Python 3.12 environment)
- ResNet for sudden cardiac death risk on 10 s 12-lead ECGs; pipeline built for the NTUH dataset from Nightingale Open Science. Swedish training data and weights are not public.
- Dataset access: https://docs.ngsci.org/access-data and https://docs.ngsci.org/datasets/arrest-ntuh-ecg/

---

## 4. Environments (important for the single-notebook constraint)

The repos have conflicting environments (PaPaGei Python 3.10, ECG-SCD Python 3.12, fairseq_signals pins older PyTorch versions). Plan for this from the start.

Recommended approach:
1. Main notebook environment: `ews_main` (PyTorch, scikit-learn, pandas, wfdb, scipy, lifelines, matplotlib).
2. Embedding extraction cells call separate environments through `subprocess` and write embeddings to disk (`cache/embeddings/*.npy`). Environments: `ecgfm_env` (fairseq_signals), `papagei_env`.
3. The main notebook only loads cached embeddings. This keeps it runnable in one file without dependency fights.
4. First, Claude Code should try installing fairseq_signals into `ews_main`. If it conflicts, fall back to the subprocess design and document why.

Resampling needs (verify each against the source data):
- MIMIC waveform records are typically 125 Hz. ECG-FM expects 500 Hz, so ECG needs linear-interpolation upsampling (ECG-FM's own preprocessing also resamples by linear interpolation). Upsampling adds no information. Consider a small fine-tune if the frequency mismatch hurts.
- PaPaGei expects 125 Hz PPG.

---

## 5. Data sources

| Source | What it gives | Notes |
|---|---|---|
| Own data | Best case: ECG, PPG, BP, HR plus adjudicated arrest labels | **VERIFY** everything: rates, labels, permissions |
| MIMIC-III Waveform Matched Subset (PhysioNet) | ECG, PPG, arterial BP from bedside monitors | Only about 2,825 patients have PPG + ECG lead II + ABP together (per the MIMIC-BP curation paper). Check access terms. |
| MIMIC-III-Ext-CA (PhysioNet, v1.0.0, March 2026) | Annotations of 31 PPG-captured arrest episodes from MIMIC-III | Only 31 events. Results will have wide confidence intervals. |
| MIMIC-IV Waveform (PhysioNet) | ECG, PPG, BP linked to MIMIC-IV | The version I saw (v0.1.0, 2022) had 200 records from 198 patients, with a larger release announced. **VERIFY** the current version. |
| MIMIC-IV-ECG | 12-lead diagnostic ECGs | Part of ECG-FM's pretraining, so not independent for that encoder. Possible patient overlap with MIMIC-III. |
| Nightingale NTUH arrest ECG (optional) | 12-lead ECGs, 257 arrests and 4,011 controls | Only for the optional slow layer |

Notes:
- MIMIC datasets need credentialed PhysioNet access. Claude Code must not download data itself without the user having completed that. Put credentials/paths in the config cell, never in the notebook output.
- Arrest labels in MIMIC-III must be derived and verified. Bedside monitor alarms are unreliable (the derived-dataset paper reports false alarms and probable misses).

---

## 6. Notebook structure (single file)

Every section: markdown header, config-driven code, cached intermediate outputs, a sanity-check cell that prints shapes and counts.

**Section 0. Config and environment check**
- One config cell: paths, `USE_SYNTHETIC`, random seed, horizons `[1, 6, 24]` hours, segment length 30 s, window length, split settings.
- Check that repos and weights exist. Print what is missing instead of failing silently.

**Section 1. Data access and inventory**
- Load records with `wfdb` (public) or the user's loader (own data).
- Print an inventory table: signals present, sampling rates, durations, patient counts.
- Synthetic mode: generate ECG-like, PPG-like, BP and HR streams with injected pre-arrest drift so downstream code can be tested.

**Section 2. Cohort and labels**
- Build the arrest event table (patient, timestamp, source).
- Choose controls. Sample control windows at random times across stays, not only from the last 24 h (Wav2Arrest used the last 24 h of stay, which may be a recovery period; report both variants).
- Build time-to-event labels per window.
- Exclusions: windows during or after resuscitation, windows where clinicians were already responding, poor-quality windows. Flag do-not-resuscitate or comfort-care patients if the data has it, and exclude or analyse separately.

**Section 3. Preprocessing**
- Synchronize streams on one clock, resample per encoder needs (section 4).
- Signal-quality gate per stream (start rule-based: flatline, saturation, SNR heuristics; note as a limitation).
- Cut 30 s segments. Split by patient from here onward (grouped K-fold stratified by event, plus a temporal hold-out if timestamps allow).

**Section 4. Engineered features**
- Heart rate variability, ectopy burden, BP trend and variability, pulse transit time from ECG R-peak to PPG foot.
- Aggregate to 5-minute windows.

**Section 5. Baselines**
- Logistic regression and gradient boosting on engineered features and vitals.
- Early warning score (MEWS-style) using whatever inputs are available. Say which components are missing.
- Every later model must beat these on precision and false alarms, not just AUROC.

**Section 6. Embeddings**
- ECG-FM embeddings per segment, PaPaGei/Pulse-PPG embeddings per segment. Cache to disk keyed by patient and time.
- Frozen encoders only in v1. Mean-pool embeddings into 5-minute windows.

**Section 7. Temporal model (Wav2Arrest-style)**
- Input: sequence of 5-minute window vectors for the past 6 to 24 h (ECG embedding, PPG embedding, engineered features, HR/BP), each modality projected then fused.
- Aggregator: BLSTM with attention (as in Wav2Arrest's FEAN). Also try a small GRU.
- Targets: binary "arrest within next H hours" for each horizon, plus a time-to-event regression head. Wav2Arrest reported that time-to-event regression helped, while a fuller survival formulation overfit with few positives.
- Loss: weighted BCE or focal loss for imbalance, plus TTE regression.
- v2 (only after v1 works): identity-adversarial head using gradient reversal against a patient-ID classifier, and PCGrad for multi-task conflicts. Pseudo-lab auxiliary targets are optional and need trained estimator networks.
- Regularize hard: few positives.

**Section 8. Evaluation**
- Metrics: AUROC, AUPRC, positive predictive value at fixed sensitivity, sensitivity at a fixed alert budget, alerts per patient-day, lead time distribution, calibration curve, and time-averaged AUC across lead-time bins (Wav2Arrest's metric).
- Patient-level bootstrap confidence intervals (mandatory with few events).
- Ablations: each stream removed, each model layer removed, frozen versus engineered features only.
- Replay simulation: stream each held-out patient's record through the model in time order, raise alerts by threshold with a refractory period, and count alerts, true alerts and lead time.
- Failure-case review table for a clinician.

**Section 9. Optional slow-risk layer**
- Only after sections 0 to 8 work. Use ECG-SCD code on NTUH data, or a daily-aggregated risk score. Keep separate from the short-horizon model.

**Section 10. Export**
- Save metrics, figures, config and git commit hashes of third-party repos into `outputs/`.

---

## 7. Rules for Claude Code (copy into CLAUDE.md)

1. Never commit data, credentials or model weights. Use `.gitignore` for `data/`, `cache/`, `weights/`, `third_party/`.
2. Split by patient at all times. Assert no patient appears in both train and test.
3. Fix seeds and log them. Print class counts and event counts at every split.
4. Exclude signal from resuscitation and post-arrest periods; assert this in code.
5. Cache every expensive step. The notebook must be re-runnable from any section.
6. Fail loudly with clear messages when a repo, weight file or dataset is missing.
7. Test each section on synthetic data before touching real data.
8. Do not report headline metrics without confidence intervals. With 31 events, say plainly that results are a pipeline demonstration.
9. Do not claim clinical validity anywhere in outputs or markdown cells.
10. Record versions and commit hashes of `ecg-fm`, `fairseq-signals`, `papagei`, `pulseppg`.

---

## 8. Risks and unknowns

- **Label scarcity** is the main limit. Public data has 31 annotated arrests.
- **Control selection bias** can inflate performance. Run both control schemes.
- **Distribution shift** between bedside monitors and wearables. Nothing here validates wearables.
- **Encoder mismatch**: ECG-FM was trained on 12-lead 500 Hz ECG; monitors give 1 to 3 leads at 125 Hz. Random lead masking in pretraining may help, but the paper says reduced-lead fine-tuning was not explored.
- **PaPaGei** scored well below PPG-GPT in Wav2Arrest's ICU comparison (AUROC 0.566 vs 0.736 in that paper), so expect PPG embeddings to be a weak point.
- **Environment conflicts** between fairseq_signals and the other repos.
- **Access delays** for PhysioNet credentialing and Nightingale.

## 9. Verification checklist (do before coding)

- [ ] Confirm Wav2Arrest code availability (paper statement or authors)
- [ ] Confirm current MIMIC-IV Waveform version and size
- [ ] Confirm PhysioNet access for MIMIC-III Waveform and MIMIC-III-Ext-CA
- [ ] Confirm ECG-FM and PaPaGei/Pulse-PPG weight licences
- [ ] Confirm Pulse-PPG input sampling rate
- [ ] Confirm sampling rates and lead availability in the chosen dataset
- [ ] Install fairseq_signals and run ECG-FM's `infer_quickstart.ipynb` on one example

## 10. Suggested kickoff prompt for Claude Code

> Read `cardiac_arrest_ews_plan.md`. Create the repo layout with `ews_experiment.ipynb`, a `requirements` file, `.gitignore`, and `CLAUDE.md` containing section 7. Implement notebook sections 0 to 3 with `USE_SYNTHETIC=True` first and show me the inventory and split summaries. Then stop and wait for my review before implementing embeddings.

Suggested build order: sections 0-3, then 4-5 (baselines), then 6 (embeddings), then 7-8, then 10. Review after each stage.

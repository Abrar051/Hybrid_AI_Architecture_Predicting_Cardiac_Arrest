# Cardiac Arrest Early-Warning System (EWS)

A single-notebook research pipeline that turns bedside-monitor signals (ECG, PPG, blood pressure, heart rate) into a rising-risk score for **cardiac arrest 1 / 6 / 24 hours ahead** — modeled after [Wav2Arrest 2.0](https://arxiv.org/abs/2509.21695) and extended to multimodal input.

> **⚠️ Research prototype only. Nothing here is validated for clinical use.** All results in this repository are produced on *synthetic* data and are a pipeline demonstration, not clinical evidence.

## How it works

```
monitor signals ──► foundation-model embeddings ──► temporal risk model ──► alerts
  (ECG/PPG/BP/HR)     (ECG-FM + PaPaGei)            (Wav2Arrest-style FEAN)    (threshold + refractory)
```

1. **Signals** — 5-minute windows of ECG (lead II), PPG, ABP and vitals, generated synthetically with a realistic pre-arrest drift (rising HR, falling BP/SpO₂, increasing pulse transit time).
2. **Embeddings** — two real foundation models summarize the raw waveforms: **ECG-FM** (90.9M-param wav2vec 2.0-style model, pretrained on MIMIC-IV-ECG + PhysioNet 2021) for ECG and **PaPaGei-S** (ICLR 2025) for PPG.
3. **Risk model** — a biLSTM + attention network (the paper's FEAN) watches the last 6 hours and outputs "arrest within 1/6/24 h" probabilities plus a time-to-event estimate, trained with weighted BCE + TTE regression.
4. **Alerts** — a replay simulation streams each patient in time order; an alert fires when 6-h risk ≥ 0.5, with a 60-minute refractory period.

The full design, build order, and rules are in [`cardiac_arrest_ews_plan.md`](cardiac_arrest_ews_plan.md).

## Results (synthetic demo, patient-level splits)

Discrimination on **held-out patients** (mean [95% patient-bootstrap CI]):

| Horizon | AUROC | AUPRC | Brier |
|---|---|---|---|
| 1 h  | 1.000 | 1.000 | 0.025 |
| 6 h  | 0.891 [0.825, 0.922] | 0.785 | 0.315 |
| 24 h | 0.946 [0.944, 0.948] | 0.976 | 0.137 |

- Time-averaged AUC across lead-time bins (0–1/1–6/6–24 h): **0.890**
- First true alert on the held-out case patient: **21.7 h before arrest**; 0 of the last-3-h windows missed
- False alarms on the held-out control patient: 5 over 24 h (4.9/day)
- Baselines: gradient boosting 6h AUROC 0.993; MEWS-style score (HR/SBP/SpO₂ only) sensitivity 0.82–1.00 at 8–11 alerts/day
- Ablations (val AUROC 6 h): vitals only 0.755 · vitals+engineered 0.845 · vitals+embeddings 0.816 · full 0.859

*These numbers are inflated by the synthetic drift being deterministic. Real ICU performance is expected to be much lower — in particular for PPG (see plan §8).*

## Quick start

```bash
# 1. main environment (numpy/pandas/scipy/sklearn/matplotlib/pyarrow/torch)
pip install -r requirements.txt

# 2. clone the foundation-model repos (plan rule: keep out of git)
mkdir third_party && cd third_party
git clone https://github.com/bowang-lab/ecg-fm
git clone https://github.com/Jwoo5/fairseq-signals
git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model
cd ..

# 3. download weights (~2.1 GB) into weights/
#    - ECG-FM:  https://huggingface.co/wanglab/ecg-fm  (mimic_iv_ecg_physionet_pretrained.pt)
#    - PaPaGei: https://zenodo.org/records/13983110     (papagei_s.pt)

# 4. build the two embedding envs (see CLAUDE.md for the full commands)
conda create -n ecgfm_env python=3.10 -y    # then: fairseq-signals editable install, torch>=2.1 CPU
conda create -n papagei_env python=3.10 -y  # then: papagei requirements.txt + pyPPG==1.0.41

# 5. run the notebook
jupyter notebook Pipeline.ipynb
```

Everything is cached (`cache/`, `outputs/`), and the notebook runs end-to-end with `USE_SYNTHETIC=True` — no hospital data, credentials, or downloads beyond the weights above. Sections 0–5 run even without the weights/envs.

## The alert simulator

The "output function" lives in [`scripts/ews_risk.py`](scripts/ews_risk.py):

```python
from ews_risk import RiskPipeline, simulate_patient, build_features

pipe = RiskPipeline.load("cache/models/ews_v1")   # trained alarm layer

# one 5-min window of sensor data -> risk of arrest within 1/6/24 h
out = pipe.assess(build_features({"hr": 120, "sbp": 95, "dbp": 60, "spo2": 93,
                                  "rrsd": 0.02, "pt_ms": 300, "amp_ppg": 0.6},
                                 eng_feats, ecg_embedding, ppg_embedding))

# stream a patient stay in time order; alerts fire with a 60-min refractory
for step in simulate_patient(pipe, windows):
    print(step["t_h"], step["risk"][6], step["alert"])
```

To plug in real sensor data: replace the synthetic waveform generator with your own reader (any source producing the same per-window feature row), then `RiskPipeline.assess()` unchanged.

## Repository layout

```
Pipeline.ipynb                 the whole experiment, sections 0-10 (executed outputs included)
cardiac_arrest_ews_plan.md     design document + GitHub links index + working rules
scripts/
  synth_signals.py             synthetic ECG/PPG/ABP waveform generators (shared)
  embed_windows.py             batch embedding via subprocess envs (Section 6)
  extract_ecgfm.py             ECG-FM embedding extraction (runs in ecgfm_env)
  extract_papagei.py           PaPaGei-S embedding extraction (runs in papagei_env)
  ews_risk.py                  FEAN model, RiskPipeline, alert simulator
  embed_demo.py                minimal end-to-end embedding demo
outputs/                       figures + metrics.json (from the last full run)
CLAUDE.md                      working rules + environment notes for contributors
```

## Working rules (from the plan)

1. Never commit data, credentials or model weights.
2. Split by patient at all times — no patient may appear in both train and test.
3. Fixed seeds; class/event counts printed at every split.
4. Post-arrest and resuscitation periods are excluded and asserted in code.
5. Every expensive step is cached; the notebook re-runs from any section.
6. No headline metrics without confidence intervals; synthetic results are labeled as such.
7. No clinical claims anywhere.

## Limitations & next steps

- **Synthetic data only** — next is credentialed PhysioNet data (MIMIC-III Waveform, MIMIC-IV, MIMIC-III-Ext-CA arrest annotations).
- **CPU embedding** of the full cohort takes hours; a GPU makes Section 6 cheap.
- **v2 of the risk model** (identity-adversarial head, PCGrad) after v1 validates on real data.
- Optional slow-risk layer (ECG-SCD / Nightingale NTUH) kept separate, see plan §9.
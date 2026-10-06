"""Single source of config for the synthetic pipeline (notebook + scripts).

The notebook's Section 0 imports this so every script and cell share one
cohort/split/training definition. Edit values here only.
"""
from pathlib import Path

# --- Early-warning horizons ---------------------------------------------
HORIZONS = (1, 6, 24)

# --- Signals, rates, segmentation ---------------------------------------
MONITOR_FS = 125.0              # native synthetic monitor rate (Hz), MIMIC-style
SEGMENT_LEN_S = 30              # seconds per segment fed to the foundation encoders
WINDOW_MIN = 5                  # aggregation window for features and labels (minutes)
ENCODER_FS = {"ecg": 500.0, "ppg": 125.0}   # resample targets per encoder

# --- Synthetic cohort ----------------------------------------------------
N_PATIENTS = 240                # reviewer fix: "a few hundred" synthetic patients
P_ARREST = 0.5                  # fraction of synthetic patients with an arrest event
STAY_MIN_H, STAY_MAX_H = 24.0, 96.0
ARREST_MIN_H, ARREST_MAX_H = 12.0, 72.0   # arrest time within the stay (hours)
DRIFT_H = 6.0                   # hours of pre-arrest physiological drift
EXCLUDE_BEFORE_ARREST_MIN = 15  # minutes before arrest: excluded ("clinicians already responding")
P_MISSING_PPG = 0.10            # fraction of patients without a PPG stream (different monitor)

# --- Splits ---------------------------------------------------------------
RANDOM_SEED = 42
N_FOLDS = 5
TEMPORAL_HOLDOUT_FRAC = 0.2     # latest-admitted patients -> held-out test set
VAL_FRAC = 0.2                  # of dev, stratified by case/control -> validation
MODEL_SEEDS = (42, 1, 2)        # main-model seeds; headline tables use seed 42

# --- Training -------------------------------------------------------------
SEQ_LEN_WINDOWS = 72            # 6 h of 5-min windows (FEAN history)
EPOCHS = 20
BATCH_SIZE = 256
TRAIN_STEP = 3                  # stride for training sequences (keeps CPU training fast)
ALERT = dict(alert_horizon=6, threshold=0.5, refractory_min=60)

# --- Embedding ------------------------------------------------------------
GROUP_SIZE = 2000               # windows per embed_all group (transient npz ~1.4 GB)
EMBED_ENVS = {"ecg": "ecgfm_env", "ppg": "papagei_env"}

# --- Paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
EMBED_DIR = CACHE_DIR / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
CACHE_VITALS = CACHE_DIR / "synthetic_vitals.parquet"
CACHE_EVENTS = CACHE_DIR / "synthetic_events.csv"
SPLITS_FILE = CACHE_DIR / "splits.json"
EWS_MODEL_DIR = CACHE_DIR / "models" / "ews_v2"
ABLATE_DIR = CACHE_DIR / "models" / "ablations_v2"

# --- Manifest columns (embed_windows.py + build_features read these) ------
MANIFEST_COLS = ["patient_id", "t_start_h", "t_end_h", "hr", "sbp", "dbp", "spo2",
                 "amp_ppg", "pt_ms", "rrsd", "has_ppg", "qf_ecg", "qf_ppg", "qf_abp",
                 "is_case", "arrest_h", "stay_h"]

__all__ = [k for k in list(globals()) if k.isupper()]
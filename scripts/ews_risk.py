"""Wav2Arrest-style early-warning risk model and alert simulator.

Research prototype - not validated for clinical use.

The model (FEAN per the plan): per-window feature projection -> bidirectional
LSTM -> attention pooling -> three binary heads ("arrest within 1/6/24 h")
plus a time-to-event regression head. Patient-level splits everywhere.

Public interface (the "output function" for the simulated environment):

    pipe = RiskPipeline.load("cache/models/ews_v1")   # trained model + stats
    feats = build_features(vitals_row, ecg_emb, ppg_emb)   # one 5-min window
    out = pipe.predict(history)                       # -> {"risk": {1:..,6:..,24:..},
                                                      #     "tte_h": ..}
    for step in simulate_patient(pipe, windows):      # streaming alerts
        ...                                           # step = {"t_h", "risk",
                                                      #         "tte_h", "alert"}
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

VITAL_COLS = ["hr", "sbp", "dbp", "spo2", "rrsd", "pt_ms", "amp_ppg"]
ENG_COLS = ["qf_pass", "rr_mean_s", "rr_sdnn_s", "rr_rmssd_s", "ectopy_burden",
            "abp_mean", "abp_std", "ptt_est_ms", "hr_trend", "sbp_trend"]
ECG_DIM, PPG_DIM = 768, 512
SEQ_LEN = 72          # 6 h of 5-min windows
IN_DIM = len(VITAL_COLS) + len(ENG_COLS) + ECG_DIM + PPG_DIM


def build_features(vitals, eng=None, ecg_emb=None, ppg_emb=None):
    """One 5-min window -> raw feature vector (vitals + engineered + embeddings).

    vitals: dict/Series with keys VITAL_COLS (missing values become 0).
    eng: engineered features (Section 4) with keys ENG_COLS; NaN -> 0.
    ecg_emb/ppg_emb: pooled encoder embeddings for the window (768 / 512 dims);
    zeros if unavailable (e.g. missing PPG stream).
    """
    v = np.array([float(vitals.get(c, 0.0)) for c in VITAL_COLS])
    if eng is None:
        g = np.zeros(len(ENG_COLS))
    elif hasattr(eng, "get"):            # dict or pd.Series keyed by ENG_COLS
        g = np.array([float(eng.get(c, np.nan)) for c in ENG_COLS])
    else:                                # plain row ordered as ENG_COLS
        g = np.asarray(eng, np.float64).ravel()
        if len(g) < len(ENG_COLS):
            g = np.pad(g, (0, len(ENG_COLS) - len(g)), constant_values=np.nan)
        g = g[:len(ENG_COLS)]
    g = np.nan_to_num(g, nan=0.0)
    e = np.concatenate([
        np.zeros(ECG_DIM) if ecg_emb is None else np.asarray(ecg_emb, np.float32).ravel(),
        np.zeros(PPG_DIM) if ppg_emb is None else np.asarray(ppg_emb, np.float32).ravel(),
    ])
    return np.concatenate([v, g, e]).astype(np.float32)


def mask_blocks(feats, use_eng=True, use_emb=True):
    """Zero out feature blocks for ablation experiments."""
    f = np.array(feats, np.float32, copy=True)
    if not use_eng:
        f[:, len(VITAL_COLS):len(VITAL_COLS) + len(ENG_COLS)] = 0
    if not use_emb:
        f[:, len(VITAL_COLS) + len(ENG_COLS):] = 0
    return f


class FEAN(nn.Module):
    """Wav2Arrest-style FEAN: projection -> biLSTM -> attention -> heads."""

    def __init__(self, in_dim=IN_DIM, hidden=64, num_layers=2, dropout=0.3,
                 horizons=(1, 6, 24)):
        super().__init__()
        self.horizons = list(horizons)
        self.win_proj = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.risk_heads = nn.ModuleList([nn.Linear(2 * hidden, 1) for _ in self.horizons])
        self.tte_head = nn.Linear(2 * hidden, 1)

    def forward(self, x, lens):
        """x: (B, T, IN_DIM) zero-padded; lens: (B,) true sequence lengths."""
        h = self.drop(torch.relu(self.win_proj(x)))
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lens.detach().cpu().long(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                  total_length=x.shape[1])
        w = self.attn(out)                                          # (B, T, 1)
        mask = torch.arange(x.shape[1])[None, :] < lens[:, None]
        w = w.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(w, dim=1)
        ctx = (out * w).sum(dim=1)                                  # (B, 2H)
        risks = {h: head(ctx).squeeze(-1)                           # logits; sigmoid at
                 for h, head in zip(self.horizons, self.risk_heads)}   # prediction time
        tte = torch.sigmoid(self.tte_head(ctx)).squeeze(-1)         # normalized [0,1] over 24 h
        return {"risk": risks, "tte_norm": tte}


class RiskPipeline:
    """Trained FEAN + normalization stats + alert policy."""

    def __init__(self, model, feat_mean, feat_std, alert_horizon=6,
                 threshold=0.5, refractory_min=60):
        self.model = model.eval()
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.alert_horizon = alert_horizon
        self.threshold = threshold
        self.refractory_min = refractory_min

    @classmethod
    def load(cls, path, device="cpu"):
        p = Path(path)
        ckpt = torch.load(p / "model.pt", map_location=device)
        model = FEAN()
        model.load_state_dict(ckpt["model"])
        stats = json.loads((p / "stats.json").read_text())
        return cls(model, np.array(stats["mean"]), np.array(stats["std"]),
                   alert_horizon=stats.get("alert_horizon", 6),
                   threshold=stats.get("threshold", 0.5),
                   refractory_min=stats.get("refractory_min", 60))

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, p / "model.pt")
        (p / "stats.json").write_text(json.dumps({
            "mean": self.feat_mean.tolist(), "std": self.feat_std.tolist(),
            "alert_horizon": self.alert_horizon, "threshold": self.threshold,
            "refractory_min": self.refractory_min,
        }, indent=1))

    def predict(self, history):
        """Risk for the current window given up to SEQ_LEN standardized features.

        history: (T, IN_DIM) array of standardized window features ending at
        the current window. Returns {"risk": {1: p, 6: p, 24: p}, "tte_h": t}.
        """
        history = np.asarray(history, np.float32).reshape(-1, IN_DIM)
        T = min(len(history), SEQ_LEN)
        x = np.zeros((1, SEQ_LEN, IN_DIM), dtype=np.float32)
        x[0, SEQ_LEN - T:] = history[-T:]
        lens = torch.tensor([T])
        with torch.no_grad():
            out = self.model(torch.from_numpy(x), lens)
        risks = {int(h): float(torch.sigmoid(v[0])) for h, v in out["risk"].items()}
        return {"risk": risks, "tte_h": float(out["tte_norm"][0]) * 24.0}

    def assess(self, feats, history=None):
        """Score one window of raw (unstandardized) features.

        Convenience wrapper: standardizes the current window, appends it to
        `history` (list of raw feature rows) and returns risk + alert flag.
        The caller may pass its own running history for streaming use.
        """
        if history is None:
            history = []
        history = list(history) + [np.asarray(feats, np.float32).reshape(-1)]
        h = self._standardize(np.stack(history))
        out = self.predict(h)
        risk = out["risk"][self.alert_horizon]
        out["alert"] = risk >= self.threshold
        return out

    def _standardize(self, feats):
        return (feats - self.feat_mean) / (self.feat_std + 1e-8)


def simulate_patient(pipe, windows):
    """Stream a stay in time order; yields per-window results with alert flags.

    windows: iterable of dicts {"t_h": float (hours from stay start),
    "feats": np.ndarray (IN_DIM,) raw features}. Alerts fire when risk at
    `alert_horizon` >= threshold, with a refractory period between alerts.
    """
    history = []
    last_alert_t = -np.inf
    refractory_h = pipe.refractory_min / 60.0
    for w in windows:
        t = float(w["t_h"])
        history.append(pipe._standardize(np.asarray(w["feats"], np.float32)))
        if len(history) > SEQ_LEN:
            history = history[-SEQ_LEN:]
        out = pipe.predict(np.stack(history))
        risk = out["risk"][pipe.alert_horizon]
        alert = risk >= pipe.threshold and (t - last_alert_t) >= refractory_h
        if alert:
            last_alert_t = t
        yield {"t_h": t, "risk": out["risk"], "tte_h": out["tte_h"], "alert": alert}
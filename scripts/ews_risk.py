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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

VITAL_COLS = ["hr", "sbp", "dbp", "spo2", "rrsd", "pt_ms", "amp_ppg"]
ENG_COLS = ["qf_pass", "rr_mean_s", "rr_sdnn_s", "rr_rmssd_s", "ectopy_burden",
            "abp_mean", "abp_std", "ptt_est_ms", "hr_trend", "sbp_trend"]
ECG_DIM, PPG_DIM = 768, 512
SEQ_LEN = 72          # 6 h of 5-min windows
IN_DIM = len(VITAL_COLS) + len(ENG_COLS) + ECG_DIM + PPG_DIM

# Feature blocks in the concatenated vector, in order (see build_features).
BLOCK_KEYS = ("vitals", "eng", "ecg", "ppg")
BLOCK_RANGES = {
    "vitals": (0, len(VITAL_COLS)),
    "eng": (len(VITAL_COLS), len(VITAL_COLS) + len(ENG_COLS)),
    "ecg": (len(VITAL_COLS) + len(ENG_COLS),
            len(VITAL_COLS) + len(ENG_COLS) + ECG_DIM),
    "ppg": (IN_DIM - PPG_DIM, IN_DIM),
}


def _block(x, key):
    lo, hi = BLOCK_RANGES[key]
    return x[..., lo:hi]


def build_features_masked(vitals, eng=None, ecg_emb=None, ppg_emb=None):
    """One 5-min window -> (raw feature vector, modality presence mask).

    Vector layout: vitals (7) + engineered (10) + ECG-FM (768) + PaPaGei (512).
    Missing values inside a present block become 0. Mask: (4,) float bits for
    [vitals, eng, ecg, ppg] presence at the SOURCE - the fusion layer uses the
    mask to route missing blocks to a learned per-modality missing embedding
    instead of treating zeros as data.
    """
    v = np.nan_to_num(np.array([float(vitals.get(c, 0.0)) for c in VITAL_COLS]),
                      nan=0.0) if vitals is not None else np.zeros(len(VITAL_COLS))
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
    mask = np.array([vitals is not None, eng is not None,
                     ecg_emb is not None, ppg_emb is not None], np.float32)
    return np.concatenate([v, g, e]).astype(np.float32), mask


def build_features(vitals, eng=None, ecg_emb=None, ppg_emb=None):
    """One 5-min window -> raw feature vector (vitals + engineered + embeddings).

    vitals: dict/Series with keys VITAL_COLS (missing values become 0).
    eng: engineered features (Section 4) with keys ENG_COLS; NaN -> 0.
    ecg_emb/ppg_emb: pooled encoder embeddings for the window (768 / 512 dims);
    zeros if unavailable (e.g. missing PPG stream). For an explicit
    missingness mask see build_features_masked.
    """
    return build_features_masked(vitals, eng, ecg_emb, ppg_emb)[0]


def mask_blocks(feats, use_eng=True, use_emb=True):
    """Zero out feature blocks for ablation experiments."""
    f = np.array(feats, np.float32, copy=True)
    if not use_eng:
        f[:, len(VITAL_COLS):len(VITAL_COLS) + len(ENG_COLS)] = 0
    if not use_emb:
        f[:, len(VITAL_COLS) + len(ENG_COLS):] = 0
    return f


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer (Ganin & Lempitsky): identity in the forward
    pass, gradient multiplied by -lam in the backward pass. Used by the
    identity-adversarial head (plan section 7 v2)."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class ModalityFusion(nn.Module):
    """Fuses per-modality projected blocks (B, 4, hidden) into (B, hidden).

    Modes (multimodal fusion comparison):
      gate   - hierarchical gated fusion: gate = sigmoid(W[h_a; h_b]);
               out = gate*h_a + (1-gate)*h_b, pairwise then final
      attn   - per-window modality attention: w = softmax(score(h_i));
               out = sum w_i h_i (context-dependent weighting)
      weight - static learned per-modality weights: out = sum softmax(s_i) h_i
    The concat baseline keeps the legacy single Linear projection in FEAN
    (win_proj) so v1 checkpoints load unchanged.
    """

    def __init__(self, mode, hidden=64):
        super().__init__()
        self.mode = mode
        if mode == "gate":
            self.g_ve = nn.Linear(2 * hidden, hidden)      # vitals+eng
            self.g_ep = nn.Linear(2 * hidden, hidden)      # ecg+ppg
            self.g_final = nn.Linear(2 * hidden, hidden)
        elif mode == "attn":
            self.score = nn.Linear(hidden, 1)
        elif mode == "weight":
            self.logits = nn.Parameter(torch.zeros(4))
        else:
            raise ValueError(f"unknown fusion mode {mode!r}; "
                             "choose gate | attn | weight")

    def forward(self, h):
        """h: (B, T, 4, hidden) projected blocks (or (B, 4, hidden) for a
        single window); returns (B, T, hidden)."""
        if self.mode == "gate":
            g = torch.sigmoid(self.g_ve(torch.cat([h[..., 0, :], h[..., 1, :]], -1)))
            h_ve = g * h[..., 0, :] + (1 - g) * h[..., 1, :]
            g = torch.sigmoid(self.g_ep(torch.cat([h[..., 2, :], h[..., 3, :]], -1)))
            h_ep = g * h[..., 2, :] + (1 - g) * h[..., 3, :]
            g = torch.sigmoid(self.g_final(torch.cat([h_ve, h_ep], -1)))
            return g * h_ve + (1 - g) * h_ep
        if self.mode == "attn":
            w = torch.softmax(self.score(h).squeeze(-1), dim=-2)   # (B, T, 4)
            return (h * w.unsqueeze(-1)).sum(-2)
        w = torch.softmax(self.logits, dim=0)                      # static
        return (h * w[None, None, :, None]).sum(-2)


class FEAN(nn.Module):
    """Wav2Arrest-style FEAN: projection -> biLSTM -> attention -> heads.

    fusion="concat" (default): single Linear projection of the concatenated
    vector - identical architecture and state dict to v1.
    fusion in {gate, attn, weight}: per-block projections + ModalityFusion;
    a per-modality presence mask (B, 4) routes missing blocks to learned
    missing embeddings (mask=None means all blocks present).
    n_ids > 0 adds an identity-adversarial head (a patient-ID classifier on
    the context vector behind a gradient reversal layer, scaled by
    `self.adv_lam`). With n_ids=0 and fusion="concat" the state dict is
    identical to v1.
    """

    def __init__(self, in_dim=IN_DIM, hidden=64, num_layers=2, dropout=0.3,
                 horizons=(1, 6, 24), n_ids=0, fusion="concat"):
        super().__init__()
        self.horizons = list(horizons)
        self.fusion = fusion
        if fusion == "concat":
            self.win_proj = nn.Linear(in_dim, hidden)
            self.proj, self.fuser, self.miss = None, None, None
        elif fusion in ("gate", "attn", "weight"):
            self.proj = nn.ModuleDict({
                key: nn.Linear(BLOCK_RANGES[key][1] - BLOCK_RANGES[key][0], hidden)
                for key in BLOCK_KEYS})
            self.miss = nn.Embedding(len(BLOCK_KEYS), hidden)   # missing blocks
            self.fuser = ModalityFusion(fusion, hidden)
        else:
            raise ValueError(f"unknown fusion {fusion!r}; "
                             "choose concat | gate | attn | weight")
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.risk_heads = nn.ModuleList([nn.Linear(2 * hidden, 1) for _ in self.horizons])
        self.tte_head = nn.Linear(2 * hidden, 1)
        self.n_ids = int(n_ids)
        self.id_head = nn.Linear(2 * hidden, self.n_ids) if self.n_ids > 0 else None
        self.adv_lam = 0.0            # set by the trainer (ramped per epoch)

    def _window_proj(self, x, mask):
        """Window vectors -> projected hidden vectors (B, T, hidden)."""
        if self.fusion == "concat":
            return self.drop(torch.relu(self.win_proj(x)))
        h = [self.proj[key](_block(x, key)) for key in BLOCK_KEYS]   # list of (B,T,H)
        h = torch.stack(h, dim=2)                                    # (B, T, 4, H)
        if mask is not None:            # route missing blocks to learned embeddings
            m = mask.unsqueeze(-1).to(h.dtype)                       # (B, T, 4, 1)
            miss = self.miss.weight[None, None, :, :]                # (1, 1, 4, H)
            h = h * m + miss * (1 - m)
        return self.drop(torch.relu(self.fuser(h)))                  # (B, T, H)

    def encode(self, x, lens, mask=None, return_attn=False):
        """x: (B, T, IN_DIM) zero-padded; lens: (B,) true sequence lengths;
        mask: optional (B, T, 4) modality presence bits (concat mode ignores).

        Returns the attention-pooled context (B, 2H); with return_attn also
        the attention weights (B, T). Used by forward() and by downstream
        diagnostics (identity probes, salience plots)."""
        h = self._window_proj(x, mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lens.detach().cpu().long(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                  total_length=x.shape[1])
        w = self.attn(out)                                          # (B, T, 1)
        wmask = torch.arange(x.shape[1])[None, :] < lens[:, None]
        w = w.masked_fill(~wmask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(w, dim=1)
        ctx = (out * w).sum(dim=1)                                  # (B, 2H)
        return (ctx, w.squeeze(-1)) if return_attn else ctx

    def forward(self, x, lens, mask=None, return_attn=False):
        """Risk logits + normalized TTE for each sequence (see encode())."""
        ctx, w = self.encode(x, lens, mask, return_attn=True)
        risks = {h: head(ctx).squeeze(-1)                           # logits; sigmoid at
                 for h, head in zip(self.horizons, self.risk_heads)}   # prediction time
        tte = torch.sigmoid(self.tte_head(ctx)).squeeze(-1)         # normalized [0,1] over 24 h
        res = {"risk": risks, "tte_norm": tte}
        if self.id_head is not None:
            res["id_logits"] = self.id_head(GradReverse.apply(ctx, self.adv_lam))
        if return_attn:
            res["attn"] = w                                         # (B, T)
        return res


def _flatten(params, grads):
    parts = [g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
             for p, g in zip(params, grads)]
    return torch.cat(parts)


def _unflatten(flat, params):
    out, off = [], 0
    for p in params:
        n = p.numel()
        out.append(flat[off:off + n].view_as(p))
        off += n
    return out


def pcgrad_step(tasks, optimizer):
    """PCGrad (Yu et al. 2020): compute each task's gradient, project it onto
    the normal plane of every conflicting task gradient, sum the projections
    and take one optimizer step. `tasks` are scalar losses sharing one graph.

    With one task this reduces to an ordinary gradient step.
    """
    params = [p for g in optimizer.param_groups for p in g["params"]
              if p.requires_grad]
    if len(tasks) == 1:
        optimizer.zero_grad(set_to_none=True)
        tasks[0].backward()
        optimizer.step()
        return
    grads = []
    for t in tasks:
        optimizer.zero_grad(set_to_none=True)
        t.backward(retain_graph=True)
        grads.append(_flatten(params, [p.grad for p in params]))
    for i in range(len(grads)):
        for j in range(len(grads)):
            if i == j:
                continue
            dot = float((grads[i] * grads[j]).sum())
            if dot < 0:                      # conflict: project g_i off g_j
                grads[i] = grads[i] - (dot / (grads[j] @ grads[j] + 1e-12)) * grads[j]
    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(params, _unflatten(sum(grads), params)):
        p.grad = g
    optimizer.step()


def train_fean(model, Xtr, Ltr, Ytr, Ttr=None, ids_tr=None, masks_tr=None,
               Xva=None, Lva=None, Yva=None, ids_va=None, masks_va=None,
               epochs=15, lr=2e-3, pos_weight=None, batch_size=None,
               adv_lam=0.5, adv_ramp_epochs=5, use_pcgrad=False,
               tte_weight=0.1, seed=7, verbose=1):
    """Train a FEAN model; returns (model, history).

    Tasks: weighted BCE per horizon + TTE MSE on positive windows (if Ttr is
    given, in hours) + identity-adversarial CE (if ids_tr is given and the
    model has an id head, scaled by adv_lam, ramped over adv_ramp_epochs).
    `use_pcgrad` projects conflicting task gradients per PCGrad.
    masks_tr/masks_va: (N, SEQ_LEN, 4) modality presence bits for fusion
    modes (ignored by concat fusion - passing them raises).

    history: list of per-epoch dicts with train loss and (if validation data
    is given) val loss, val 6 h AUROC, and identity accuracies.
    """
    torch.manual_seed(seed)
    if masks_tr is not None and model.fusion == "concat":
        raise ValueError("masks_tr given but fusion='concat' ignores masks")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    H = model.horizons
    X = torch.from_numpy(np.asarray(Xtr, np.float32))
    L = torch.from_numpy(np.asarray(Ltr, np.int64))
    Y = {h: torch.from_numpy(np.asarray(Ytr[h], np.float32)) for h in H}
    T = None if Ttr is None else torch.from_numpy(np.asarray(Ttr, np.float32))
    ids = None if ids_tr is None else torch.from_numpy(np.asarray(ids_tr, np.int64))
    M = None if masks_tr is None else torch.from_numpy(np.asarray(masks_tr, np.float32))
    if pos_weight is None:
        pos_weight = {h: float((Y[h] == 0).sum() / max(1, (Y[h] == 1).sum()))
                      for h in H}
    has_adv = ids is not None and model.id_head is not None
    if ids is not None and not has_adv:
        raise ValueError("ids_tr given but the model has no id head (n_ids=0)")
    Xva_t = None if Xva is None else torch.from_numpy(np.asarray(Xva, np.float32))
    Lva_t = None if Lva is None else torch.from_numpy(np.asarray(Lva, np.int64))
    Yva_t = None if Yva is None else {h: torch.from_numpy(np.asarray(Yva[h], np.float32))
                                      for h in H}
    ids_va_t = None if ids_va is None else torch.from_numpy(np.asarray(ids_va, np.int64))
    Mva = None if masks_va is None else torch.from_numpy(np.asarray(masks_va, np.float32))
    n = len(X)
    hist = []

    def run_tasks(xb, lb, yb, tb, idb, mb):
        """Forward pass -> list of task losses for one batch."""
        out = model(xb, lb, mask=mb)
        tasks = [F.binary_cross_entropy_with_logits(
            out["risk"][h], yb[h], pos_weight=torch.tensor(pos_weight[h])) for h in H]
        if tb is not None:
            pos = yb[max(H)] == 1          # TTE target only on positive windows
            if pos.any():
                tasks.append(tte_weight * F.mse_loss(
                    out["tte_norm"][pos], (tb[pos] / 24.0)))
        if idb is not None:
            tasks.append(F.cross_entropy(out["id_logits"], idb))
        return tasks, out

    for epoch in range(epochs):
        if has_adv and adv_ramp_epochs > 0:
            model.adv_lam = adv_lam * min(1.0, (epoch + 1) / adv_ramp_epochs)
        elif has_adv:
            model.adv_lam = adv_lam
        model.train()
        if batch_size is None:
            tasks, _ = run_tasks(X, L, Y, T, ids, M)
            if use_pcgrad:
                pcgrad_step(tasks, opt)
            else:
                opt.zero_grad(set_to_none=True)
                sum(tasks).backward()
                opt.step()
            tr_loss = float(sum(t.detach() for t in tasks))
        else:
            rs = np.random.default_rng(seed + epoch)
            idx = rs.permutation(n)
            tr_loss = 0.0
            for b in range(0, n, batch_size):
                ix = idx[b:b + batch_size]
                xb, lb = X[ix], L[ix]
                yb = {h: Y[h][ix] for h in H}
                tb = None if T is None else T[ix]
                idb = None if ids is None else ids[ix]
                mb = None if M is None else M[ix]
                tasks, _ = run_tasks(xb, lb, yb, tb, idb, mb)
                if use_pcgrad:
                    pcgrad_step(tasks, opt)
                else:
                    opt.zero_grad(set_to_none=True)
                    sum(tasks).backward()
                    opt.step()
                tr_loss += float(sum(t.detach() for t in tasks)) * len(ix) / n
        row = {"epoch": epoch, "train_loss": tr_loss, "adv_lam": model.adv_lam}
        if Xva_t is not None:
            model.eval()
            with torch.no_grad():
                vtasks, vout = run_tasks(Xva_t, Lva_t, Yva_t, None, ids_va_t, Mva)
                row["val_loss"] = float(sum(t.detach() for t in vtasks))
                y6 = Yva_t[6].numpy()
                p6 = torch.sigmoid(vout["risk"][6]).numpy()
                if y6.sum() > 0 and (y6 == 0).sum() > 0:
                    row["val_auc_6h"] = float(roc_auc_score(y6, p6))
                else:
                    row["val_auc_6h"] = None
                if "id_logits" in vout and ids_va_t is not None:
                    row["id_acc_va"] = float((vout["id_logits"].argmax(1) == ids_va_t)
                                             .float().mean())
            if verbose:
                print(f"  epoch {epoch:>2}: train {tr_loss:.4f} | val "
                      f"{row.get('val_loss', float('nan')):.4f} | "
                      f"val AUC6h {row.get('val_auc_6h')} | adv_lam {model.adv_lam:.2f}")
        elif verbose:
            print(f"  epoch {epoch:>2}: train {tr_loss:.4f} | adv_lam {model.adv_lam:.2f}")
        hist.append(row)
    model.eval()
    return model, hist


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
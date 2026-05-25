
from __future__ import annotations
import argparse, json, math, random, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
from scipy import signal as scipy_signal
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import load_csv_signal
from biosignal_agent.tools.peak_detectors import neurokit_nabian2018_peaks

OUT = Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH = OUT / 'ecg_apnea_rri_amp_edr_cnn_transformer_lstm_model.pt'


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def robust_norm(seq, clip=6.0):
    seq = np.asarray(seq, dtype=np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(seq))
    iqr = float(np.percentile(seq, 75) - np.percentile(seq, 25))
    if iqr < 1e-8:
        iqr = float(np.std(seq) + 1e-8)
    return np.clip((seq - med) / iqr, -clip, clip).astype(np.float32)


def safe_savgol(seq, width):
    seq = np.asarray(seq, dtype=np.float32)
    width = max(5, int(width))
    if width % 2 == 0:
        width += 1
    if len(seq) < width:
        return seq.copy()
    return scipy_signal.savgol_filter(seq, width, 2).astype(np.float32)


def interp_feature(grid, t, values, fill=0.0):
    t = np.asarray(t, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    keep = np.isfinite(t) & np.isfinite(values)
    t = t[keep]; values = values[keep]
    if len(t) >= 2:
        order = np.argsort(t)
        t = t[order]; values = values[order]
        return np.interp(grid, t, values, left=float(values[0]), right=float(values[-1])).astype(np.float32)
    if len(values) == 1:
        return np.full_like(grid, float(values[0]), dtype=np.float32)
    return np.full_like(grid, fill, dtype=np.float32)


def minute_sota_features(values, fs, seq_len=64):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    duration = len(values) / float(fs) if fs else 0.0
    grid = np.linspace(0, max(duration, 1e-6), seq_len, dtype=np.float32)
    if len(values) < max(8, int(fs * 2)):
        return np.zeros((10, seq_len), dtype=np.float32)

    x = robust_norm(values, clip=8.0)
    peaks, _ = neurokit_nabian2018_peaks(values, fs, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks > 1) & (peaks < len(values) - 2)]

    # Raw morphology branches: downsampled normalized ECG and envelopes. This is the main addition
    # over the old RR/EDR-only model and mirrors recent RRI+amplitude/raw-signal SOTA variants.
    raw_grid = np.linspace(0, max(len(x) - 1, 1), seq_len)
    raw = np.interp(raw_grid, np.arange(len(x)), x).astype(np.float32)
    abs_env = np.interp(raw_grid, np.arange(len(x)), np.abs(x)).astype(np.float32)
    try:
        band = scipy_signal.filtfilt(*scipy_signal.butter(2, [0.5 / (fs / 2), min(35.0 / (fs / 2), 0.95)], btype='band'), x).astype(np.float32)
        band = np.interp(raw_grid, np.arange(len(band)), robust_norm(band, clip=8.0)).astype(np.float32)
    except Exception:
        band = raw.copy()

    if len(peaks) >= 4:
        pt = peaks / float(fs)
        rr = np.diff(pt)
        rt = (pt[:-1] + pt[1:]) / 2.0
        keep = np.isfinite(rr) & (rr >= 0.25) & (rr <= 3.0)
        rr = rr[keep]; rt = rt[keep]
        rr_seq = interp_feature(grid, rt, rr, fill=float(np.median(rr)) if len(rr) else 0.8)
        amp = values[peaks]
        amp_seq = interp_feature(grid, pt, amp, fill=float(np.median(amp)) if len(amp) else 0.0)
        slope = np.zeros(len(peaks), dtype=np.float32)
        win = max(1, int(0.04 * fs))
        for i, p in enumerate(peaks):
            lo = max(0, p - win); hi = min(len(values), p + win + 1)
            slope[i] = float(np.max(values[lo:hi]) - np.min(values[lo:hi]))
        slope_seq = interp_feature(grid, pt, slope, fill=float(np.median(slope)) if len(slope) else 0.0)
    else:
        rr_seq = np.full(seq_len, 0.8, dtype=np.float32)
        amp_seq = np.zeros(seq_len, dtype=np.float32)
        slope_seq = np.zeros(seq_len, dtype=np.float32)

    rr_n = robust_norm(rr_seq)
    amp_n = robust_norm(amp_seq)
    slope_n = robust_norm(slope_seq)
    edr_n = amp_n
    rr_delta = robust_norm(np.gradient(rr_n))
    amp_delta = robust_norm(np.gradient(amp_n))
    slow_w = max(9, seq_len // 5)
    rr_slow = robust_norm(safe_savgol(rr_n, slow_w))
    edr_slow = robust_norm(safe_savgol(edr_n, slow_w))
    chans = [rr_n, edr_n, amp_n, slope_n, rr_delta, amp_delta, rr_slow, edr_slow, robust_norm(raw), robust_norm(band + 0.25 * abs_env)]
    return np.nan_to_num(np.stack(chans), nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)


def build_minute_features(manifest, per_minute_len):
    feats = []; y = []; groups = []; records = []; minutes = []; sources = []
    dataset = manifest.get('dataset', 'unknown')
    for i, r in enumerate(manifest['records'], start=1):
        if i % 250 == 0:
            print(f'features {i}', flush=True)
        d = load_csv_signal(r['path'], float(r['sampling_rate']), None)
        feats.append(minute_sota_features(d.values, d.sampling_rate, per_minute_len))
        y.append(1 if r['label'] == 'apnea' else 0)
        rec = str(r['record'])
        groups.append(rec); records.append(rec); minutes.append(int(r['minute']))
        sources.append(r.get('source', dataset))
    return np.asarray(feats, np.float32), np.asarray(y, np.int64), groups, records, np.asarray(minutes, np.int64), sources


def load_manifest(path, context_radius=10, per_minute_len=64):
    m = json.load(open(path))
    F, y, groups, records, minutes, sources = build_minute_features(m, per_minute_len)
    index = {(records[i], int(minutes[i])): i for i in range(len(records))}
    X = []
    for rec, minute in zip(records, minutes):
        parts = []
        for off in range(-context_radius, context_radius + 1):
            j = index.get((rec, int(minute) + off), index[(rec, int(minute))])
            parts.append(F[j])
        X.append(np.concatenate(parts, axis=1))
    return np.asarray(X, np.float32), y, groups, sources


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class CNNTransformerLSTM(nn.Module):
    def __init__(self, in_channels=10, dropout=0.35, d_model=128, nhead=4, layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 48, 11, padding=5, bias=False), nn.BatchNorm1d(48), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(48, 96, 9, padding=4, bias=False), nn.BatchNorm1d(96), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(96, d_model, 7, padding=3, bias=False), nn.BatchNorm1d(d_model), nn.GELU(), nn.MaxPool1d(2),
        )
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.rnn = nn.LSTM(d_model, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(dropout), nn.Linear(256, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 1))
    def forward(self, x):
        z = self.cnn(x).transpose(1, 2)
        z = self.transformer(self.pos(z))
        z, _ = self.rnn(z)
        pooled = torch.cat([z.mean(1), z.amax(1)], dim=1)
        return self.head(pooled).squeeze(-1)


def predict(model, X, device, batch_size=256):
    model.eval(); out = []
    with torch.no_grad():
        for xb in DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size):
            out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out).astype(float)


def train_fold(X, y, tr, va, epochs, seed, device):
    seed_all(seed)
    model = CNNTransformerLSTM(in_channels=X.shape[1]).to(device)
    pos = max(float((y[tr] == 1).sum()), 1.0); neg = max(float((y[tr] == 0).sum()), 1.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-5)
    loader = DataLoader(TensorDataset(torch.tensor(X[tr], dtype=torch.float32), torch.tensor(y[tr], dtype=torch.float32)), batch_size=160, shuffle=True)
    best = None; best_loss = math.inf; stale = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4); opt.step()
        sched.step()
        if va is None:
            continue
        model.eval()
        with torch.no_grad():
            val = float(crit(model(torch.tensor(X[va], dtype=torch.float32, device=device)), torch.tensor(y[va], dtype=torch.float32, device=device)).cpu())
        if val < best_loss - 1e-4:
            best_loss = val; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        if stale >= 9:
            break
    if best is not None:
        model.load_state_dict(best)
    return model, best_loss


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    d = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'average_precision': float(average_precision_score(y, p)),
    }
    try: d['roc_auc'] = float(roc_auc_score(y, p))
    except Exception: d['roc_auc'] = 0.0
    return d


def source_metrics(y, p, sources, threshold):
    out = {}; sources = np.asarray(sources)
    for source in sorted(set(map(str, sources))):
        mask = sources == source
        item = metrics(y[mask], p[mask], threshold)
        item['num'] = int(mask.sum()); item['positives'] = int(y[mask].sum())
        out[source] = item
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_large_manifest.json')
    ap.add_argument('--context-radius', type=int, default=10)
    ap.add_argument('--per-minute-len', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--seed', type=int, default=57)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--model-path', type=Path, default=MODEL_PATH)
    ap.add_argument('--report-path', type=Path, default=OUT / 'ecg_apnea_rri_amp_edr_cnn_transformer_lstm_report.json')
    args = ap.parse_args(); seed_all(args.seed)
    X, y, groups, sources = load_manifest(args.manifest, args.context_radius, args.per_minute_len)
    print(json.dumps({'num': len(y), 'labels': dict(Counter(map(int, y))), 'groups': len(set(groups)), 'shape': list(X.shape), 'context_minutes': 2 * args.context_radius + 1, 'device': args.device}, indent=2), flush=True)
    proba = np.zeros(len(y)); folds = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=groups)):
        model, vl = train_fold(X, y, tr, va, args.epochs, args.seed + fold, args.device)
        proba[va] = predict(model, X[va], args.device)
        rep = {'fold': fold, 'train_size': int(len(tr)), 'val_size': int(len(va)), 'val_loss': float(vl), 'val_label_counts': dict(Counter(map(int, y[va])))}
        folds.append(rep); print('fold', fold, rep, flush=True)
    threshold = float(max(((f1_score(y, proba >= t, zero_division=0), t) for t in np.linspace(0.1, 0.9, 81)), key=lambda z: z[0])[1])
    cv = metrics(y, proba, threshold); by_source = source_metrics(y, proba, sources, threshold)
    final, _ = train_fold(X, y, np.arange(len(y)), None, max(args.epochs, 50), args.seed + 999, args.device)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': final.cpu().state_dict(), 'architecture': 'CNNTransformerLSTM_RRI_AMP_EDR_RAW', 'num_channels': int(X.shape[1]), 'context_radius': args.context_radius, 'per_minute_len': args.per_minute_len, 'threshold': threshold, 'cv_metrics': cv, 'source_metrics': by_source, 'fold_reports': folds, 'label_counts': dict(Counter(map(int, y)))}, args.model_path)
    report = {'model_path': str(args.model_path), 'num_rows': int(len(y)), 'label_counts': dict(Counter(map(int, y))), 'context_radius': args.context_radius, 'per_minute_len': args.per_minute_len, 'threshold': threshold, 'cv_metrics': cv, 'source_metrics': by_source, 'fold_reports': folds}
    args.report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()

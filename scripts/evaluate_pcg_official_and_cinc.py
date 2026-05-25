
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_pcg_murmur_patient_multiloc_cnn import (  # noqa: E402
    LOCATIONS,
    PatientMultiLocCNN,
    build_patients,
    load_wav,
    resample,
    crop,
    spec_image,
    train_fold,
    set_seed,
)


def _clean(x: Any, default: str = "") -> str:
    if x is None:
        return default
    s = str(x)
    if s.lower() == "nan":
        return default
    return s


def load_circor_rows(csv_path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[str(row["Patient ID"])] = row
    return rows


def write_label_file(path: Path, patient: dict[str, Any], row: dict[str, Any]) -> None:
    pid = str(patient["patient_id"])
    records = patient["records"]
    locs = []
    for rec in records:
        loc = _clean(rec.get("location"), "AV")
        wav = Path(rec["path"]).name
        locs.append((loc, wav))
    lines = [f"{pid} {len(locs)} 4000"]
    for loc, wav in locs:
        lines.append(f"{loc} -1 {wav} -1 -1 -1")
    age = _clean(row.get("Age"), "Unknown")
    sex = _clean(row.get("Sex"), "Unknown")
    height = _clean(row.get("Height"), "nan")
    weight = _clean(row.get("Weight"), "nan")
    preg = _clean(row.get("Pregnancy status"), "False")
    murmur = _clean(row.get("Murmur"), "Unknown")
    outcome = _clean(row.get("Outcome"), "Normal")
    lines.extend([
        f"#Age: {age}",
        f"#Sex: {sex}",
        f"#Height: {height}",
        f"#Weight: {weight}",
        f"#Pregnancy status: {preg}",
        f"#Murmur: {murmur}",
        f"#Outcome: {outcome}",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_output_file(path: Path, pid: str, p_present: float, threshold: float) -> None:
    p_present = float(np.clip(p_present, 0.0, 1.0))
    # This binary model has no unknown head; expose Unknown with zero probability.
    murmur_label = "Present" if p_present >= threshold else "Absent"
    labels = [1 if murmur_label == "Present" else 0, 0, 1 if murmur_label == "Absent" else 0]
    outcome_label = "Abnormal" if p_present >= threshold else "Normal"
    out_labels = [1 if outcome_label == "Abnormal" else 0, 1 if outcome_label == "Normal" else 0]
    probs = [p_present, 0.0, 1.0 - p_present, p_present, 1.0 - p_present]
    path.write_text(
        f"#{pid}\n"
        "Present,Unknown,Absent,Abnormal,Normal\n"
        + ",".join(str(x) for x in labels + out_labels) + "\n"
        + ",".join(f"{x:.8f}" for x in probs) + "\n"
    )


def run_official_eval(label_dir: Path, output_dir: Path, scores_csv: Path) -> dict[str, Any]:
    evaluator = ROOT / "external" / "evaluation-2022" / "evaluate_model.py"
    cmd = [sys.executable, str(evaluator), str(label_dir), str(output_dir), str(scores_csv)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return {"stdout": proc.stdout, "stderr": proc.stderr, "scores_csv": str(scores_csv)}


def _coerce_score_value(value: str) -> float | str:
    try:
        return float(value)
    except Exception:
        return value


def parse_scores_csv(path: Path) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    current = None
    header = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            current = line.lstrip("#").strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            sections[current] = {}
            header = None
            continue
        parts = [x.strip() for x in line.split(",")]
        if current is None:
            continue
        if header is None:
            header = parts
            continue
        if current.endswith("per_class") or "per_class" in current:
            metric = parts[0].lower().replace("-", "_")
            sections[current][metric] = {k: _coerce_score_value(v) for k, v in zip(header[1:], parts[1:])}
        else:
            sections[current] = {k.lower().replace("-", "_").replace(" ", "_"): _coerce_score_value(v) for k, v in zip(header, parts)}
            header = None
    return sections


def run_circor_oof(args) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text())
    patients = build_patients(manifest, include_unknown=False)
    patients = [p for p in patients if p["y"] in {0, 1}]
    y = np.asarray([p["y"] for p in patients], dtype=int)
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    set_seed(args.seed)
    all_rows = []
    fold_reports = []
    for fold, (tr, te) in enumerate(cv.split(np.arange(len(patients)), y), 1):
        best = train_fold(patients, tr, te, args, args.seed + fold)
        for idx, yy, pp in zip(te, best["y"], best["prob"]):
            all_rows.append({"patient_id": patients[int(idx)]["patient_id"], "y": int(yy), "prob": float(pp), "fold": fold})
        fold_reports.append({"fold": fold, "epoch": best["epoch"], "metrics": best["metrics"], "best_threshold_metrics": best["best_threshold_metrics"]})
        print(json.dumps(fold_reports[-1]), flush=True)
    probs = np.asarray([r["prob"] for r in all_rows], dtype=float)
    yy = np.asarray([r["y"] for r in all_rows], dtype=int)
    pred = (probs >= args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    binary = {
        "true_positive": int(tp), "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn),
        "accuracy": float(accuracy_score(yy, pred)),
        "precision": float(precision_score(yy, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(yy, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(yy, pred, zero_division=0)),
        "auroc": float(roc_auc_score(yy, probs)),
        "threshold": float(args.threshold),
    }
    out_root = Path(args.out_dir)
    label_dir = out_root / "circor_official_labels"
    output_dir = out_root / "circor_oof_outputs"
    label_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_pid = load_circor_rows(Path(args.circor_csv))
    patient_by_pid = {str(p["patient_id"]): p for p in patients}
    for r in all_rows:
        pid = str(r["patient_id"])
        write_label_file(label_dir / f"{pid}.txt", patient_by_pid[pid], rows_by_pid.get(pid, {}))
        write_output_file(output_dir / f"{pid}.csv", pid, r["prob"], args.threshold)
    scores_csv = out_root / "circor_oof_official_scores.csv"
    official = run_official_eval(label_dir, output_dir, scores_csv)
    scores = parse_scores_csv(scores_csv)
    pred_path = out_root / "circor_oof_predictions.json"
    pred_path.write_text(json.dumps({"rows": all_rows, "folds": fold_reports, "binary_metrics": binary, "official_scores": scores}, indent=2))
    return {"num_patients": len(patients), "binary_metrics": binary, "official_scores": scores, "predictions": str(pred_path), "official_run": official}


class SinglePCGDataset(Dataset):
    def __init__(self, rows, args):
        self.rows = rows
        self.args = args
        self.length = int(args.target_fs * args.seconds)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        rec = self.rows[i]
        fs, values = load_wav(rec["path"])
        values = resample(values, fs, self.args.target_fs)
        rng = np.random.default_rng(1234 + i)
        values = crop(values, self.length, False, rng)
        img = spec_image(values, self.args.target_fs, self.args.freq_bins, self.args.time_bins)
        xs = np.zeros((len(LOCATIONS), 1, self.args.freq_bins, self.args.time_bins), dtype=np.float32)
        mask = np.zeros(len(LOCATIONS), dtype=np.float32)
        xs[0, 0] = img
        mask[0] = 1.0
        return torch.from_numpy(xs), torch.from_numpy(mask), torch.tensor(int(rec["y"])), rec["id"]


def run_cinc_external(args) -> dict[str, Any]:
    ref = Path(args.cinc_dir) / "REFERENCE.csv"
    rows = []
    with ref.open() as f:
        for line in f:
            if not line.strip():
                continue
            rid, lab = [x.strip() for x in line.split(",")[:2]]
            wav = Path(args.cinc_dir) / f"{rid}.wav"
            if wav.exists():
                # PhysioNet/CinC 2016 convention: -1 normal, +1 abnormal.
                rows.append({"id": rid, "path": str(wav), "y": 1 if int(lab) == 1 else 0})
    payload = torch.load(args.model, map_location="cpu")
    model = PatientMultiLocCNN(int(payload.get("embedding_dim", args.embedding_dim)))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    ds = SinglePCGDataset(rows, args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    ys, ps, ids = [], [], []
    with torch.no_grad():
        for x, mask, y, rid in loader:
            logits = model(x, mask)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            ps.extend(prob); ys.extend(y.numpy().tolist()); ids.extend(list(rid))
    y = np.asarray(ys, dtype=int); p = np.asarray(ps, dtype=float)
    pred = (p >= args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "num_records": len(rows),
        "label_counts": {"normal": int(np.sum(y == 0)), "abnormal": int(np.sum(y == 1))},
        "true_positive": int(tp), "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "threshold": float(args.threshold),
        "note": "External proxy validation: CinC 2016 labels are normal/abnormal, not murmur present/absent; the murmur probability is used as abnormal proxy.",
    }
    out = Path(args.out_dir) / "cinc2016_training_a_external_proxy_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "rows": [{"id": i, "y": int(yy), "prob": float(pp)} for i, yy, pp in zip(ids, ys, ps)]}, indent=2))
    return {"metrics": metrics, "report": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["circor-oof", "cinc-external", "both"], default="both")
    ap.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg_murmur_manifest.json")
    ap.add_argument("--circor-csv", default="/data1/jiahui/biosignal-agent/datasets/raw/circor-heart-sound/1.0.3/training_data.csv")
    ap.add_argument("--cinc-dir", default="/data1/jiahui/biosignal-agent/datasets/raw/dedicated_common/challenge-2016/training-a")
    ap.add_argument("--model", default="/data1/jiahui/biosignal-agent/outputs/pcg_murmur_patient_multiloc_cnn_e20.pt")
    ap.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/outputs/pcg_official_eval")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--target-fs", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--freq-bins", type=int, default=80)
    ap.add_argument("--time-bins", type=int, default=128)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=59)
    ap.add_argument("--threshold", type=float, default=0.38)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    result = {}
    if args.mode in {"circor-oof", "both"}:
        result["circor_oof"] = run_circor_oof(args)
    if args.mode in {"cinc-external", "both"}:
        result["cinc_external"] = run_cinc_external(args)
    summary = Path(args.out_dir) / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

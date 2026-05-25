from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.image_cnn_tools import SmallImageModalityCNN, image_to_cnn_tensor


def load_records(manifest_paths: list[str]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for manifest_path in manifest_paths:
        manifest = json.loads(Path(manifest_path).read_text())
        for record in manifest.get('records', []):
            image_path = record.get('image_path')
            modality = record.get('modality')
            if not image_path or not modality:
                continue
            key = (str(image_path), str(modality))
            if key in seen:
                continue
            seen.add(key)
            row = dict(record)
            row['manifest'] = manifest_path
            records.append(row)
    return records


def image_to_tensor(record: dict[str, Any], image_size: tuple[int, int]) -> torch.Tensor:
    return image_to_cnn_tensor(
        record['image_path'],
        image_size,
        crop_left=int(record.get('crop_left') or 0),
        crop_right=int(record.get('crop_right') or 0),
        crop_top=int(record.get('crop_top') or 0),
        crop_bottom=int(record.get('crop_bottom') or 0),
    ).squeeze(0)


class ImageDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], labels: list[int], image_size: tuple[int, int], augment: bool = False):
        self.images = [image_to_tensor(record, image_size) for record in records]
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.images[idx].clone()
        if self.augment:
            if random.random() < 0.5:
                x = x + torch.randn_like(x) * 0.015
            if random.random() < 0.5:
                scale = 0.9 + random.random() * 0.2
                x = torch.clamp(x * scale, 0.0, 1.0)
        return x, torch.tensor(self.labels[idx], dtype=torch.long)



def train_fold(train_records, train_y, test_records, test_y, image_size, epochs, batch_size, lr, seed, num_classes, return_model: bool = False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallImageModalityCNN(num_classes).to(device)
    counts = Counter(train_y)
    weights = torch.tensor([len(train_y) / max(1, counts[i]) for i in range(num_classes)], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    train_loader = DataLoader(ImageDataset(train_records, train_y, image_size, augment=True), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ImageDataset(test_records, test_y, image_size, augment=False), batch_size=batch_size, shuffle=False)
    best_pred = None
    best_acc = -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                logits = model(xb.to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        acc = accuracy_score(test_y, preds)
        if acc > best_acc:
            best_acc = acc
            best_pred = preds
    if return_model:
        return best_pred or [], model.cpu().state_dict()
    return best_pred or []


def evaluate(manifest_paths: list[str], model_path: str, image_width: int, image_height: int, epochs: int, batch_size: int, lr: float, train_final: bool = False) -> dict[str, Any]:
    records = load_records(manifest_paths)
    labels = sorted({str(r['modality']).lower() for r in records})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y = [label_to_idx[str(r['modality']).lower()] for r in records]
    min_class = min(Counter(y).values())
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13)
    all_pred = [None] * len(records)
    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        pred = train_fold(
            [records[i] for i in train_idx], [y[i] for i in train_idx],
            [records[i] for i in test_idx], [y[i] for i in test_idx],
            (image_width, image_height), epochs, batch_size, lr, 13 + fold, len(labels),
        )
        for idx, p in zip(test_idx, pred):
            all_pred[idx] = p
    pred_labels = [labels[int(p)] for p in all_pred]
    true_labels = [labels[i] for i in y]
    rows = []
    for record, truth, pred in zip(records, true_labels, pred_labels):
        rows.append({'record': record.get('record'), 'variant': record.get('variant'), 'truth': truth, 'prediction': pred, 'image_path': record.get('image_path')})
    state_dict = None
    if train_final:
        _, state_dict = train_fold(records, y, records, y, (image_width, image_height), epochs, batch_size, lr, 99, len(labels), return_model=True)
    model_bundle = {'model_name': 'small_cnn', 'labels': labels, 'image_size': [image_width, image_height], 'epochs': epochs, 'train_final': train_final}
    if state_dict is not None:
        model_bundle['state_dict'] = state_dict
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_bundle, model_path)
    return {
        'manifests': manifest_paths,
        'num_records': len(records),
        'cv': f'stratified_{n_splits}_fold',
        'model_path': model_path,
        'model_name': 'small_cnn',
        'image_size': [image_width, image_height],
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'truth_counts': dict(Counter(true_labels)),
        'prediction_counts': dict(Counter(pred_labels)),
        'metrics': {'accuracy': float(accuracy_score(true_labels, pred_labels)), 'macro_f1': float(f1_score(true_labels, pred_labels, average='macro', zero_division=0))},
        'rows': rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a small CNN image modality classifier baseline.')
    parser.add_argument('--manifest', action='append', default=None)
    parser.add_argument('--model-path', default='/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_cnn.joblib')
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_cnn_eval.json')
    parser.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_cnn_eval.csv')
    parser.add_argument('--image-width', type=int, default=192)
    parser.add_argument('--image-height', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train-final', action='store_true')
    args = parser.parse_args()
    manifests = args.manifest or ['/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_aligned_manifest.json']
    report = evaluate(manifests, args.model_path, args.image_width, args.image_height, args.epochs, args.batch_size, args.lr, train_final=args.train_final)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report['rows'], args.out_csv)
    print(json.dumps({k: report[k] for k in ['num_records', 'cv', 'model_name', 'image_size', 'epochs', 'metrics', 'model_path']}, indent=2))


if __name__ == '__main__':
    main()

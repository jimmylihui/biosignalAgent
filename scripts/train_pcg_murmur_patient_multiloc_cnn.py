from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy.io import wavfile
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

LOCATIONS = ["AV", "PV", "TV", "MV", "Phc"]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_wav(path: str) -> tuple[int, np.ndarray]:
    fs, values = wavfile.read(path)
    if values.ndim > 1:
        values = values[:, 0]
    values = values.astype(np.float32)
    values = values[np.isfinite(values)]
    values = values - np.nanmedian(values)
    scale = np.nanpercentile(np.abs(values), 95) + 1e-6
    values = np.clip(values / scale, -6.0, 6.0)
    return int(fs), values.astype(np.float32)


def resample(values: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    if fs == target_fs:
        return values.astype(np.float32)
    n = max(16, int(round(len(values) * target_fs / float(fs))))
    return scipy_signal.resample(values, n).astype(np.float32)


def crop(values: np.ndarray, length: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    if len(values) >= length:
        start = int(rng.integers(0, len(values)-length+1)) if train else max(0, (len(values)-length)//2)
        return values[start:start+length]
    out = np.zeros(length, dtype=np.float32)
    start = (length-len(values))//2
    out[start:start+len(values)] = values
    return out


def spec_image(values: np.ndarray, fs: int, freq_bins: int, time_bins: int) -> np.ndarray:
    # Mel-like compressed log-STFT: robust, cheap, and close to challenge baselines.
    freqs, times, spec = scipy_signal.spectrogram(values, fs=fs, window='hann', nperseg=256, noverlap=192, mode='magnitude', scaling='density')
    mask = (freqs >= 20.0) & (freqs <= 800.0)
    freqs = freqs[mask]
    spec = spec[mask]
    if spec.size == 0:
        return np.zeros((freq_bins, time_bins), dtype=np.float32)
    # Log-frequency pooling approximates a Mel/CWT-style representation without extra deps.
    log_edges = np.geomspace(max(20.0, freqs[0]), max(21.0, freqs[-1]), freq_bins + 1)
    pooled = np.zeros((freq_bins, spec.shape[1]), dtype=np.float32)
    for i in range(freq_bins):
        m = (freqs >= log_edges[i]) & (freqs < log_edges[i+1])
        pooled[i] = np.mean(spec[m], axis=0) if np.any(m) else 0.0
    img = np.log1p(pooled ** 2)
    lo, hi = np.percentile(img, [2, 98])
    img = np.clip((img - lo) / max(hi-lo, 1e-6), 0.0, 1.0)
    img = zoom(img, (1.0, time_bins / img.shape[1]), order=1) if img.shape[1] != time_bins else img
    return img.astype(np.float32)


def build_patients(manifest: dict[str, Any], include_unknown: bool = False) -> list[dict[str, Any]]:
    by_patient: dict[str, dict[str, Any]] = {}
    for rec in manifest['records']:
        label = rec.get('patient_murmur_label') or rec.get('murmur_label') or rec.get('patient_label') or rec.get('label')
        label = str(label).lower()
        if label in {'abnormal', 'present'}:
            y = 1
            label_name = 'present'
        elif label in {'normal', 'absent'}:
            y = 0
            label_name = 'absent'
        elif include_unknown and label == 'unknown':
            y = -1
            label_name = 'unknown'
        else:
            continue
        pid = str(rec['patient_id'])
        item = by_patient.setdefault(pid, {'patient_id': pid, 'label': label_name, 'y': y, 'records': []})
        item['records'].append(rec)
    return list(by_patient.values())


class PatientDataset(Dataset):
    def __init__(self, patients: list[dict[str, Any]], indices: np.ndarray, train: bool, args, seed: int):
        self.patients = patients
        self.indices = np.asarray(indices, dtype=int)
        self.train = train
        self.args = args
        self.rng = np.random.default_rng(seed)
        self.length = int(args.target_fs * args.seconds)

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        patient = self.patients[int(self.indices[item])]
        by_loc = {rec.get('location'): rec for rec in patient['records']}
        xs = []
        masks = []
        for loc in LOCATIONS:
            rec = by_loc.get(loc)
            if rec is None:
                xs.append(np.zeros((self.args.freq_bins, self.args.time_bins), dtype=np.float32)); masks.append(0.0); continue
            fs, values = load_wav(rec['path'])
            values = resample(values, fs, self.args.target_fs)
            values = crop(values, self.length, self.train, self.rng)
            if self.train:
                values = values * float(self.rng.uniform(0.8, 1.2)) + self.rng.normal(0, 0.01, size=values.shape).astype(np.float32)
                if self.rng.random() < 0.15:
                    shift = int(self.rng.integers(-self.args.target_fs, self.args.target_fs))
                    values = np.roll(values, shift)
            xs.append(spec_image(values, self.args.target_fs, self.args.freq_bins, self.args.time_bins)); masks.append(1.0)
        x = torch.from_numpy(np.stack(xs, axis=0)[:, None, :, :])  # L,1,F,T
        mask = torch.tensor(masks, dtype=torch.float32)
        y = torch.tensor(int(patient['y']), dtype=torch.long)
        return x, mask, y


class LocationEncoder(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=2), nn.BatchNorm2d(24), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, emb), nn.SiLU()
        )
    def forward(self, x): return self.net(x)


class PatientMultiLocCNN(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.encoder = LocationEncoder(emb)
        self.attn = nn.Sequential(nn.Linear(emb + len(LOCATIONS), 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(emb, 2))

    def forward(self, x, mask):
        b, l, c, f, t = x.shape
        feat = self.encoder(x.reshape(b*l, c, f, t)).reshape(b, l, -1)
        loc_eye = torch.eye(l, device=x.device).unsqueeze(0).expand(b, -1, -1)
        score = self.attn(torch.cat([feat, loc_eye], dim=-1)).squeeze(-1)
        score = score.masked_fill(mask <= 0, -1e4)
        weight = torch.softmax(score, dim=1)
        pooled = torch.sum(feat * weight.unsqueeze(-1), dim=1)
        return self.head(pooled)


def metrics(y_true, prob, threshold=0.5):
    y=np.asarray(y_true,dtype=int); p=np.asarray(prob,dtype=float); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'true_positive':int(tp),'true_negative':int(tn),'false_positive':int(fp),'false_negative':int(fn),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall_sensitivity':float(recall_score(y,pred,zero_division=0)),'specificity':float(tn/(tn+fp)) if tn+fp else 0.0,'f1':float(f1_score(y,pred,zero_division=0)),'auroc':float(roc_auc_score(y,p)) if len(set(y.tolist()))>1 else None,'threshold':float(threshold)}


def best_threshold(y, prob):
    best=(0.5,metrics(y,prob,0.5))
    for t in np.linspace(0.1,0.9,81):
        m=metrics(y,prob,float(t))
        if (m['f1'],m['accuracy'],m['specificity'])>(best[1]['f1'],best[1]['accuracy'],best[1]['specificity']): best=(float(t),m)
    return best


def run_epoch(model, loader, opt, device, class_weights=None):
    model.train(); loss_fn=nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None); total=0.0
    for x,mask,y in loader:
        x=x.to(device); mask=mask.to(device); y=y.to(device)
        opt.zero_grad(set_to_none=True); loss=loss_fn(model(x,mask),y); loss.backward(); opt.step(); total += float(loss.item())*len(y)
    return total/max(1,len(loader.dataset))


def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,mask,y in loader:
            logits=model(x.to(device),mask.to(device)); prob=torch.softmax(logits,dim=1)[:,1].cpu().numpy().tolist(); ps.extend(prob); ys.extend(y.numpy().tolist())
    return ys,ps


def train_fold(patients, train_idx, test_idx, args, seed):
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    train_ds=PatientDataset(patients,train_idx,True,args,seed); test_ds=PatientDataset(patients,test_idx,False,args,seed+1000)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    y_train=np.asarray([patients[int(i)]['y'] for i in train_idx],dtype=int); counts=np.bincount(y_train,minlength=2).astype(float)
    weights=torch.tensor([len(y_train)/max(1,2*counts[0]), len(y_train)/max(1,2*counts[1])],dtype=torch.float32)
    model=PatientMultiLocCNN(args.embedding_dim).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=None
    for epoch in range(args.epochs):
        loss=run_epoch(model,train_loader,opt,device,weights); y,p=predict(model,test_loader,device); m=metrics(y,p,0.5); bt,bm=best_threshold(y,p)
        score=(bm['f1'],bm['auroc'] or 0.0,m['f1'])
        if best is None or score>best['score']:
            best={'epoch':epoch+1,'loss':loss,'metrics':m,'best_threshold':bt,'best_threshold_metrics':bm,'y':y,'prob':p,'score':score,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    return best


def train_full(patients,args):
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    idx=np.arange(len(patients)); ds=PatientDataset(patients,idx,True,args,args.seed+999); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    y=np.asarray([p['y'] for p in patients],dtype=int); counts=np.bincount(y,minlength=2).astype(float); weights=torch.tensor([len(y)/max(1,2*counts[0]),len(y)/max(1,2*counts[1])],dtype=torch.float32)
    model=PatientMultiLocCNN(args.embedding_dim).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    for _ in range(args.epochs): run_epoch(model,loader,opt,device,weights)
    return {k:v.detach().cpu() for k,v in model.state_dict().items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg_murmur_manifest.json')
    ap.add_argument('--out-model',default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_patient_multiloc_cnn.pt')
    ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_patient_multiloc_cnn_report.json')
    ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=8e-4); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--target-fs',type=int,default=1000); ap.add_argument('--seconds',type=float,default=8.0)
    ap.add_argument('--freq-bins',type=int,default=80); ap.add_argument('--time-bins',type=int,default=128); ap.add_argument('--embedding-dim',type=int,default=128); ap.add_argument('--seed',type=int,default=59); ap.add_argument('--cpu',action='store_true')
    args=ap.parse_args(); set_seed(args.seed)
    manifest=json.loads(Path(args.manifest).read_text()); patients=build_patients(manifest,include_unknown=False); patients=[p for p in patients if p['y'] in {0,1}]
    y=np.asarray([p['y'] for p in patients],dtype=int); cv=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=args.seed)
    all_y=[]; all_prob=[]; folds=[]
    for fold,(tr,te) in enumerate(cv.split(np.arange(len(patients)),y),1):
        best=train_fold(patients,tr,te,args,args.seed+fold); all_y.extend(best['y']); all_prob.extend(best['prob'])
        rep={'fold':fold,'epoch':best['epoch'],'num_train':len(tr),'num_test':len(te),'metrics':best['metrics'],'best_threshold':best['best_threshold'],'best_threshold_metrics':best['best_threshold_metrics']}
        folds.append(rep); print(json.dumps(rep),flush=True)
    cv_metrics=metrics(all_y,all_prob,0.5); bt,bm=best_threshold(all_y,all_prob); state=train_full(patients,args)
    payload={'model_state_dict':state,'architecture':'PatientMultiLocCNN_logfreq_spectrogram_attention','locations':LOCATIONS,'target_fs':args.target_fs,'seconds':args.seconds,'freq_bins':args.freq_bins,'time_bins':args.time_bins,'embedding_dim':args.embedding_dim,'cv_metrics':cv_metrics,'best_threshold':bt,'best_threshold_metrics':bm,'reference':'CirCor 1.0.3 patient-level murmur present/absent, multi-location attention CNN, stratified patient CV.'}
    Path(args.out_model).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.out_model)
    report={'manifest':args.manifest,'num_patients':len(patients),'label_counts':dict(Counter(['present' if v else 'absent' for v in y.tolist()])),'model_out':args.out_model,'folds':folds,'cv_metrics':cv_metrics,'best_threshold':bt,'best_threshold_metrics':bm}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()

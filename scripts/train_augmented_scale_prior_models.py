from __future__ import annotations
import json, sys, os
from pathlib import Path
from collections import defaultdict, Counter
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/data1/jiahui/biosignal-agent/code/biosignal-agent')
from scripts.train_image_scale_prior import DURATIONS
from scripts.train_image_scale_prior_features import build_features

OUT = Path('/data1/jiahui/biosignal-agent/outputs/image_scale_prior_aug')
ROWS = OUT / 'scale_prior_dataset.json'
rows=json.loads(ROWS.read_text())

MODELS = {
  'extra_trees': lambda: ExtraTreesClassifier(n_estimators=350, random_state=31, class_weight='balanced', min_samples_leaf=2, max_features='sqrt', n_jobs=-1),
  'random_forest': lambda: RandomForestClassifier(n_estimators=250, random_state=31, class_weight='balanced', min_samples_leaf=2, max_features='sqrt', n_jobs=-1),
  'logistic': lambda: Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=1.0))]),
}

def fit_select(X,y,seed=31):
    labels=sorted(set(int(v) for v in y))
    min_count=min(Counter(int(v) for v in y).values())
    stratify=y if min_count>=2 else None
    test_size=max(0.25, len(labels)/max(1,len(y)))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed,stratify=stratify)
    results={}; best_name=None; best_score=-1.0
    for name,maker in MODELS.items():
        model=maker(); model.fit(Xtr,ytr)
        pred=model.predict(Xte); proba=model.predict_proba(Xte)
        classes=list(getattr(model,'classes_',labels)); k=min(3,len(classes))
        acc=accuracy_score(yte,pred)
        try: top3=top_k_accuracy_score(yte,proba,k=k,labels=classes)
        except Exception: top3=acc
        results[name]={'accuracy':float(acc),'top3_accuracy':float(top3),'classes':[int(c) for c in classes], 'confusion_matrix':confusion_matrix(yte,pred,labels=classes).tolist()}
        score=top3 + 0.1*acc
        if score>best_score: best_score=score; best_name=name
    best=MODELS[best_name](); best.fit(X,y)
    return best,best_name,results,labels

report={'durations':DURATIONS,'num_rows':len(rows)}
# unified
X,y,names=build_features(rows)
best,best_name,results,labels=fit_select(X,y,31)
unified_path=OUT/'image_scale_prior_feature_model_aug.joblib'
joblib.dump({'model':best,'feature_names':names,'durations':DURATIONS,'results':results,'best_model':best_name}, unified_path)
report['unified']={'best_model':best_name,'results':results,'model_path':str(unified_path),'num_rows':int(len(y))}
print('unified done', best_name, flush=True)
# per modality
by=defaultdict(list)
for r in rows: by[str(r.get('modality')).lower()].append(r)
per_models={}; per_report={}
for mod,mod_rows in sorted(by.items()):
    X,y,names=build_features(mod_rows)
    counts=Counter(int(v) for v in y)
    entry={'num_rows':int(len(y)),'class_counts':dict(counts)}
    if len(counts)<2 or min(counts.values())<2:
        entry['trained']=False; entry['skip_reason']='too_few_samples_or_classes'; per_report[mod]=entry; continue
    best,best_name,results,labels=fit_select(X,y,37)
    per_models[mod]={'model':best,'feature_names':names,'durations':DURATIONS,'classes':labels,'num_rows':int(len(y)),'best_model':best_name}
    entry.update({'trained':True,'best_model':best_name,'results':results})
    per_report[mod]=entry
    print('per modality done', mod, best_name, flush=True)
per_path=OUT/'image_scale_prior_per_modality_models.joblib'
joblib.dump({'models':per_models,'durations':DURATIONS,'feature_schema':'trace_spectral_v2'}, per_path)
report['per_modality']={'model_path':str(per_path),'modalities':per_report}
(OUT/'image_scale_prior_aug_train_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report, indent=2)[:16000])

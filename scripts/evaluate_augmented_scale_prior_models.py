from __future__ import annotations
import json, sys, os
from pathlib import Path
from collections import defaultdict, Counter
import joblib
import numpy as np
sys.path.insert(0, '/data1/jiahui/biosignal-agent/code/biosignal-agent')
from scripts.train_image_scale_prior_features import build_features

AUG=Path('/data1/jiahui/biosignal-agent/outputs/image_scale_prior_aug')
unified=joblib.load(AUG/'image_scale_prior_feature_model_aug.joblib')
per=joblib.load(AUG/'image_scale_prior_per_modality_models.joblib')
old=joblib.load('/data1/jiahui/biosignal-agent/outputs/image_scale_prior/image_scale_prior_feature_model.joblib')

def predict_bundle(bundle, image_path, modality=None, top_k=3):
    row={'image_path':image_path,'modality':modality or 'unknown','label':0}
    X,_,_=build_features([row])
    names=bundle['feature_names']
    # build_features order should match when names equal; if not, rebuild by name through tool not available here.
    # Since all current bundles use the same build_features sorted schema, use X directly when length matches.
    if X.shape[1] != len(names):
        raise ValueError((X.shape, len(names)))
    model=bundle['model']; proba=np.asarray(model.predict_proba(X)[0],float); classes=[int(c) for c in getattr(model,'classes_',range(len(proba)))]
    durs=[float(v) for v in bundle['durations']]
    out=[]
    for idx in np.argsort(proba)[::-1][:top_k]:
        li=classes[int(idx)] if int(idx)<len(classes) else int(idx)
        if 0 <= li < len(durs): out.append(durs[li])
    return out, float(np.max(proba))

def predict_per(image_path, modality, top_k=3):
    m=per['models'].get(str(modality).lower())
    if not m: return predict_bundle(unified, image_path, modality, top_k)
    return predict_bundle(m, image_path, modality, top_k)

def eval_manifest(mp):
    data=json.load(open(mp)); rows=[]
    for r in data.get('records',[]):
        img=r.get('image_path')
        if not img or not os.path.exists(img): continue
        d=float(r.get('duration_s')); mod=r.get('modality')
        preds={}
        for name,fn in [('old_unified',lambda:predict_bundle(old,img,mod)),('aug_unified',lambda:predict_bundle(unified,img,mod)),('aug_per_modality',lambda:predict_per(img,mod))]:
            c,conf=fn(); preds[name]={'candidates':c,'confidence':conf,'top1':bool(c) and abs(c[0]-d)<0.2,'top3':any(abs(x-d)<0.2 for x in c)}
        rows.append({'modality':mod,'duration_s':d,'predictions':preds})
    summary={}
    for name in ['old_unified','aug_unified','aug_per_modality']:
        summary[name]={
          'top1_accuracy':sum(x['predictions'][name]['top1'] for x in rows)/max(1,len(rows)),
          'top3_accuracy':sum(x['predictions'][name]['top3'] for x in rows)/max(1,len(rows)),
          'mean_confidence':sum(x['predictions'][name]['confidence'] for x in rows)/max(1,len(rows)),
        }
        by=defaultdict(lambda:{'n':0,'top3_hits':0})
        for x in rows:
            by[x['modality']]['n']+=1; by[x['modality']]['top3_hits']+=int(x['predictions'][name]['top3'])
        summary[name]['by_modality']=dict(by)
    return {'manifest':mp,'n':len(rows),'duration_counts':dict(Counter(x['duration_s'] for x in rows)),'summary':summary,'examples':rows[:8]}

manifests=['/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json','/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_longresp_30s_manifest.json','/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_manifest.json']
report={os.path.basename(mp):eval_manifest(mp) for mp in manifests}
out=AUG/'image_scale_prior_aug_benchmark_eval.json'
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report, indent=2)[:12000])
print('saved', out)

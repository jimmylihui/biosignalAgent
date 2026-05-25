from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_ecg_rhythm_feature_classifier import feature_row

OUT = Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH = OUT / 'ecg_af_feature_classifier.joblib'

def metrics(y, proba, threshold):
    pred=(proba>=threshold).astype(int)
    out={
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y,pred)),
        'precision': float(precision_score(y,pred,zero_division=0)),
        'recall': float(recall_score(y,pred,zero_division=0)),
        'f1': float(f1_score(y,pred,zero_division=0)),
        'average_precision': float(average_precision_score(y,proba)) if len(set(y))>1 else None,
        'roc_auc': float(roc_auc_score(y,proba)) if len(set(y))>1 else None,
        'class_report': classification_report(y,pred,labels=[0,1],target_names=['non_af','af'],zero_division=0,output_dict=True),
    }
    return out

def group_metrics(y, proba, groups, threshold):
    rows=[]
    for g in sorted(set(groups)):
        idx=np.asarray([i for i,x in enumerate(groups) if x==g])
        if len(set(y[idx]))<1: continue
        rows.append({'group':str(g),'n':int(len(idx)),'positives':int(np.sum(y[idx])),'metrics':metrics(y[idx],proba[idx],threshold)})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/ecg_rhythm_beat_full_plus_afdb_3rec_manifest.json')
    ap.add_argument('--model-path',type=Path,default=MODEL_PATH)
    ap.add_argument('--report-path',type=Path,default=OUT/'ecg_af_feature_classifier_train_report.json')
    args=ap.parse_args()
    manifest=json.load(open(args.manifest)); records=manifest['records']
    rows=[]; y=[]; groups=[]
    for i,r in enumerate(records,1):
        if i%100==0: print('features',i,flush=True)
        rows.append(feature_row(r)); y.append(1 if r['coarse_rhythm_label']=='af' else 0); groups.append(str(r['record']))
    feature_names=list(rows[0].keys())
    X=np.asarray([[row[k] for k in feature_names] for row in rows],dtype=float); y=np.asarray(y,dtype=int); groups=np.asarray(groups)
    candidates={
      'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,class_weight='balanced')),
      'extra_trees': ExtraTreesClassifier(n_estimators=600,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',random_state=31,n_jobs=-1),
      'random_forest': RandomForestClassifier(n_estimators=450,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',random_state=37,n_jobs=-1),
      'hgb': HistGradientBoostingClassifier(max_iter=300,learning_rate=0.04,l2_regularization=0.05,random_state=41),
    }
    splits=list(GroupKFold(n_splits=5).split(X,y,groups=groups))
    reports={}; best_name=None; best_score=-1; best_proba=None; best_thr=0.5
    for name,model in candidates.items():
        proba=np.zeros(len(y),dtype=float)
        for tr,va in splits:
            m=clone(model); m.fit(X[tr],y[tr])
            if hasattr(m,'predict_proba'):
                proba[va]=m.predict_proba(X[va])[:,1]
            else:
                proba[va]=m.predict(X[va])
        best=max(((f1_score(y,proba>=t,zero_division=0),t) for t in np.linspace(0.05,0.95,91)), key=lambda z:z[0])
        rep=metrics(y,proba,best[1]); rep['group_metrics']=group_metrics(y,proba,groups,best[1])
        reports[name]=rep
        print(name,json.dumps({k:rep[k] for k in ['threshold','accuracy','precision','recall','f1','average_precision','roc_auc']},indent=2),flush=True)
        score=rep['f1'] + 0.15*(rep['average_precision'] or 0)
        if score>best_score:
            best_score=score; best_name=name; best_proba=proba; best_thr=best[1]
    final=clone(candidates[best_name]); final.fit(X,y)
    bundle={'model':final,'feature_names':feature_names,'classes':[0,1],'positive_label':'af','best_model':best_name,'threshold':float(best_thr),'cv_metrics':reports[best_name],'all_model_reports':reports,'label_counts':dict(Counter(map(int,y))),'manifest':args.manifest}
    args.model_path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,args.model_path)
    report={'model_path':str(args.model_path),**{k:bundle[k] for k in ['best_model','threshold','cv_metrics','all_model_reports','label_counts','manifest']}}
    args.report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()

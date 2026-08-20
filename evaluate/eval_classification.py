import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score
from scipy.stats import norm

def wilson_ci(p,n,conf=0.95):
    if n==0:
        return (0.0,1.0)
    z = norm.ppf((1+conf)/2)
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (centre‑margin, centre+margin)

def calc_all_metrics(y_true,y_pred,y_score):
    auc = roc_auc_score(y_true,y_score)
    cm = confusion_matrix(y_true,y_pred)
    tn,fp,fn,tp = cm.ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    ppv = tp/(tp+fp) if (tp+fp)>0 else 0
    npv = tn/(tn+fn) if (tn+fn)>0 else 0
    f1 = f1_score(y_true,y_pred)
    ci_ppv = wilson_ci(ppv, tp+fp)
    ci_npv = wilson_ci(npv, tn+fn)
    return {
        "accuracy":acc,
        "sensitivity":sens,
        "specificity":spec,
        "auc":auc,
        "ppv":ppv,
        "ppv_95ci_low":ci_ppv[0],
        "ppv_95ci_high":ci_ppv[1],
        "npv":npv,
        "npv_95ci_low":ci_npv[0],
        "npv_95ci_high":ci_npv[1],
        "f1_score":f1,
        "confusion_matrix":cm.tolist()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv",type=str,required=True)
    parser.add_argument("--gt_csv",type=str,required=True)
    args = parser.parse_args()
    df_pred = pd.read_csv(args.pred_csv)
    df_gt = pd.read_csv(args.gt_csv)
    y_true = df_gt["label"].values
    y_pred = df_pred["pred"].values
    y_score = df_pred["score_ar"].values
    res = calc_all_metrics(y_true,y_pred,y_score)
    import json
    print(json.dumps(res,indent=2))

if __name__ == "__main__":
    main()
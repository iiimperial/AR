import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score
from scipy.stats import norm

def wilson_ci(p,n,conf=0.95):
    """Wilson binomial confidence interval for PPV/NPV"""
    if n==0:return (0,1)
    z=norm.ppf((1+conf)/2)
    denominator = 1+z*z/n
    centre = (p + z*z/(2*n)) / denominator
    margin = z * np.sqrt(p*(1‑p)/n + z*z/(4*n*n)) / denominator
    return (centre‑margin, centre+margin)

def calc_metrics(y_true,y_pred,y_score):
    auc=roc_auc_score(y_true,y_score)
    cm=confusion_matrix(y_true,y_pred)
    tn,fp,fn,tp = cm.ravel()
    acc=(tp+tn)/(tp+tn+fp+fn)
    sens=tp/(tp+fn)
    spec=tn/(tn+fp)
    ppv=tp/(tp+fp) if (tp+fp)>0 else 0
    npv=tn/(tn+fn) if (tn+fn)>0 else 0
    f1 = f1_score(y_true,y_pred)
    ci_ppv = wilson_ci(ppv, tp+fp)
    ci_npv = wilson_ci(npv, tn+fn)
    return {
        "acc":acc,"sensitivity":sens,"specificity":spec,"auc":auc,
        "ppv":ppv,"ppv_95ci":ci_ppv,"npv":npv,"npv_95ci":ci_npv,"f1":f1,
        "confusion_matrix":cm
    }
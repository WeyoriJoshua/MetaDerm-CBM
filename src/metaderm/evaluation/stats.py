from __future__ import annotations
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def bootstrap_metric(y_true, prob, metric_fn, n_boot=10000, seed=2026):
    y=np.asarray(y_true); p=np.asarray(prob); rng=np.random.default_rng(seed); vals=[]; n=len(y)
    for _ in range(n_boot):
        idx=rng.integers(0,n,size=n)
        try:
            value = float(metric_fn(y[idx], p[idx]))
            if np.isfinite(value):
                vals.append(value)
        except Exception:
            pass
    vals=np.asarray(vals,float)
    estimate=float(metric_fn(y,p))
    if len(vals) == 0:
        return {"estimate":estimate,"ci_low":float("nan"),"ci_high":float("nan"),"valid_bootstraps":0}
    return {"estimate":estimate,"ci_low":float(np.quantile(vals,0.025)),"ci_high":float(np.quantile(vals,0.975)),"valid_bootstraps":int(len(vals))}


def bootstrap_auc(y_true, prob, n_boot=10000, seed=2026):
    return bootstrap_metric(y_true, prob, roc_auc_score, n_boot, seed)


def paired_bootstrap_difference(y_true, prob_a, prob_b, metric_fn, n_boot=10000, seed=2026):
    y=np.asarray(y_true); a=np.asarray(prob_a); b=np.asarray(prob_b); rng=np.random.default_rng(seed); diffs=[]; n=len(y)
    for _ in range(n_boot):
        idx=rng.integers(0,n,size=n)
        try:
            diff=float(metric_fn(y[idx],a[idx])-metric_fn(y[idx],b[idx]))
            if np.isfinite(diff):
                diffs.append(diff)
        except Exception:
            pass
    diffs=np.asarray(diffs,float)
    est=float(metric_fn(y,a)-metric_fn(y,b))
    if len(diffs) == 0:
        return {"difference":est,"ci_low":float("nan"),"ci_high":float("nan"),"bootstrap_p":float("nan"),"valid_bootstraps":0}
    p=2*min((diffs<=0).mean(),(diffs>=0).mean())
    return {"difference":est,"ci_low":float(np.quantile(diffs,0.025)),"ci_high":float(np.quantile(diffs,0.975)),"bootstrap_p":float(min(p,1.0)),"valid_bootstraps":int(len(diffs))}


def mcnemar_exact(y_true, pred_a, pred_b):
    y=np.asarray(y_true); a=np.asarray(pred_a); b=np.asarray(pred_b)
    ca=(a==y); cb=(b==y)
    n01=int((~ca & cb).sum()); n10=int((ca & ~cb).sum())
    n=n01+n10
    p=1.0 if n==0 else binomtest(min(n01,n10), n=n, p=0.5, alternative="two-sided").pvalue
    return {"a_wrong_b_right":n01,"a_right_b_wrong":n10,"p_value":float(p)}


def holm_adjust(pvalues):
    p=np.asarray(pvalues,float); m=len(p); order=np.argsort(p); adjusted=np.empty(m,float); running=0.0
    for rank, idx in enumerate(order):
        val=(m-rank)*p[idx]; running=max(running,val); adjusted[idx]=min(running,1.0)
    return adjusted

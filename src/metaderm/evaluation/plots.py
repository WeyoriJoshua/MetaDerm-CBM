from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_risk_coverage(curve, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,4.5)); ax.plot(curve[:,0],curve[:,1],linewidth=2)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (error rate)"); ax.set_title("Risk–Coverage Curve"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)


def plot_reliability_hist(reliability, y_true, y_pred, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    correct=np.asarray(y_true)==np.asarray(y_pred); r=np.asarray(reliability)
    fig,ax=plt.subplots(figsize=(6,4.5)); ax.hist(r[correct],bins=15,alpha=.65,label="Correct"); ax.hist(r[~correct],bins=15,alpha=.65,label="Incorrect")
    ax.set_xlabel("Metacognitive reliability"); ax.set_ylabel("Cases"); ax.legend(); ax.set_title("Reliability by prediction correctness")
    fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

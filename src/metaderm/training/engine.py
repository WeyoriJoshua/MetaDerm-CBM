from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from ..constants import CONCEPT_COLUMNS
from ..models.losses import concept_loss
from ..utils.io import save_checkpoint


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_concepts(model, train_loader, valid_loader, cfg, class_weights, device, seed):
    tcfg=cfg["training"]; opt=AdamW(model.parameters(),lr=float(tcfg["learning_rate"]),weight_decay=float(tcfg["weight_decay"])); sched=CosineAnnealingLR(opt,T_max=int(tcfg["epochs"]));
    scaler=torch.amp.GradScaler("cuda", enabled=bool(tcfg.get("amp",True) and device.type=="cuda"))
    best=-np.inf; bad=0; history=[]; ckpt=Path(cfg["project"]["checkpoint_root"])/f"concepts_seed{seed}_best.pt"
    for epoch in range(1,int(tcfg["epochs"])+1):
        model.train(); losses=[]
        for batch in tqdm(train_loader,desc=f"concept train {epoch}",leave=False):
            x=batch["image"].to(device); y={c:batch["concepts"][c].to(device) for c in CONCEPT_COLUMNS}; opt.zero_grad(set_to_none=True)
            with autocast_context(device, bool(tcfg.get("amp",True))):
                logits=model(x)["logits"]; loss,_=concept_loss(logits,y,class_weights,float(tcfg.get("label_smoothing",0.0)))
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),float(tcfg.get("gradient_clip_norm",1.0))); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        sched.step()
        metrics=evaluate_concepts(model,valid_loader,device); metrics["train_loss"]=float(np.mean(losses)); metrics["epoch"]=epoch; history.append(metrics)
        score=metrics["mean_macro_f1"]
        if score>best:
            best=score; bad=0; save_checkpoint(ckpt,model,opt,epoch,metrics,cfg)
        else: bad+=1
        print(f"epoch={epoch} train_loss={metrics['train_loss']:.4f} val_f1={score:.4f}")
        if bad>=int(tcfg["patience"]): break
    return history, ckpt


def evaluate_concepts(model,loader,device):
    model.eval(); yt={c:[] for c in CONCEPT_COLUMNS}; yp={c:[] for c in CONCEPT_COLUMNS}; losses=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["image"].to(device); logits=model(x)["logits"]
            for c in CONCEPT_COLUMNS:
                yt[c].extend(batch["concepts"][c].numpy().tolist()); yp[c].extend(logits[c].argmax(1).cpu().numpy().tolist())
    scores={c:f1_score(yt[c],yp[c],average="macro",zero_division=0) for c in CONCEPT_COLUMNS}
    return {"mean_macro_f1":float(np.mean(list(scores.values()))),**{f"f1_{c}":float(v) for c,v in scores.items()}}


def train_blackbox(model,train_loader,valid_loader,cfg,pos_weight,device,seed):
    tcfg=cfg["training"]; opt=AdamW(model.parameters(),lr=float(tcfg["learning_rate"]),weight_decay=float(tcfg["weight_decay"])); sched=CosineAnnealingLR(opt,T_max=int(tcfg["epochs"])); scaler=torch.amp.GradScaler("cuda", enabled=bool(tcfg.get("amp",True) and device.type=="cuda")); criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    best=-np.inf; bad=0; history=[]; ckpt=Path(cfg["project"]["checkpoint_root"])/f"blackbox_seed{seed}_best.pt"
    for epoch in range(1,int(tcfg["epochs"])+1):
        model.train(); losses=[]
        for batch in tqdm(train_loader,desc=f"blackbox train {epoch}",leave=False):
            x=batch["image"].to(device); y=batch["label"].to(device); opt.zero_grad(set_to_none=True)
            with autocast_context(device, bool(tcfg.get("amp",True))): logits=model(x)["logit"]; loss=criterion(logits,y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),float(tcfg.get("gradient_clip_norm",1.0))); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        sched.step(); metrics=evaluate_blackbox(model,valid_loader,device); metrics["train_loss"]=float(np.mean(losses)); metrics["epoch"]=epoch; history.append(metrics); score=metrics["auroc"]
        if score>best: best=score; bad=0; save_checkpoint(ckpt,model,opt,epoch,metrics,cfg)
        else: bad+=1
        print(f"epoch={epoch} train_loss={metrics['train_loss']:.4f} val_auroc={score:.4f}")
        if bad>=int(tcfg["patience"]): break
    return history, ckpt


def evaluate_blackbox(model,loader,device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for batch in loader:
            logits=model(batch["image"].to(device))["logit"]; ys.extend(batch["label"].numpy()); ps.extend(torch.sigmoid(logits).cpu().numpy())
    try: auc=roc_auc_score(ys,ps)
    except Exception: auc=float("nan")
    return {"auroc":float(auc)}

from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from ..constants import CONCEPT_COLUMNS


def collect_concept_outputs(model, loader, device):
    model.eval(); logits={c:[] for c in CONCEPT_COLUMNS}; targets={c:[] for c in CONCEPT_COLUMNS}; labels=[]; indices=[]; embeddings=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="collect-concepts", leave=False):
            x=batch["image"].to(device); full=model(x); out=full["logits"]; embeddings.append(full["embedding"].cpu())
            for c in CONCEPT_COLUMNS:
                logits[c].append(out[c].cpu()); targets[c].append(batch["concepts"][c].cpu())
            labels.append(batch["label"].cpu()); indices.append(batch["source_index"].cpu())
    return {
        "logits":{c:torch.cat(v) for c,v in logits.items()},
        "targets":{c:torch.cat(v) for c,v in targets.items()},
        "labels":torch.cat(labels), "source_indices":torch.cat(indices), "embeddings":torch.cat(embeddings),
    }


def collect_blackbox_outputs(model, loader, device):
    model.eval(); logits=[]; labels=[]; indices=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="collect-blackbox", leave=False):
            x=batch["image"].to(device); logits.append(model(x)["logit"].cpu()); labels.append(batch["label"].cpu()); indices.append(batch["source_index"].cpu())
    return {"logits":torch.cat(logits),"labels":torch.cat(labels),"source_indices":torch.cat(indices)}

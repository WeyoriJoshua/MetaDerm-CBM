from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from ..constants import CONCEPT_COLUMNS
from ..training.collect import collect_concept_outputs, collect_blackbox_outputs
from ..reasoning.seven_point import SoftSevenPointReasoner
from ..reasoning.consistency import build_profile_map, hard_profile_from_probs, lookup_profile_reliability
from ..reasoning.metacognition import metacognitive_features, reliability_score, risk_coverage_curve
from ..data.derm7pt import load_metadata, load_split_indices


def softmax_np(t, temperature=1.0): return torch.softmax(t/float(temperature),dim=1).numpy()
def sigmoid_np(t, temperature=1.0): return torch.sigmoid(t/float(temperature)).numpy()


def evaluate_components(cfg, concept_model, blackbox_model, loader, device, calibration=None):
    c=collect_concept_outputs(concept_model,loader,device); b=collect_blackbox_outputs(blackbox_model,loader,device)
    if not np.array_equal(c["source_indices"].numpy(),b["source_indices"].numpy()): raise RuntimeError("Prediction ordering mismatch")
    temps=(calibration or {}).get("concept_temperatures",{}); bbtemp=(calibration or {}).get("blackbox_temperature",1.0)
    concept_probs={name:softmax_np(c["logits"][name],temps.get(name,1.0)) for name in CONCEPT_COLUMNS}; blackbox_prob=sigmoid_np(b["logits"],bbtemp)
    reasoner=SoftSevenPointReasoner(cfg); reason=reasoner.predict(concept_probs)
    df=load_metadata(cfg); tr=df.iloc[load_split_indices(cfg,"train")].copy(); pmap=build_profile_map(tr,cfg); profiles=hard_profile_from_probs(concept_probs); pr=lookup_profile_reliability(profiles,pmap,cfg["metacognition"].get("unseen_profile_reliability",0.0))
    features=metacognitive_features(concept_probs,blackbox_prob,reason["probability"],pr); reliability=reliability_score(features,cfg)
    return {"concept_raw":c,"concept_probs":concept_probs,"blackbox_prob":blackbox_prob,"reason":reason,"profile_reliability":pr,"features":features,"reliability":reliability,"labels":b["labels"].numpy().astype(int),"source_indices":b["source_indices"].numpy().astype(int),"profile_map":pmap}

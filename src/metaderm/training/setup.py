from __future__ import annotations
import numpy as np
from ..constants import CONCEPT_COLUMNS
from ..data.derm7pt import load_metadata, load_split_indices, concept_mappings
from ..models.losses import balanced_class_weights, binary_pos_weight


def training_class_weights(cfg):
    df=load_metadata(cfg); idx=load_split_indices(cfg,"train"); part=df.iloc[idx]; maps=concept_mappings(cfg); weights={}
    for c in CONCEPT_COLUMNS:
        y=part[c].astype(str).map(maps[c]).to_numpy(); weights[c]=balanced_class_weights(y,len(cfg["data"]["concepts"][c]))
    return weights


def training_pos_weight(cfg):
    df=load_metadata(cfg); idx=load_split_indices(cfg,"train"); return binary_pos_weight(df.iloc[idx]["binary_label"].to_numpy())

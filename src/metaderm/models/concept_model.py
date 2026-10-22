from __future__ import annotations
import torch
import torch.nn as nn
from .backbones import build_backbone
from ..constants import CONCEPT_COLUMNS


class ConceptPerceptionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone, dim = build_backbone(cfg["model"]["backbone"], cfg["model"].get("pretrained", True))
        dropout = float(cfg["model"].get("dropout", 0.0))
        self.dropout = nn.Dropout(dropout)
        vocab = cfg["data"]["concepts"]
        self.heads = nn.ModuleDict({c: nn.Linear(dim, len(vocab[c])) for c in CONCEPT_COLUMNS})

    def forward(self, x):
        z = self.dropout(self.backbone(x))
        logits = {c: self.heads[c](z) for c in CONCEPT_COLUMNS}
        return {"embedding": z, "logits": logits}

    @torch.no_grad()
    def probabilities(self, x):
        out = self.forward(x)
        out["probs"] = {c: torch.softmax(v, dim=1) for c, v in out["logits"].items()}
        return out

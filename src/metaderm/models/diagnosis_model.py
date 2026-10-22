from __future__ import annotations
import torch.nn as nn
from .backbones import build_backbone


class BlackBoxDiagnosisModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone, dim = build_backbone(cfg["model"]["backbone"], cfg["model"].get("pretrained", True))
        self.head = nn.Sequential(nn.Dropout(float(cfg["model"].get("dropout", 0.0))), nn.Linear(dim, 1))

    def forward(self, x):
        z = self.backbone(x)
        return {"embedding": z, "logit": self.head(z).squeeze(1)}

from __future__ import annotations
import torch.nn as nn


class _TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    def forward(self, x):
        return self.net(x)

from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    swin_t, Swin_T_Weights,
)


def build_backbone(name: str, pretrained: bool = True):
    name = name.lower()
    if name == "tiny_cnn":
        return _TinyCNN(), 32
    if name == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = efficientnet_b3(weights=weights)
        dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, dim
    if name in {"swin_t", "swin_tiny"}:
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        model = swin_t(weights=weights)
        dim = model.head.in_features
        model.head = nn.Identity()
        return model, dim
        # Swin / other timm feature-extractor backbones
    if str(name).startswith("swin_"):
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "The selected Swin backbone requires timm. "
                "Install it with: python -m pip install timm"
            ) from exc

        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        dim = int(model.num_features)
        return model, dim

    raise ValueError(f"Unsupported backbone: {name}")

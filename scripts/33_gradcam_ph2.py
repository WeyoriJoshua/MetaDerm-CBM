
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T

from metaderm.config import load_config
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel


def extract_state_dict(obj, model):
    target = set(model.state_dict().keys())
    candidates = []

    def walk(v, path=()):
        if not isinstance(v, dict):
            return
        tensors = {
            (str(k)[7:] if str(k).startswith("module.") else str(k)): val
            for k, val in v.items()
            if torch.is_tensor(val)
        }
        if tensors:
            candidates.append(
                (len(set(tensors) & target), path, tensors)
            )
        for k, child in v.items():
            if isinstance(child, dict):
                walk(child, path + (str(k),))

    walk(obj)
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        raise RuntimeError("No model state found in checkpoint.")

    _, path, state = candidates[0]
    compatible = {
        k: v for k, v in state.items()
        if k in model.state_dict()
        and tuple(v.shape) == tuple(model.state_dict()[k].shape)
    }
    coverage = len(compatible) / len(target)
    print(
        "checkpoint state:",
        ".".join(path) if path else "<root>",
        f"coverage={coverage:.1%}",
    )
    if coverage < 0.95:
        raise RuntimeError("Checkpoint/model coverage below 95%.")
    return compatible


def load_model(cfg, checkpoint, device):
    model = BlackBoxDiagnosisModel(cfg).to(device)
    obj = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        extract_state_dict(obj, model),
        strict=False,
    )
    model.eval()
    return model


def get_logit(output):
    if torch.is_tensor(output):
        if output.ndim == 1:
            return output
        if output.ndim == 2 and output.shape[1] == 1:
            return output[:, 0]
        if output.ndim == 2 and output.shape[1] == 2:
            return output[:, 1] - output[:, 0]

    candidates = []

    def walk(v, path=()):
        if torch.is_tensor(v):
            if v.ndim == 1:
                candidates.append((path, v))
            elif v.ndim == 2 and v.shape[1] == 1:
                candidates.append((path, v[:, 0]))
            elif v.ndim == 2 and v.shape[1] == 2:
                candidates.append((path, v[:, 1] - v[:, 0]))
        elif isinstance(v, dict):
            for k, child in v.items():
                walk(child, path + (str(k),))
        elif isinstance(v, (list, tuple)):
            for i, child in enumerate(v):
                walk(child, path + (str(i),))

    walk(output)
    if not candidates:
        raise TypeError("No diagnosis logit found.")

    candidates.sort(
        key=lambda item: sum(
            term in ".".join(item[0]).lower()
            for term in ["logit", "diagnos", "pred", "class"]
        ),
        reverse=True,
    )
    return candidates[0][1]


def choose_last_conv(backbone):
    last_name = None
    last_module = None
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
            last_module = module
    if last_module is None:
        raise RuntimeError("No Conv2d layer found in backbone.")
    return last_name, last_module


def transform_pair(cfg):
    data = cfg.get("data", {})
    size = int(data.get("image_size", 300))
    mean = data.get("mean", [0.485, 0.456, 0.406])
    std = data.get("std", [0.229, 0.224, 0.225])

    prep = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return prep, size


def median_case(group, prob_col):
    if len(group) == 0:
        return None
    med = group[prob_col].median()
    idx = (group[prob_col] - med).abs().idxmin()
    return group.loc[idx]


def select_cases(pred):
    p = pred["blackbox_prob"]
    y = pred["label"].astype(int)
    pred_label = (p >= 0.5).astype(int)

    categories = {
        "TP": pred[(y == 1) & (pred_label == 1)],
        "FN": pred[(y == 1) & (pred_label == 0)],
        "TN": pred[(y == 0) & (pred_label == 0)],
        "FP": pred[(y == 0) & (pred_label == 1)],
        "QUERY": pred[pred["action"].astype(str).str.lower() == "query"],
        "ABSTAIN": pred[pred["action"].astype(str).str.lower() == "abstain"],
    }

    chosen = []
    seen = set()

    for name, g in categories.items():
        row = median_case(g, "blackbox_prob")
        if row is None:
            continue
        image_id = str(row["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)
        chosen.append((name, row))

    return chosen


def gradcam(model, layer, x):
    activations = []
    gradients = []

    def fwd(_m, _inp, out):
        activations.append(out)

    def bwd(_m, _gin, gout):
        gradients.append(gout[0])

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)

    model.zero_grad(set_to_none=True)
    out = model(x)
    logit = get_logit(out)
    score = logit.sum()
    score.backward()

    h1.remove()
    h2.remove()

    a = activations[-1]
    g = gradients[-1]

    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = (weights * a).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam,
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    cam = cam[0, 0]
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    prob = torch.sigmoid(logit.detach())[0].item()
    return cam.detach().cpu().numpy(), prob


def main():
    p = argparse.ArgumentParser(
        description="Representative PH2 Grad-CAM for frozen black-box seed 2026."
    )
    p.add_argument(
        "--ph2-metadata",
        default="data_external/ph2/ph2_canonical.csv",
    )
    p.add_argument(
        "--predictions",
        default=(
            "results/ph2_external/efficientnet_b3/"
            "seed2026/ph2_predictions.csv"
        ),
    )
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--overlay", default="configs/final_design.yaml")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/efficientnet_b3_final/blackbox_seed2026_best.pt",
    )
    p.add_argument(
        "--calibration",
        default="results/efficientnet_b3_final/calibration/calibration_seed2026.json",
    )
    p.add_argument(
        "--output-root",
        default="results/interpretability/gradcam_ph2_seed2026",
    )
    args = p.parse_args()

    cfg = load_config(args.config, args.overlay)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    model = load_model(
        cfg,
        Path(args.checkpoint),
        device,
    )

    calibration = json.loads(
        Path(args.calibration).read_text(encoding="utf-8")
    )
    blackbox_temperature = calibration.get(
        "blackbox_temperature",
        calibration.get("diagnosis_temperature"),
    )
    if isinstance(blackbox_temperature, dict):
        blackbox_temperature = blackbox_temperature.get("temperature")
    if blackbox_temperature is None:
        raise KeyError(
            f"Could not find black-box temperature in {args.calibration}"
        )
    blackbox_temperature = float(blackbox_temperature)
    print("Frozen black-box temperature:", blackbox_temperature)
    layer_name, layer = choose_last_conv(model.backbone)
    print("Grad-CAM target layer:", layer_name)

    meta = pd.read_csv(args.ph2_metadata)
    pred = pd.read_csv(args.predictions)
    merged = pred.merge(
        meta[["image_id", "image_path"]],
        on="image_id",
        how="left",
    )

    chosen = select_cases(merged)
    if not chosen:
        raise RuntimeError("No representative PH2 cases could be selected.")

    prep, size = transform_pair(cfg)
    outroot = Path(args.output_root)
    outroot.mkdir(parents=True, exist_ok=True)

    rows = []

    for category, row in chosen:
        image_id = str(row["image_id"])
        path = Path(row["image_path"])
        image = Image.open(path).convert("RGB")
        resized = image.resize((size, size))

        x = prep(image).unsqueeze(0).to(device)
        cam, raw_prob = gradcam(
            model,
            layer,
            x,
        )

        raw_prob = float(raw_prob)
        raw_logit = np.log(
            np.clip(raw_prob, 1e-12, 1 - 1e-12)
            / np.clip(1.0 - raw_prob, 1e-12, 1.0)
        )
        calibrated_prob = float(
            1.0 / (1.0 + np.exp(-(raw_logit / blackbox_temperature)))
        )

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        axes[0].imshow(resized)
        axes[0].set_title("Dermoscopic image")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(resized)
        axes[2].imshow(cam, cmap="jet", alpha=0.42)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        fig.suptitle(
            f"{category}: {image_id} | "
            f"label={int(row['label'])} | "
            f"p(melanoma)={float(row['blackbox_prob']):.3f}"
        )
        fig.tight_layout()

        stem = outroot / f"GradCAM_{category}_{image_id}"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        rows.append(
            {
                "category": category,
                "image_id": image_id,
                "label": int(row["label"]),
                "stored_blackbox_prob": float(row["blackbox_prob"]),
                "recomputed_raw_blackbox_prob": raw_prob,
                "recomputed_calibrated_blackbox_prob": calibrated_prob,
                "stored_minus_recomputed_calibrated": float(row["blackbox_prob"]) - calibrated_prob,
                "action": str(row["action"]),
                "image_path": str(path),
                "gradcam_target_layer": layer_name,
            }
        )

    pd.DataFrame(rows).to_csv(
        outroot / "gradcam_cases.csv",
        index=False,
    )

    manifest = {
        "seed": 2026,
        "checkpoint": args.checkpoint,
        "calibration": args.calibration,
        "blackbox_temperature": blackbox_temperature,
        "selection_rule": (
            "One deterministic median-probability representative case from "
            "each available TP/FN/TN/FP/QUERY/ABSTAIN category, without "
            "performance-based manual cherry-picking."
        ),
        "target_layer": layer_name,
        "note": (
            "Grad-CAM is a qualitative localization aid and is not evidence "
            "that the highlighted region is a causal clinical explanation."
        ),
    }
    (outroot / "gradcam_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nSelected cases:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== GRAD-CAM COMPLETE ===")
    print("saved:", outroot)


if __name__ == "__main__":
    main()

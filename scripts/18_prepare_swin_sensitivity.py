from __future__ import annotations

import copy
from pathlib import Path
import sys

import yaml

from metaderm.config import load_config


BASE = Path("configs/base.yaml")
FINAL = Path("configs/final_design.yaml")
OUT = Path("configs/swin_tiny_final.yaml")

SWIN_NAME = "swin_tiny_patch4_window7_224.ms_in1k"


def walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (str(i),))
    else:
        yield path, obj


def nested_set(d, path, value):
    cur = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def is_project_path(path):
    low = ".".join(path).lower()
    return any(
        x in low
        for x in [
            "output_root",
            "checkpoint_root",
            "results_root",
            "save_dir",
            "output_dir",
        ]
    )


def candidate_backbone_paths(cfg):
    exact = []
    semantic = []

    for path, value in walk(cfg):
        if not isinstance(value, str):
            continue
        lv = value.lower()

        if "efficientnet" in lv and not is_project_path(path):
            exact.append((path, value))
            continue

        lp = ".".join(path).lower()
        if (
            any(k in lp for k in ["backbone", "encoder", "architecture", "model_name"])
            and "efficientnet" in lv
        ):
            semantic.append((path, value))

    # de-duplicate while preserving order
    seen = set()
    out = []
    for item in exact + semantic:
        if item[0] not in seen:
            out.append(item)
            seen.add(item[0])
    return out


def image_size_paths(cfg):
    out = []
    for path, value in walk(cfg):
        if not path:
            continue
        key = path[-1].lower()
        if key in {"image_size", "img_size", "input_size"}:
            if isinstance(value, int):
                out.append((path, 224))
            elif (
                isinstance(value, (list, tuple))
                and len(value) in {2, 3}
                and all(isinstance(x, int) for x in value)
            ):
                if len(value) == 2:
                    out.append((path, [224, 224]))
                elif len(value) == 3:
                    # Preserve channels if this is [C,H,W].
                    if value[0] in {1, 3}:
                        out.append((path, [value[0], 224, 224]))
                    else:
                        out.append((path, [224, 224, value[2]]))
    return out


def main():
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    if not FINAL.exists():
        raise FileNotFoundError(
            f"{FINAL} is missing. Copy the frozen-design v1.4 patch first."
        )

    cfg = load_config(str(BASE), str(FINAL))
    final_overlay = yaml.safe_load(FINAL.read_text(encoding="utf-8")) or {}
    overlay = copy.deepcopy(final_overlay)

    candidates = candidate_backbone_paths(cfg)
    if not candidates:
        print("Could not automatically find an EfficientNet model field.")
        print("\nPotential model-related string fields:")
        for path, value in walk(cfg):
            if isinstance(value, str):
                lp = ".".join(path).lower()
                if any(
                    k in lp
                    for k in ["model", "backbone", "encoder", "architecture"]
                ):
                    print("  ", ".".join(path), "=", value)
        sys.exit(2)

    print("Detected EfficientNet backbone field(s):")
    for path, value in candidates:
        print("  ", ".".join(path), "=", value)
        nested_set(overlay, path, SWIN_NAME)

    sizes = image_size_paths(cfg)
    if sizes:
        print("\nSetting Swin input-size field(s) to 224:")
        for path, value in sizes:
            print("  ", ".".join(path), "->", value)
            nested_set(overlay, path, value)
    else:
        print("\nNo explicit image-size field detected; model defaults will be used.")

    overlay.setdefault("project", {})
    overlay["project"]["output_root"] = "results/swin_tiny_final"
    overlay["project"]["checkpoint_root"] = "checkpoints/swin_tiny_final"

    OUT.write_text(
        yaml.safe_dump(overlay, sort_keys=False),
        encoding="utf-8",
    )

    print("\nGenerated:", OUT)
    print(OUT.read_text(encoding="utf-8"))

    # Re-load through the project's own config merger.
    merged = load_config(str(BASE), str(OUT))

    print("\nVerifying the requested model exists in timm...")
    import timm
    available = timm.list_models(
        "swin_tiny_patch4_window7_224*",
        pretrained=True,
    )
    print("Available Swin-Tiny variants:", available[:10])

    # This validates compatibility with the current MetaDerm model wrappers.
    print("\nInstantiating MetaDerm model wrappers...")
    from metaderm.models.concept_model import ConceptPerceptionModel
    from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel

    cm = ConceptPerceptionModel(merged)
    bm = BlackBoxDiagnosisModel(merged)

    n_cm = sum(p.numel() for p in cm.parameters())
    n_bm = sum(p.numel() for p in bm.parameters())

    print("Concept model parameters:", f"{n_cm:,}")
    print("Black-box parameters:", f"{n_bm:,}")
    print("\nSWIN CONFIG VALIDATION PASSED")


if __name__ == "__main__":
    main()

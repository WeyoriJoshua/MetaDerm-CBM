
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt

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
            overlap = len(set(tensors) & target)
            candidates.append((overlap, path, tensors))
        for k, child in v.items():
            if isinstance(child, dict):
                walk(child, path + (str(k),))

    walk(obj)
    if not candidates:
        raise RuntimeError("No tensor state_dict found in checkpoint.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    overlap, path, state = candidates[0]
    compatible = {
        k: v for k, v in state.items()
        if k in model.state_dict()
        and tuple(v.shape) == tuple(model.state_dict()[k].shape)
    }
    coverage = len(compatible) / max(1, len(target))
    print(
        "checkpoint state:",
        ".".join(path) if path else "<root>",
        f"coverage={coverage:.1%}",
    )
    if coverage < 0.95:
        raise RuntimeError(
            f"Checkpoint coverage too low: {coverage:.1%}"
        )
    return compatible


def load_model(cfg, checkpoint, device):
    model = BlackBoxDiagnosisModel(cfg).to(device)
    obj = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    state = extract_state_dict(obj, model)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def eval_transform(cfg):
    data = cfg.get("data", {})
    size = int(data.get("image_size", 300))
    mean = data.get("mean", [0.485, 0.456, 0.406])
    std = data.get("std", [0.229, 0.224, 0.225])
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def resolve_path(root: Path, value):
    if pd.isna(value):
        return None
    p = Path(str(value))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                root / p,
                root / "images" / p,
            ]
        )
    for c in candidates:
        if c.exists():
            return c.resolve()

    name = p.name
    hits = list(root.rglob(name))
    return hits[0].resolve() if hits else None


def read_split_indices(path: Path, col=None):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    values = []

    if col and col in df.columns:
        values = df[col].tolist()
    else:
        values = df.to_numpy().ravel().tolist()

    idx = []
    for v in values:
        try:
            idx.append(int(v))
        except Exception:
            pass

    if len(idx) < 50:
        raw = pd.read_csv(path, header=None)
        idx2 = []
        for v in raw.to_numpy().ravel():
            try:
                idx2.append(int(v))
            except Exception:
                pass
        if len(idx2) > len(idx):
            idx = idx2

    return idx


def build_derm7pt_table(root: Path, cfg):
    data = cfg["data"]

    meta_path = Path(data["meta_csv"])
    if not meta_path.is_absolute():
        meta_path = root / meta_path
    if not meta_path.exists():
        # Common release layout fallback.
        alt = root / "meta" / "meta.csv"
        if alt.exists():
            meta_path = alt

    meta = pd.read_csv(meta_path)

    split_rel = data["split_files"]["test"]
    split_path = Path(split_rel)
    if not split_path.is_absolute():
        split_path = root / split_path
    if not split_path.exists():
        alt = root / "meta" / Path(split_rel).name
        if alt.exists():
            split_path = alt

    idx = read_split_indices(
        split_path,
        data.get("split_index_col"),
    )
    idx = [i for i in idx if 0 <= i < len(meta)]
    test = meta.iloc[idx].copy().reset_index(drop=True)

    image_col = data["image_col"]
    diagnosis_col = data["diagnosis_col"]
    prefix = str(
        data.get("diagnosis_positive_prefix", "melanoma")
    ).strip().lower()

    image_dir = Path(data.get("image_dir", ""))
    image_root = (
        image_dir
        if image_dir.is_absolute()
        else root / image_dir
    )
    if not image_root.exists():
        image_root = root

    paths = []
    for value in test[image_col]:
        p = resolve_path(image_root, value)
        if p is None:
            p = resolve_path(root, value)
        paths.append(str(p) if p else "")

    labels = (
        test[diagnosis_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.startswith(prefix)
        .astype(int)
    )

    out = pd.DataFrame(
        {
            "sample_id": [
                f"derm7pt_{i}" for i in idx
            ],
            "image_path": paths,
            "label": labels,
            "dataset": "Derm7pt",
        }
    )
    return out[out["image_path"].astype(bool)].reset_index(drop=True)


def build_ph2_table(csv_path: Path):
    df = pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "sample_id": df["image_id"].astype(str),
            "image_path": df["image_path"].astype(str),
            "label": df["label"].astype(int),
            "dataset": "PH2",
        }
    )


class ImageTableDataset(Dataset):
    def __init__(self, table, transform):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        r = self.table.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        return (
            self.transform(img),
            int(r["label"]),
            str(r["sample_id"]),
            str(r["dataset"]),
        )


def tensor_from_nested(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for v in output.values():
            try:
                return tensor_from_nested(v)
            except Exception:
                pass
    if isinstance(output, (tuple, list)):
        for v in output:
            try:
                return tensor_from_nested(v)
            except Exception:
                pass
    raise TypeError("No tensor found in backbone output.")


def flatten_representation(output):
    x = tensor_from_nested(output)
    if x.ndim == 4:
        x = torch.nn.functional.adaptive_avg_pool2d(
            x, (1, 1)
        ).flatten(1)
    elif x.ndim > 2:
        x = x.flatten(1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def main():
    p = argparse.ArgumentParser(
        description=(
            "Frozen black-box representation visualization "
            "for Derm7pt and PH2."
        )
    )
    p.add_argument("--derm7pt-root", required=True)
    p.add_argument(
        "--ph2-metadata",
        default="data_external/ph2/ph2_canonical.csv",
    )
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--overlay", default="configs/final_design.yaml")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/efficientnet_b3_final/blackbox_seed2026_best.pt",
    )
    p.add_argument(
        "--output-root",
        default="results/interpretability/representation_seed2026",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    cfg = load_config(args.config, args.overlay)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    d7 = build_derm7pt_table(
        Path(args.derm7pt_root).resolve(),
        cfg,
    )
    ph2 = build_ph2_table(
        Path(args.ph2_metadata).resolve()
    )

    table = pd.concat([d7, ph2], ignore_index=True)
    print("Derm7pt samples:", len(d7))
    print("PH2 samples:", len(ph2))
    print("Total:", len(table))

    transform = eval_transform(cfg)
    loader = DataLoader(
        ImageTableDataset(table, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(
        cfg,
        Path(args.checkpoint),
        device,
    )

    captured = []

    def hook(_module, _inputs, output):
        captured.append(
            flatten_representation(output)
            .detach()
            .cpu()
        )

    handle = model.backbone.register_forward_hook(hook)

    with torch.no_grad():
        for x, _, _, _ in loader:
            x = x.to(device)
            _ = model(x)

    handle.remove()

    features = torch.cat(captured, dim=0).numpy()
    if len(features) != len(table):
        raise RuntimeError(
            f"Feature/sample mismatch: {len(features)} vs {len(table)}"
        )

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "features.npy", features)
    table.to_csv(out / "embedding_metadata.csv", index=False)

    # PCA
    pca = PCA(n_components=2, random_state=args.seed)
    pca_xy = pca.fit_transform(features)

    pca_df = table.copy()
    pca_df["x"] = pca_xy[:, 0]
    pca_df["y"] = pca_xy[:, 1]
    pca_df.to_csv(out / "pca_coordinates.csv", index=False)

    # t-SNE on a compact PCA representation for stability/speed.
    n_pre = min(50, features.shape[1], len(features) - 1)
    pre = PCA(
        n_components=n_pre,
        random_state=args.seed,
    ).fit_transform(features)

    perplexity = min(30, max(5, (len(features) - 1) // 10))
    tsne_xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(pre)

    tsne_df = table.copy()
    tsne_df["x"] = tsne_xy[:, 0]
    tsne_df["y"] = tsne_xy[:, 1]
    tsne_df.to_csv(out / "tsne_coordinates.csv", index=False)

    # Optional UMAP if already installed; do not require it.
    umap_df = None
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            random_state=args.seed,
            n_neighbors=15,
            min_dist=0.1,
        )
        umap_xy = reducer.fit_transform(pre)
        umap_df = table.copy()
        umap_df["x"] = umap_xy[:, 0]
        umap_df["y"] = umap_xy[:, 1]
        umap_df.to_csv(out / "umap_coordinates.csv", index=False)
        print("UMAP: generated")
    except Exception:
        print("UMAP: package not installed; PCA and t-SNE generated.")

    def plot_embedding(df, title, stem):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        markers = {"Derm7pt": "o", "PH2": "^"}

        for dataset in ["Derm7pt", "PH2"]:
            for label in [0, 1]:
                g = df[
                    (df["dataset"] == dataset)
                    & (df["label"] == label)
                ]
                if len(g) == 0:
                    continue
                ax.scatter(
                    g["x"],
                    g["y"],
                    marker=markers[dataset],
                    alpha=0.65,
                    s=28,
                    label=(
                        f"{dataset} - "
                        f"{'melanoma' if label else 'non-melanoma'}"
                    ),
                )

        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    plot_embedding(
        pca_df,
        "Frozen EfficientNet-B3 representation: PCA",
        "Fig_representation_PCA",
    )
    plot_embedding(
        tsne_df,
        "Frozen EfficientNet-B3 representation: t-SNE",
        "Fig_representation_tSNE",
    )
    if umap_df is not None:
        plot_embedding(
            umap_df,
            "Frozen EfficientNet-B3 representation: UMAP",
            "Fig_representation_UMAP",
        )

    manifest = {
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "n_derm7pt": int(len(d7)),
        "n_ph2": int(len(ph2)),
        "feature_dimension": int(features.shape[1]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "tsne_perplexity": float(perplexity),
        "note": (
            "Representation visualization is descriptive/qualitative and "
            "does not modify or select the frozen model."
        ),
    }
    (out / "representation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== REPRESENTATION VISUALIZATION COMPLETE ===")
    print("saved:", out)


if __name__ == "__main__":
    main()

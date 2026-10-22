from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .transforms import build_transforms
from ..constants import CONCEPT_COLUMNS


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    d = cfg["data"]
    root = Path(d["root"]).expanduser().resolve()
    return {
        "root": root,
        "images": root / d["image_dir"],
        "meta": root / d["meta_csv"],
        **{f"split_{k}": root / v for k, v in d["split_files"].items()},
    }


def validate_release(cfg: dict[str, Any]) -> None:
    paths = resolve_paths(cfg)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Derm7pt release is incomplete. Missing:\n" + "\n".join(missing))


def load_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    validate_release(cfg)
    d = cfg["data"]
    df = pd.read_csv(resolve_paths(cfg)["meta"])
    required = [d["image_col"], d["diagnosis_col"], *CONCEPT_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"meta.csv is missing required columns: {missing}")
    df = df.copy()
    df["binary_label"] = (
        df[d["diagnosis_col"]].astype(str).str.lower().str.startswith(d["diagnosis_positive_prefix"].lower())
    ).astype(int)
    return df


def load_split_indices(cfg: dict[str, Any], split: str) -> list[int]:
    paths = resolve_paths(cfg)
    f = paths[f"split_{split}"]
    col = cfg["data"].get("split_index_col", "indexes")
    sdf = pd.read_csv(f)
    if col not in sdf.columns:
        raise ValueError(f"{f} must contain column '{col}'")
    return sdf[col].astype(int).tolist()


def validate_concept_vocab(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    vocab = cfg["data"]["concepts"]
    problems = []
    for concept in CONCEPT_COLUMNS:
        expected = set(vocab[concept])
        observed = set(df[concept].dropna().astype(str).unique())
        unknown = observed - expected
        if unknown:
            problems.append(f"{concept}: unknown states {sorted(unknown)}")
    if problems:
        raise ValueError("Concept vocabulary mismatch:\n" + "\n".join(problems))


def concept_mappings(cfg: dict[str, Any]):
    vocab = cfg["data"]["concepts"]
    to_index = {c: {v: i for i, v in enumerate(vocab[c])} for c in CONCEPT_COLUMNS}
    return to_index


class Derm7ptDataset(Dataset):
    def __init__(self, cfg, split: str, train_transform: bool | None = None):
        self.cfg = cfg
        self.df_all = load_metadata(cfg)
        validate_concept_vocab(self.df_all, cfg)
        self.indices = load_split_indices(cfg, split)
        self.df = self.df_all.iloc[self.indices].copy().reset_index().rename(columns={"index": "source_index"})
        self.paths = resolve_paths(cfg)
        self.map = concept_mappings(cfg)
        if train_transform is None:
            train_transform = split == "train"
        self.transform = build_transforms(cfg["data"]["image_size"], train_transform)
        self.image_col = cfg["data"]["image_col"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.paths["images"] / str(row[self.image_col])
        with Image.open(image_path) as im:
            image = self.transform(im.convert("RGB"))
        concepts = {
            c: torch.tensor(self.map[c][str(row[c])], dtype=torch.long)
            for c in CONCEPT_COLUMNS
        }
        return {
            "image": image,
            "concepts": concepts,
            "label": torch.tensor(float(row["binary_label"]), dtype=torch.float32),
            "source_index": torch.tensor(int(row["source_index"]), dtype=torch.long),
        }


def make_loader(cfg, split: str, shuffle: bool | None = None, batch_size: int | None = None):
    ds = Derm7ptDataset(cfg, split)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size or cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        persistent_workers=bool(cfg["data"].get("num_workers", 0) > 0),
    )

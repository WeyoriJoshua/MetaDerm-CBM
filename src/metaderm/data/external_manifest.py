from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .transforms import build_transforms


class ExternalManifestDataset(Dataset):
    """Generic external-validation adapter.

    Required CSV columns:
      image_path, label
    Optional overlapping concept columns may be included and analysed separately.
    `label` must already be encoded as 0/1 using the external dataset's documented diagnosis mapping.
    """
    def __init__(self, csv_path, image_root, image_size=224):
        self.df = pd.read_csv(csv_path)
        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"External manifest missing columns: {sorted(missing)}")
        self.image_root = Path(image_root)
        self.transform = build_transforms(image_size, train=False)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(self.image_root / str(row["image_path"])) as im:
            image = self.transform(im.convert("RGB"))
        return {"image": image, "label": torch.tensor(float(row["label"])), "row": idx}

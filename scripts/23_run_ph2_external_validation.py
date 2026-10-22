
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from metaderm.config import load_config
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel

try:
    from metaderm.reasoning.seven_point import SoftSevenPointReasoner
except Exception as exc:
    raise ImportError(
        "Could not import SoftSevenPointReasoner from "
        "metaderm.reasoning.seven_point"
    ) from exc

try:
    from metaderm.reasoning.consistency import (
        build_profile_map,
        hard_profile_from_probs,
        lookup_profile_reliability,
    )
except Exception as exc:
    raise ImportError(
        "Could not import the frozen profile-reliability helpers from "
        "metaderm.reasoning.consistency"
    ) from exc

try:
    from metaderm.reasoning.metacognition import (
        metacognitive_features,
        reliability_score,
        assign_actions,
    )
except Exception as exc:
    raise ImportError(
        "Could not import the frozen metacognition helpers from "
        "metaderm.reasoning.metacognition"
    ) from exc


CONCEPTS = [
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
]

STATES = {
    "pigment_network": ["absent", "typical", "atypical"],
    "streaks": ["absent", "regular", "irregular"],
    "pigmentation": [
        "absent",
        "diffuse regular",
        "localized regular",
        "localized irregular",
        "diffuse irregular",
    ],
    "regression_structures": [
        "absent",
        "blue areas",
        "white areas",
        "combinations",
    ],
    "dots_and_globules": ["absent", "regular", "irregular"],
    "blue_whitish_veil": ["absent", "present"],
    "vascular_structures": [
        "absent",
        "comma",
        "wreath",
        "arborizing",
        "hairpin",
        "dotted",
        "within regression",
        "linear irregular",
    ],
}


def state_index(concept, state):
    return STATES[concept].index(state)


def load_checkpoint_flexible(model, path: Path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    target_state = model.state_dict()
    target_keys = set(target_state.keys())

    def normalize_key(k):
        k = str(k)
        for prefix in ("module.", "model.", "net."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        return k

    candidates = []

    def walk(value, path_parts=()):
        if not isinstance(value, dict):
            return

        tensor_items = {
            normalize_key(k): v
            for k, v in value.items()
            if torch.is_tensor(v)
        }

        if tensor_items:
            overlap = len(set(tensor_items.keys()) & target_keys)
            ratio = overlap / max(1, len(target_keys))
            candidates.append(
                (overlap, ratio, path_parts, tensor_items)
            )

        for k, v in value.items():
            if isinstance(v, dict):
                walk(v, path_parts + (str(k),))

    walk(obj)

    if not candidates:
        raise RuntimeError(
            f"No nested tensor state_dict found in checkpoint: {path}"
        )

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    overlap, ratio, chosen_path, state = candidates[0]

    print(
        f"checkpoint {path.name}: selected "
        f"{'.'.join(chosen_path) if chosen_path else '<root>'}; "
        f"matched {overlap}/{len(target_keys)} keys"
    )

    compatible = {}
    shape_mismatches = []

    for k, v in state.items():
        if k not in target_state:
            continue
        if tuple(v.shape) != tuple(target_state[k].shape):
            shape_mismatches.append(
                (k, tuple(v.shape), tuple(target_state[k].shape))
            )
            continue
        compatible[k] = v

    coverage = len(compatible) / max(1, len(target_keys))

    print(
        f"checkpoint {path.name}: compatible "
        f"{len(compatible)}/{len(target_keys)} "
        f"({coverage:.1%})"
    )

    if shape_mismatches:
        print(
            f"checkpoint {path.name}: "
            f"{len(shape_mismatches)} shape mismatch(es)"
        )
        for item in shape_mismatches[:10]:
            print("  ", item)

    if coverage < 0.95:
        missing = sorted(target_keys - set(compatible.keys()))
        print("First missing model keys:", missing[:20])
        raise RuntimeError(
            f"Checkpoint/model coverage too low for {path}: "
            f"{coverage:.1%}. External validation stopped."
        )

    missing, unexpected = model.load_state_dict(
        compatible,
        strict=False,
    )

    if unexpected:
        raise RuntimeError(
            f"Unexpected keys remained for {path}: {unexpected[:20]}"
        )

    if missing:
        print(
            f"checkpoint {path.name}: "
            f"{len(missing)} key(s) left at initialization"
        )
        print("missing sample:", missing[:20])

    print(f"checkpoint {path.name}: LOAD PASSED")

def calibration_values(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))

    concept_t = (
        d.get("concept_temperatures")
        or d.get("concept_temperature")
        or d.get("concepts")
    )
    if not isinstance(concept_t, dict):
        raise KeyError(
            f"Could not find concept temperatures in {path}"
        )

    bb = (
        d.get("blackbox_temperature")
        or d.get("blackbox")
        or d.get("diagnosis_temperature")
    )
    if isinstance(bb, dict):
        bb = bb.get("temperature")

    if bb is None:
        raise KeyError(
            f"Could not find black-box temperature in {path}"
        )

    return concept_t, float(bb)


def build_eval_transform(cfg):
    data_cfg = cfg.get("data", {})
    size = int(data_cfg.get("image_size", 300))

    mean = data_cfg.get(
        "mean",
        [0.485, 0.456, 0.406],
    )
    std = data_cfg.get(
        "std",
        [0.229, 0.224, 0.225],
    )

    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    ), size


class PH2Dataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["label"]).copy()
        self.df["label"] = self.df["label"].astype(int)
        self.df = self.df.reset_index(drop=True)
        self.transform = transform

        missing = [
            p for p in self.df["image_path"].astype(str)
            if not Path(p).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} PH2 images are missing. "
                f"First missing path: {missing[0]}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image = Image.open(r["image_path"]).convert("RGB")
        x = self.transform(image)
        return x, int(r["label"]), idx


def unpack_concepts(output):
    if isinstance(output, dict):
        if all(c in output for c in CONCEPTS):
            return {c: output[c] for c in CONCEPTS}

        for key in ["concept_logits", "logits", "outputs"]:
            val = output.get(key)
            if isinstance(val, dict) and all(
                c in val for c in CONCEPTS
            ):
                return {c: val[c] for c in CONCEPTS}

    if isinstance(output, (list, tuple)):
        if len(output) == len(CONCEPTS) and all(
            torch.is_tensor(v) for v in output
        ):
            return dict(zip(CONCEPTS, output))
        for item in output:
            try:
                return unpack_concepts(item)
            except Exception:
                continue

    raise TypeError(
        "Could not identify the seven concept logits from "
        "ConceptPerceptionModel output."
    )


def unpack_blackbox(output):
    # Direct tensor output.
    if torch.is_tensor(output):
        if output.ndim == 1:
            return output
        if output.ndim == 2 and output.shape[1] in (1, 2):
            return output
        raise TypeError(
            f"Direct black-box tensor has non-diagnostic shape: {tuple(output.shape)}"
        )

    # Recursively collect tensor candidates and retain their structural path.
    candidates = []

    def walk(value, path=()):
        if torch.is_tensor(value):
            shape = tuple(value.shape)
            diagnostic = (
                value.ndim == 1
                or (
                    value.ndim == 2
                    and value.shape[1] in (1, 2)
                )
            )
            if diagnostic:
                candidates.append((path, value))
            return

        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, path + (str(k),))
            return

        if isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                walk(v, path + (str(i),))
            return

    walk(output)

    if not candidates:
        # Diagnostic dump without exposing large tensor contents.
        def describe(value, path=(), depth=0):
            if depth > 4:
                return
            if torch.is_tensor(value):
                print(
                    "black-box output tensor:",
                    ".".join(path) if path else "<root>",
                    tuple(value.shape),
                )
            elif isinstance(value, dict):
                for k, v in value.items():
                    describe(v, path + (str(k),), depth + 1)
            elif isinstance(value, (tuple, list)):
                for i, v in enumerate(value):
                    describe(v, path + (str(i),), depth + 1)
            else:
                print(
                    "black-box output object:",
                    ".".join(path) if path else "<root>",
                    type(value).__name__,
                )

        print("Could not find a diagnosis-shaped tensor. Output structure:")
        describe(output)
        raise TypeError(
            "Could not identify black-box diagnosis logits from model output."
        )

    # Prefer semantically named paths when more than one diagnosis-shaped
    # tensor exists.
    preferred_terms = (
        "logit",
        "diagnos",
        "class",
        "pred",
        "output",
    )

    scored = []
    for path, tensor in candidates:
        low = ".".join(path).lower()
        score = sum(term in low for term in preferred_terms)
        # A 2-logit tensor is a very common binary classifier output.
        if tensor.ndim == 2 and tensor.shape[1] == 2:
            score += 1
        scored.append((score, path, tensor))

    scored.sort(key=lambda x: x[0], reverse=True)
    score, path, tensor = scored[0]

    print(
        "black-box output selected:",
        ".".join(path) if path else "<root>",
        "shape=",
        tuple(tensor.shape),
    )

    return tensor

def binary_probability(logits, temperature):
    logits = logits / float(temperature)

    if logits.ndim == 1:
        return torch.sigmoid(logits)

    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])

    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]

    raise ValueError(
        f"Unexpected black-box logits shape: {tuple(logits.shape)}"
    )


def ece_binary(y, p, n_bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0

    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1.0 - p)

    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if not mask.any():
            continue

        bin_acc = np.mean(pred[mask] == y[mask])
        bin_conf = np.mean(conf[mask])
        ece += mask.mean() * abs(bin_acc - bin_conf)

    return float(ece)


def diagnostic_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "sensitivity_recall": float(
            tp / (tp + fn) if (tp + fn) else np.nan
        ),
        "specificity": float(
            tn / (tn + fp) if (tn + fp) else np.nan
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(
            average_precision_score(y, p)
        ),
        "brier": float(brier_score_loss(y, p)),
        "ece_10": ece_binary(y, p, 10),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def selective_metrics(y, p, actions):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    actions = np.asarray(actions).astype(str)

    pm = actions == "predict"
    qm = actions == "query"
    am = actions == "abstain"

    out = {
        "predict_coverage": float(pm.mean()),
        "query_rate": float(qm.mean()),
        "abstain_rate": float(am.mean()),
    }

    if pm.any():
        yp = y[pm]
        pp = pred[pm]
        tn, fp, fn, tp = confusion_matrix(
            yp, pp, labels=[0, 1]
        ).ravel()

        out.update(
            {
                "selective_risk": float(
                    np.mean(pp != yp)
                ),
                "accepted_accuracy": float(
                    np.mean(pp == yp)
                ),
                "accepted_balanced_accuracy": float(
                    balanced_accuracy_score(yp, pp)
                ),
                "accepted_sensitivity": float(
                    tp / (tp + fn)
                    if (tp + fn)
                    else np.nan
                ),
                "accepted_specificity": float(
                    tn / (tn + fp)
                    if (tn + fp)
                    else np.nan
                ),
                "accepted_tn": int(tn),
                "accepted_fp": int(fp),
                "accepted_fn": int(fn),
                "accepted_tp": int(tp),
            }
        )

    out["benign_predict_coverage"] = float(
        np.sum(pm & (y == 0)) / np.sum(y == 0)
    )
    out["melanoma_predict_coverage"] = float(
        np.sum(pm & (y == 1)) / np.sum(y == 1)
    )

    errors = pred != y
    out["error_capture_rate"] = float(
        np.sum(errors & (~pm)) / np.sum(errors)
    ) if np.sum(errors) else np.nan

    return out


def load_derm7pt_train_metadata(root: Path, cfg):
    meta_path = root / "meta" / "meta.csv"
    split_path = root / "meta" / "train_indexes.csv"

    if not split_path.exists():
        split_path = root / "train_indexes.csv"

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    meta = pd.read_csv(meta_path)

    # Recreate the exact binary diagnosis field expected by
    # reasoning.consistency.build_profile_map().
    if "binary_label" not in meta.columns:
        data_cfg = cfg.get("data", {})
        diagnosis_col = data_cfg.get("diagnosis_col")
        positive_prefix = str(
            data_cfg.get("diagnosis_positive_prefix", "melanoma")
        ).strip().lower()

        if not diagnosis_col or diagnosis_col not in meta.columns:
            raise KeyError(
                "Derm7pt metadata does not contain 'binary_label', and the "
                f"configured diagnosis column could not be found: {diagnosis_col!r}. "
                f"Available columns: {list(meta.columns)}"
            )

        diagnosis_text = (
            meta[diagnosis_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        meta["binary_label"] = (
            diagnosis_text.str.startswith(positive_prefix)
        ).astype(int)

        print(
            "Reconstructed Derm7pt binary_label from",
            diagnosis_col,
            "using positive prefix",
            repr(positive_prefix),
        )
        print(
            "Derm7pt full metadata binary counts:",
            meta["binary_label"].value_counts().sort_index().to_dict(),
        )

    # The split file may contain a header or a named index column.
    split = pd.read_csv(split_path)

    split_index_col = cfg.get("data", {}).get("split_index_col")
    idx = []

    if split_index_col and split_index_col in split.columns:
        values = split[split_index_col].tolist()
    else:
        # Fallback: collect all numeric-looking entries.
        values = split.to_numpy().ravel().tolist()

    for value in values:
        try:
            idx.append(int(value))
        except Exception:
            continue

    idx = [i for i in idx if 0 <= i < len(meta)]

    # Some one-column CSVs written without a header are easier to recover
    # by reading header=None if the first valid index was swallowed as a header.
    if len(idx) < 300:
        raw = pd.read_csv(split_path, header=None)
        idx2 = []
        for value in raw.to_numpy().ravel().tolist():
            try:
                idx2.append(int(value))
            except Exception:
                continue
        idx2 = [i for i in idx2 if 0 <= i < len(meta)]
        if len(idx2) > len(idx):
            idx = idx2

    if not idx:
        raise RuntimeError(
            f"No valid train indices could be read from {split_path}"
        )

    train_meta = meta.iloc[idx].copy().reset_index(drop=True)

    print(
        "Derm7pt profile-map training rows:",
        len(train_meta),
        "binary counts:",
        train_meta["binary_label"].value_counts().sort_index().to_dict(),
    )

    return train_meta

def reasoner_probability(reasoner, concept_probs):
    out = reasoner.predict(concept_probs)

    if isinstance(out, dict):
        for key in [
            "probability",
            "prob",
            "melanoma_probability",
            "score",
        ]:
            if key in out:
                return np.asarray(out[key], dtype=float)

    if isinstance(out, np.ndarray):
        return out.astype(float)

    raise TypeError(
        "Could not extract reasoner probability from "
        "SoftSevenPointReasoner.predict()."
    )


def build_profile_reliability(train_meta, concept_probs, cfg):
    profile_map = build_profile_map(train_meta, cfg)
    profiles = hard_profile_from_probs(concept_probs)

    unseen = cfg.get(
        "metacognition", {}
    ).get(
        "unseen_profile_reliability",
        0.0,
    )

    return lookup_profile_reliability(
        profiles,
        profile_map,
        unseen,
    )


def run_seed(
    seed,
    cfg,
    ph2_csv,
    derm7pt_root,
    checkpoint_root,
    internal_results_root,
    output_root,
    batch_size,
    device,
):
    print(f"\n=== PH2 external seed {seed} ===")

    concept_ckpt = (
        checkpoint_root
        / f"concepts_seed{seed}_best.pt"
    )
    blackbox_ckpt = (
        checkpoint_root
        / f"blackbox_seed{seed}_best.pt"
    )
    calibration_path = (
        internal_results_root
        / "calibration"
        / f"calibration_seed{seed}.json"
    )
    internal_metrics_path = (
        internal_results_root
        / "final"
        / f"seed{seed}"
        / "metrics.json"
    )

    for path in [
        concept_ckpt,
        blackbox_ckpt,
        calibration_path,
        internal_metrics_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    concept_t, blackbox_t = calibration_values(
        calibration_path
    )
    internal_metrics = json.loads(
        internal_metrics_path.read_text(
            encoding="utf-8"
        )
    )
    tau_low = float(internal_metrics["tau_low"])
    tau_high = float(internal_metrics["tau_high"])

    concept_model = ConceptPerceptionModel(cfg).to(device)
    blackbox_model = BlackBoxDiagnosisModel(cfg).to(device)

    load_checkpoint_flexible(
        concept_model,
        concept_ckpt,
        device,
    )
    load_checkpoint_flexible(
        blackbox_model,
        blackbox_ckpt,
        device,
    )

    concept_model.eval()
    blackbox_model.eval()

    transform, image_size = build_eval_transform(cfg)
    ds = PH2Dataset(ph2_csv, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    labels = []
    source_idx = []
    bb_probs = []
    concept_buffers = {
        c: [] for c in CONCEPTS
    }

    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)

            cout = unpack_concepts(
                concept_model(x)
            )
            bout = unpack_blackbox(
                blackbox_model(x)
            )

            for concept in CONCEPTS:
                temp = float(concept_t[concept])
                prob = torch.softmax(
                    cout[concept] / temp,
                    dim=1,
                )
                concept_buffers[concept].append(
                    prob.cpu().numpy()
                )

            bb_prob = binary_probability(
                bout,
                blackbox_t,
            )
            bb_probs.append(
                bb_prob.cpu().numpy()
            )

            labels.extend(y.numpy().tolist())
            source_idx.extend(idx.numpy().tolist())

    labels = np.asarray(labels, dtype=int)
    bb_probs = np.concatenate(
        bb_probs,
        axis=0,
    )
    concept_probs = {
        c: np.concatenate(v, axis=0)
        for c, v in concept_buffers.items()
    }

    reasoner = SoftSevenPointReasoner(cfg)
    reason_probs = reasoner_probability(
        reasoner,
        concept_probs,
    )

    train_meta = load_derm7pt_train_metadata(
        derm7pt_root,
        cfg,
    )
    profile_rel = build_profile_reliability(
        train_meta,
        concept_probs,
        cfg,
    )

    features = metacognitive_features(
        concept_probs,
        bb_probs,
        reason_probs,
        profile_rel,
    )
    reliability = reliability_score(
        features,
        cfg,
    )
    actions = assign_actions(
        reliability,
        tau_low,
        tau_high,
    )
    actions = np.asarray(actions).astype(str)

    blackbox_metrics = diagnostic_metrics(
        labels,
        bb_probs,
    )
    reason_metrics = diagnostic_metrics(
        labels,
        reason_probs,
    )
    selective = selective_metrics(
        labels,
        bb_probs,
        actions,
    )

    meta = ds.df.iloc[
        source_idx
    ].reset_index(drop=True)

    pred_df = pd.DataFrame(
        {
            "image_id": meta["image_id"],
            "clinical_diagnosis": meta[
                "clinical_diagnosis"
            ],
            "label": labels,
            "blackbox_prob": bb_probs,
            "reason_prob": reason_probs,
            "reliability": reliability,
            "action": actions,
        }
    )

    seed_dir = output_root / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(
        seed_dir / "ph2_predictions.csv",
        index=False,
    )

    concept_rows = []

    # 1) Pigment network: evaluate typical vs atypical only.
    y_text = meta["pigment_network"].astype(str)
    valid = y_text.isin(["typical", "atypical"])
    if valid.any():
        ids = np.flatnonzero(valid.to_numpy())
        y = (
            y_text[valid] == "atypical"
        ).astype(int).to_numpy()

        pt = concept_probs[
            "pigment_network"
        ][ids, state_index(
            "pigment_network",
            "typical",
        )]
        pa = concept_probs[
            "pigment_network"
        ][ids, state_index(
            "pigment_network",
            "atypical",
        )]
        denom = np.clip(pt + pa, 1e-12, None)
        p = pa / denom
        pred = (p >= 0.5).astype(int)

        concept_rows.append(
            {
                "seed": seed,
                "concept": "pigment_network_typical_vs_atypical",
                "n": len(y),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(
                    y, pred, average="macro",
                    zero_division=0,
                ),
                "auroc": roc_auc_score(y, p),
            }
        )

    # 2) Dots/globules exact 3-state mapping.
    y_text = meta[
        "dots_and_globules"
    ].astype(str)
    valid = y_text.isin(
        ["absent", "typical", "atypical"]
    )
    if valid.any():
        ids = np.flatnonzero(valid.to_numpy())
        mapping = {
            "absent": state_index(
                "dots_and_globules",
                "absent",
            ),
            "typical": state_index(
                "dots_and_globules",
                "regular",
            ),
            "atypical": state_index(
                "dots_and_globules",
                "irregular",
            ),
        }
        y = np.asarray(
            [mapping[v] for v in y_text[valid]]
        )
        probs = concept_probs[
            "dots_and_globules"
        ][ids]
        pred = probs.argmax(axis=1)

        aucs = []
        for cls in range(3):
            yy = (y == cls).astype(int)
            if len(np.unique(yy)) == 2:
                aucs.append(
                    roc_auc_score(
                        yy,
                        probs[:, cls],
                    )
                )

        concept_rows.append(
            {
                "seed": seed,
                "concept": "dots_and_globules_3state",
                "n": len(y),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(
                    y,
                    pred,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                ),
                "auroc": float(np.mean(aucs)),
            }
        )

    # 3-5) Presence mappings.
    binary_specs = [
        (
            "streaks",
            lambda pr: (
                pr[:, state_index(
                    "streaks", "regular"
                )]
                + pr[:, state_index(
                    "streaks", "irregular"
                )]
            ),
        ),
        (
            "regression_structures",
            lambda pr: (
                1.0
                - pr[:, state_index(
                    "regression_structures",
                    "absent",
                )]
            ),
        ),
        (
            "blue_whitish_veil",
            lambda pr: pr[:, state_index(
                "blue_whitish_veil",
                "present",
            )],
        ),
    ]

    for concept, prob_fn in binary_specs:
        y_text = meta[concept].astype(str)
        valid = y_text.isin(
            ["absent", "present"]
        )
        if not valid.any():
            continue

        ids = np.flatnonzero(
            valid.to_numpy()
        )
        y = (
            y_text[valid] == "present"
        ).astype(int).to_numpy()
        p = prob_fn(
            concept_probs[concept][ids]
        )
        pred = (p >= 0.5).astype(int)

        concept_rows.append(
            {
                "seed": seed,
                "concept": f"{concept}_presence",
                "n": len(y),
                "accuracy": accuracy_score(
                    y, pred
                ),
                "macro_f1": f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0,
                ),
                "auroc": roc_auc_score(y, p),
            }
        )

    concept_df = pd.DataFrame(
        concept_rows
    )
    concept_df.to_csv(
        seed_dir / "concept_transfer.csv",
        index=False,
    )

    result = {
        "seed": seed,
        "n": int(len(labels)),
        "melanoma_n": int(labels.sum()),
        "melanoma_prevalence": float(
            labels.mean()
        ),
        "image_size": image_size,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "blackbox": blackbox_metrics,
        "soft_reasoner": reason_metrics,
        "selective": selective,
    }

    (
        seed_dir / "metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print("blackbox:", blackbox_metrics)
    print("soft_reasoner:", reason_metrics)
    print("selective:", selective)
    print("\npartial concept transfer")
    print(concept_df.to_string(index=False))

    return result, concept_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen five-seed MetaDerm-CBM external "
            "validation on PH2."
        )
    )
    parser.add_argument(
        "--ph2-metadata",
        default=(
            "data_external/ph2/"
            "ph2_canonical.csv"
        ),
    )
    parser.add_argument(
        "--derm7pt-root",
        required=True,
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )
    parser.add_argument(
        "--overlay",
        default="configs/final_design.yaml",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/efficientnet_b3_final",
    )
    parser.add_argument(
        "--internal-results-root",
        default="results/efficientnet_b3_final",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "results/ph2_external/"
            "efficientnet_b3"
        ),
    )
    parser.add_argument(
        "--seeds",
        default="2026,2027,2028,2029,2030",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        args.overlay,
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("device:", device)

    seeds = [
        int(x.strip())
        for x in args.seeds.split(",")
        if x.strip()
    ]

    ph2_csv = Path(
        args.ph2_metadata
    ).resolve()
    derm7pt_root = Path(
        args.derm7pt_root
    ).resolve()
    checkpoint_root = Path(
        args.checkpoint_root
    )
    internal_results_root = Path(
        args.internal_results_root
    )
    output_root = Path(
        args.output_root
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_rows = []
    selective_rows = []
    concept_frames = []

    for seed in seeds:
        result, concept_df = run_seed(
            seed=seed,
            cfg=cfg,
            ph2_csv=ph2_csv,
            derm7pt_root=derm7pt_root,
            checkpoint_root=checkpoint_root,
            internal_results_root=internal_results_root,
            output_root=output_root,
            batch_size=args.batch_size,
            device=device,
        )

        for model_name in [
            "blackbox",
            "soft_reasoner",
        ]:
            diagnostic_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    **result[model_name],
                }
            )

        selective_rows.append(
            {
                "seed": seed,
                **result["selective"],
            }
        )
        concept_frames.append(
            concept_df
        )

    diag = pd.DataFrame(
        diagnostic_rows
    )
    sel = pd.DataFrame(
        selective_rows
    )
    concept_all = pd.concat(
        concept_frames,
        ignore_index=True,
    )

    diag.to_csv(
        output_root
        / "diagnostic_per_seed.csv",
        index=False,
    )
    sel.to_csv(
        output_root
        / "selective_per_seed.csv",
        index=False,
    )
    concept_all.to_csv(
        output_root
        / "concept_transfer_per_seed.csv",
        index=False,
    )

    # Publication summaries.
    metric_cols = [
        c for c in diag.columns
        if c not in {
            "seed", "model",
            "tn", "fp", "fn", "tp",
        }
        and pd.api.types.is_numeric_dtype(
            diag[c]
        )
    ]

    summary_rows = []
    for model, group in diag.groupby(
        "model"
    ):
        for metric in metric_cols:
            summary_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": group[
                        metric
                    ].mean(),
                    "sd": group[
                        metric
                    ].std(ddof=1),
                    "min": group[
                        metric
                    ].min(),
                    "max": group[
                        metric
                    ].max(),
                }
            )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        output_root
        / "diagnostic_mean_sd.csv",
        index=False,
    )

    sel_cols = [
        c for c in sel.columns
        if c != "seed"
        and pd.api.types.is_numeric_dtype(
            sel[c]
        )
    ]
    pd.DataFrame(
        [
            {
                "metric": c,
                "mean": sel[c].mean(),
                "sd": sel[c].std(ddof=1),
                "min": sel[c].min(),
                "max": sel[c].max(),
            }
            for c in sel_cols
        ]
    ).to_csv(
        output_root
        / "selective_mean_sd.csv",
        index=False,
    )

    (
        concept_all
        .groupby("concept", as_index=False)
        .agg(
            n=("n", "mean"),
            accuracy_mean=(
                "accuracy", "mean"
            ),
            accuracy_sd=(
                "accuracy", "std"
            ),
            macro_f1_mean=(
                "macro_f1", "mean"
            ),
            macro_f1_sd=(
                "macro_f1", "std"
            ),
            auroc_mean=(
                "auroc", "mean"
            ),
            auroc_sd=(
                "auroc", "std"
            ),
        )
        .to_csv(
            output_root
            / "concept_transfer_mean_sd.csv",
            index=False,
        )
    )

    print(
        "\n=== PH2 EXTERNAL VALIDATION COMPLETE ==="
    )
    print("saved:", output_root)


if __name__ == "__main__":
    main()

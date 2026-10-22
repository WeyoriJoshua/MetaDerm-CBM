from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


@dataclass
class GuardResult:
    status: str
    domain_score: float
    distance: float
    threshold: float
    reason: str
    quality_flags: list[str]
    reference_size: int
    gate_version: str = 'one-class-embedding-gate-v1.0'

    def to_dict(self):
        return asdict(self)


def _tensor_from_any(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ('features', 'feature', 'embedding', 'embeddings', 'x', 'out'):
            if key in value:
                t = _tensor_from_any(value[key])
                if t is not None:
                    return t
        for child in value.values():
            t = _tensor_from_any(child)
            if t is not None:
                return t
    if isinstance(value, (list, tuple)):
        for child in value:
            t = _tensor_from_any(child)
            if t is not None:
                return t
    return None


def _pool_feature_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = x.mean(dim=(2, 3))
    elif x.ndim == 3:
        x = x.mean(dim=1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 4:
        x = x.mean(dim=tuple(range(2, x.ndim)))
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    return torch.nn.functional.normalize(x.float(), p=2, dim=1)


def extract_embedding(artifacts, image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    x = artifacts.transform(image).unsqueeze(0).to(artifacts.device)
    backbone = artifacts.blackbox_model.backbone
    with torch.no_grad():
        if hasattr(backbone, 'forward_features'):
            out = backbone.forward_features(x)
        else:
            out = backbone(x)
    t = _tensor_from_any(out)
    if t is None:
        raise RuntimeError('Could not obtain backbone representation for input gate.')
    feat = _pool_feature_tensor(t)
    return feat[0].detach().cpu().numpy().astype(np.float32)


def _quality_flags(image: Image.Image) -> list[str]:
    flags = []
    w, h = image.size
    if min(w, h) < 128:
        flags.append('image_resolution_too_small')
    ratio = max(w, h) / max(1, min(w, h))
    if ratio > 3.5:
        flags.append('extreme_aspect_ratio')
    gray = image.convert('L').resize((256, 256))
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    mean, std = float(arr.mean()), float(arr.std())
    if mean < 0.025:
        flags.append('image_almost_black')
    if mean > 0.975:
        flags.append('image_almost_white')
    if std < 0.018:
        flags.append('very_low_visual_information')
    return flags


def _candidate_reference_images(dataset_root: Path) -> List[Path]:
    paths = [
        p for p in dataset_root.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise FileNotFoundError(f'No reference images found below {dataset_root}')
    derm = [p for p in paths if any(tok in str(p).lower() for tok in ('dermosc', 'derm'))]
    if len(derm) >= 100:
        paths = derm
    return sorted(paths)


def _sample(paths: List[Path], n: int) -> List[Path]:
    if len(paths) <= n:
        return paths
    idx = np.linspace(0, len(paths) - 1, n, dtype=int)
    return [paths[i] for i in idx]


def _knn_self_distances(features: np.ndarray, k: int) -> np.ndarray:
    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, features.shape[0] - 1)
    top = np.partition(sim, -k, axis=1)[:, -k:]
    return 1.0 - top.mean(axis=1)


def build_or_load_reference(
    artifacts,
    derm7pt_root: Path,
    cache_path: Path,
    max_reference: int = 256,
    k: int = 5,
    quantile: float = 0.995,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        return {
            'features': d['features'],
            'threshold': float(d['threshold'][0]),
            'k': int(d['k'][0]),
            'quantile': float(d['quantile'][0]),
        }

    chosen = _sample(_candidate_reference_images(Path(derm7pt_root)), max_reference)
    feats, used = [], []
    for path in chosen:
        try:
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                if _quality_flags(rgb):
                    continue
                feats.append(extract_embedding(artifacts, rgb))
                used.append(str(path))
        except Exception:
            continue

    if len(feats) < 50:
        raise RuntimeError(f'Only {len(feats)} usable reference images were found; at least 50 are required.')

    features = np.stack(feats).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True).clip(1e-12)
    self_dist = _knn_self_distances(features, k)
    threshold = float(np.quantile(self_dist, quantile))

    np.savez_compressed(
        cache_path,
        features=features,
        threshold=np.asarray([threshold], dtype=np.float32),
        k=np.asarray([k], dtype=np.int32),
        quantile=np.asarray([quantile], dtype=np.float32),
    )
    manifest = {
        'gate_version': 'one-class-embedding-gate-v1.0',
        'reference_dataset': 'Derm7pt',
        'reference_images_used': len(used),
        'k': k,
        'threshold_quantile': quantile,
        'threshold': threshold,
        'note': 'Post-publication defensive system layer; not part of locked diagnostic experiments.',
    }
    cache_path.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return {'features': features, 'threshold': threshold, 'k': k, 'quantile': quantile}


def assess_input(artifacts, image: Image.Image, reference: dict) -> GuardResult:
    flags = _quality_flags(image)
    if flags:
        return GuardResult(
            status='REJECT', domain_score=0.0, distance=float('inf'),
            threshold=float(reference['threshold']),
            reason='The image failed basic suitability checks. No diagnostic output was generated.',
            quality_flags=flags, reference_size=int(reference['features'].shape[0]),
        )

    q = extract_embedding(artifacts, image)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    ref = reference['features']
    sims = ref @ q
    k = min(int(reference['k']), len(sims))
    top = np.partition(sims, -k)[-k:]
    distance = float(1.0 - top.mean())
    threshold = float(reference['threshold'])
    ratio = distance / max(threshold, 1e-8)
    domain_score = float(np.clip(1.0 - 0.5 * ratio, 0.0, 1.0))

    if distance > threshold:
        return GuardResult(
            status='REJECT', domain_score=domain_score, distance=distance,
            threshold=threshold,
            reason='The image lies outside the accepted Derm7pt representation envelope. Diagnostic outputs were withheld.',
            quality_flags=[], reference_size=int(ref.shape[0]),
        )
    if distance > 0.90 * threshold:
        return GuardResult(
            status='REVIEW', domain_score=domain_score, distance=distance,
            threshold=threshold,
            reason='The image is close to the dermoscopic-domain boundary. Verify the input before diagnostic analysis.',
            quality_flags=[], reference_size=int(ref.shape[0]),
        )
    return GuardResult(
        status='PASS', domain_score=domain_score, distance=distance,
        threshold=threshold,
        reason='The image passed the one-class dermoscopic-domain suitability gate.',
        quality_flags=[], reference_size=int(ref.shape[0]),
    )

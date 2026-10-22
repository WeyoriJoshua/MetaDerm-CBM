from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from metaderm.config import load_config
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.reasoning.seven_point import SoftSevenPointReasoner
from metaderm.reasoning.consistency import build_profile_map, hard_profile_from_probs, lookup_profile_reliability
from metaderm.reasoning.metacognition import metacognitive_features, reliability_score, assign_actions

CONCEPTS = [
    'pigment_network', 'streaks', 'pigmentation', 'regression_structures',
    'dots_and_globules', 'blue_whitish_veil', 'vascular_structures'
]

@dataclass
class FrozenArtifacts:
    cfg: dict
    concept_model: torch.nn.Module
    blackbox_model: torch.nn.Module
    reasoner: object
    concept_temperatures: Dict[str, float]
    blackbox_temperature: float
    tau_low: float
    tau_high: float
    profile_map: dict
    device: torch.device
    transform: object


def _normalize_key(k: str) -> str:
    k = str(k)
    for prefix in ('module.', 'model.', 'net.'):
        if k.startswith(prefix):
            k = k[len(prefix):]
    return k


def _extract_state_dict(obj, model):
    target = model.state_dict()
    target_keys = set(target.keys())
    candidates = []

    def walk(value, path=()):
        if not isinstance(value, dict):
            return
        tensors = {_normalize_key(k): v for k, v in value.items() if torch.is_tensor(v)}
        if tensors:
            candidates.append((len(set(tensors) & target_keys), path, tensors))
        for k, child in value.items():
            if isinstance(child, dict):
                walk(child, path + (str(k),))

    walk(obj)
    if not candidates:
        raise RuntimeError('No tensor state_dict found in checkpoint.')
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, path, state = candidates[0]
    compatible = {k: v for k, v in state.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    coverage = len(compatible) / max(1, len(target_keys))
    if coverage < 0.95:
        raise RuntimeError(f'Checkpoint coverage too low ({coverage:.1%}) at {".".join(path) if path else "<root>"}.')
    return compatible


def _load_checkpoint(model, path: Path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(obj, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected checkpoint keys for {path.name}: {unexpected[:10]}')
    if len(missing) > max(2, int(0.05 * len(model.state_dict()))):
        raise RuntimeError(f'Too many missing checkpoint keys for {path.name}: {len(missing)}')


def _calibration_values(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    concept_t = data.get('concept_temperatures') or data.get('concept_temperature') or data.get('concepts')
    if not isinstance(concept_t, dict):
        raise KeyError(f'Concept temperatures not found in {path}')
    bb = data.get('blackbox_temperature') or data.get('diagnosis_temperature') or data.get('blackbox')
    if isinstance(bb, dict):
        bb = bb.get('temperature')
    if bb is None:
        raise KeyError(f'Black-box temperature not found in {path}')
    return {k: float(v) for k, v in concept_t.items()}, float(bb)


def _eval_transform(cfg):
    data = cfg.get('data', {})
    size = int(data.get('image_size', 300))
    mean = data.get('mean', [0.485, 0.456, 0.406])
    std = data.get('std', [0.229, 0.224, 0.225])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])


def _read_split_indices(path: Path, index_col=None):
    df = pd.read_csv(path)
    values = df[index_col].tolist() if index_col and index_col in df.columns else df.to_numpy().ravel().tolist()
    idx = []
    for value in values:
        try:
            idx.append(int(value))
        except Exception:
            pass
    if len(idx) < 100:
        raw = pd.read_csv(path, header=None)
        idx2 = []
        for value in raw.to_numpy().ravel().tolist():
            try:
                idx2.append(int(value))
            except Exception:
                pass
        if len(idx2) > len(idx):
            idx = idx2
    return idx


def _load_derm7pt_training_metadata(derm7pt_root: Path, cfg):
    data = cfg['data']
    meta_path = Path(data['meta_csv'])
    if not meta_path.is_absolute():
        meta_path = derm7pt_root / meta_path
    if not meta_path.exists():
        alt = derm7pt_root / 'meta' / 'meta.csv'
        if alt.exists():
            meta_path = alt

    split_path = Path(data['split_files']['train'])
    if not split_path.is_absolute():
        split_path = derm7pt_root / split_path
    if not split_path.exists():
        alt = derm7pt_root / 'meta' / Path(data['split_files']['train']).name
        if alt.exists():
            split_path = alt

    meta = pd.read_csv(meta_path)
    if 'binary_label' not in meta.columns:
        diagnosis_col = data['diagnosis_col']
        prefix = str(data.get('diagnosis_positive_prefix', 'melanoma')).strip().lower()
        meta['binary_label'] = (
            meta[diagnosis_col].fillna('').astype(str).str.strip().str.lower().str.startswith(prefix).astype(int)
        )

    idx = [i for i in _read_split_indices(split_path, data.get('split_index_col')) if 0 <= i < len(meta)]
    if not idx:
        raise RuntimeError('Could not recover Derm7pt training indices.')
    return meta.iloc[idx].copy().reset_index(drop=True)


def _unpack_concepts(output):
    if isinstance(output, dict):
        if all(c in output for c in CONCEPTS):
            return {c: output[c] for c in CONCEPTS}
        for key in ('concept_logits', 'logits', 'outputs', 'concepts'):
            val = output.get(key)
            if isinstance(val, dict) and all(c in val for c in CONCEPTS):
                return {c: val[c] for c in CONCEPTS}
    if isinstance(output, (list, tuple)):
        if len(output) == len(CONCEPTS) and all(torch.is_tensor(v) for v in output):
            return dict(zip(CONCEPTS, output))
        for item in output:
            try:
                return _unpack_concepts(item)
            except Exception:
                pass
    raise TypeError('Could not identify concept logits.')


def _unpack_blackbox(output):
    if torch.is_tensor(output):
        if output.ndim == 1 or (output.ndim == 2 and output.shape[1] in (1, 2)):
            return output
    candidates = []
    def walk(v, path=()):
        if torch.is_tensor(v):
            if v.ndim == 1 or (v.ndim == 2 and v.shape[1] in (1, 2)):
                candidates.append((path, v))
        elif isinstance(v, dict):
            for k, child in v.items():
                walk(child, path + (str(k),))
        elif isinstance(v, (list, tuple)):
            for i, child in enumerate(v):
                walk(child, path + (str(i),))
    walk(output)
    if not candidates:
        raise TypeError('Could not identify black-box logits.')
    preferred = ('logit', 'diagnos', 'class', 'pred', 'output')
    candidates.sort(key=lambda item: sum(term in '.'.join(item[0]).lower() for term in preferred), reverse=True)
    return candidates[0][1]


def _binary_probability(logits, temperature):
    logits = logits / float(temperature)
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    raise ValueError(f'Unexpected logits shape: {tuple(logits.shape)}')


def _reasoner_probability(reasoner, concept_probs):
    output = reasoner.predict(concept_probs)
    if isinstance(output, dict):
        for key in ('probability', 'prob', 'melanoma_probability', 'score'):
            if key in output:
                return np.asarray(output[key], dtype=float)
    if isinstance(output, np.ndarray):
        return output.astype(float)
    raise TypeError('Could not extract reasoner probability.')


def _profile_reliability(artifacts, concept_probs):
    profiles = hard_profile_from_probs(concept_probs)
    unseen = artifacts.cfg.get('metacognition', {}).get('unseen_profile_reliability', 0.0)
    return lookup_profile_reliability(profiles, artifacts.profile_map, unseen)


def _reliability_and_action(artifacts, concept_probs, blackbox_prob, reason_prob):
    profile_rel = _profile_reliability(artifacts, concept_probs)
    features = metacognitive_features(concept_probs, blackbox_prob, reason_prob, profile_rel)
    reliability = reliability_score(features, artifacts.cfg)
    actions = assign_actions(reliability, artifacts.tau_low, artifacts.tau_high)
    return float(np.asarray(reliability).reshape(-1)[0]), str(np.asarray(actions).reshape(-1)[0]), float(np.asarray(profile_rel).reshape(-1)[0])


def load_frozen_artifacts(repo_root: Path, derm7pt_root: Path, seed: int = 2026) -> FrozenArtifacts:
    repo_root = Path(repo_root).resolve()
    derm7pt_root = Path(derm7pt_root).resolve()
    cfg = load_config(str(repo_root / 'configs' / 'base.yaml'), str(repo_root / 'configs' / 'final_design.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    concept_model = ConceptPerceptionModel(cfg).to(device)
    blackbox_model = BlackBoxDiagnosisModel(cfg).to(device)
    checkpoint_root = repo_root / 'checkpoints' / 'efficientnet_b3_final'
    _load_checkpoint(concept_model, checkpoint_root / f'concepts_seed{seed}_best.pt', device)
    _load_checkpoint(blackbox_model, checkpoint_root / f'blackbox_seed{seed}_best.pt', device)
    concept_model.eval(); blackbox_model.eval()

    calibration_path = repo_root / 'results' / 'efficientnet_b3_final' / 'calibration' / f'calibration_seed{seed}.json'
    concept_t, blackbox_t = _calibration_values(calibration_path)
    metrics_path = repo_root / 'results' / 'efficientnet_b3_final' / 'final' / f'seed{seed}' / 'metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))

    train_meta = _load_derm7pt_training_metadata(derm7pt_root, cfg)
    profile_map = build_profile_map(train_meta, cfg)

    return FrozenArtifacts(
        cfg=cfg, concept_model=concept_model, blackbox_model=blackbox_model,
        reasoner=SoftSevenPointReasoner(cfg), concept_temperatures=concept_t,
        blackbox_temperature=blackbox_t, tau_low=float(metrics['tau_low']),
        tau_high=float(metrics['tau_high']), profile_map=profile_map,
        device=device, transform=_eval_transform(cfg)
    )


def infer_image(artifacts: FrozenArtifacts, image: Image.Image):
    image = image.convert('RGB')
    x = artifacts.transform(image).unsqueeze(0).to(artifacts.device)
    with torch.no_grad():
        c_out = _unpack_concepts(artifacts.concept_model(x))
        b_out = _unpack_blackbox(artifacts.blackbox_model(x))

    concept_probs = {}
    for concept in CONCEPTS:
        temp = artifacts.concept_temperatures[concept]
        concept_probs[concept] = torch.softmax(c_out[concept] / temp, dim=1).detach().cpu().numpy()

    blackbox_prob = _binary_probability(b_out, artifacts.blackbox_temperature).detach().cpu().numpy()
    reason_prob = _reasoner_probability(artifacts.reasoner, concept_probs)
    reliability, action, profile_rel = _reliability_and_action(artifacts, concept_probs, blackbox_prob, reason_prob)

    return {
        'blackbox_prob': float(blackbox_prob[0]),
        'reason_prob': float(reason_prob[0]),
        'reliability': reliability,
        'action': action,
        'profile_reliability': profile_rel,
        'concept_probs': concept_probs,
    }


def concept_state_labels(artifacts: FrozenArtifacts, concept: str) -> List[str]:
    return list(artifacts.cfg['data']['concepts'][concept])


def concept_summary_table(artifacts: FrozenArtifacts, concept_probs):
    rows = []
    for concept in CONCEPTS:
        labels = concept_state_labels(artifacts, concept)
        p = np.asarray(concept_probs[concept])[0]
        idx = int(np.argmax(p))
        entropy = float(-np.sum(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))) / np.log(len(p)))
        rows.append({'concept': concept, 'predicted_state': labels[idx], 'confidence': float(p[idx]), 'entropy': entropy})
    return pd.DataFrame(rows)


def _one_hot(n, idx):
    arr = np.zeros((1, n), dtype=float)
    arr[0, idx] = 1.0
    return arr


def apply_expert_corrections(artifacts, original_concept_probs, blackbox_prob: float, corrections):
    updated = {k: np.array(v, copy=True) for k, v in original_concept_probs.items()}
    for concept, chosen_state in corrections.items():
        if chosen_state in (None, '', '<keep model prediction>'):
            continue
        labels = concept_state_labels(artifacts, concept)
        updated[concept] = _one_hot(len(labels), labels.index(chosen_state))

    reason_prob = _reasoner_probability(artifacts.reasoner, updated)
    reliability, action, profile_rel = _reliability_and_action(
        artifacts, updated, np.asarray([blackbox_prob], dtype=float), reason_prob
    )
    return {'concept_probs': updated, 'reason_prob': float(reason_prob[0]), 'reliability': reliability, 'action': action, 'profile_reliability': profile_rel}


def concept_reasoning_impact(artifacts, concept_probs):
    base = float(_reasoner_probability(artifacts.reasoner, concept_probs)[0])
    rows = []
    for concept in CONCEPTS:
        labels = concept_state_labels(artifacts, concept)
        effects = []
        for idx, state in enumerate(labels):
            trial = {k: np.array(v, copy=True) for k, v in concept_probs.items()}
            trial[concept] = _one_hot(len(labels), idx)
            p = float(_reasoner_probability(artifacts.reasoner, trial)[0])
            effects.append({'state': state, 'reason_prob': p, 'delta': p - base})
        rows.append({'concept': concept, 'max_abs_reasoning_impact': max(abs(x['delta']) for x in effects), 'counterfactuals': effects})
    rows.sort(key=lambda x: x['max_abs_reasoning_impact'], reverse=True)
    return rows


def find_last_conv(backbone):
    last_name = None; last_module = None
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name; last_module = module
    if last_module is None:
        raise RuntimeError('No Conv2d layer found for Grad-CAM.')
    return last_name, last_module


def gradcam(artifacts: FrozenArtifacts, image: Image.Image):
    image = image.convert('RGB')
    x = artifacts.transform(image).unsqueeze(0).to(artifacts.device)
    x.requires_grad_(True)
    name, layer = find_last_conv(artifacts.blackbox_model.backbone)
    activations = []; gradients = []
    h1 = layer.register_forward_hook(lambda _m, _i, out: activations.append(out))
    h2 = layer.register_full_backward_hook(lambda _m, _gi, gout: gradients.append(gout[0]))
    artifacts.blackbox_model.zero_grad(set_to_none=True)
    logits = _unpack_blackbox(artifacts.blackbox_model(x))
    if logits.ndim == 1:
        raw_logit = logits[0]; score = logits.sum()
    elif logits.ndim == 2 and logits.shape[1] == 1:
        raw_logit = logits[0, 0]; score = logits[:, 0].sum()
    elif logits.ndim == 2 and logits.shape[1] == 2:
        margin = logits[:, 1] - logits[:, 0]; raw_logit = margin[0]; score = margin.sum()
    else:
        h1.remove(); h2.remove(); raise ValueError('Unexpected diagnosis-logit shape.')
    score.backward(); h1.remove(); h2.remove()
    a = activations[-1]; g = gradients[-1]
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * a).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy(); cam -= cam.min()
    if cam.max() > 0: cam /= cam.max()
    calibrated_prob = float(torch.sigmoid(raw_logit / artifacts.blackbox_temperature).detach().cpu().item())
    return {'cam': cam, 'layer': name, 'calibrated_prob': calibrated_prob}

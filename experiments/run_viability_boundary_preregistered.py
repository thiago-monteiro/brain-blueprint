from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, ttest_ind
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.eval.rdm import compute_rdm
from egomuscle.training.train import EgoMuscleLightningModule, apply_override, load_config

VIABILITY_TAXONOMY_PATH = Path("experiments/viability_taxonomy.json")

MODEL_VARIANTS = {
    "E2": [
        "model.video_model_name=MCG-NJU/videomae-base",
        "model.use_video=true",
        "model.use_muscle=true",
        "model.fusion_mode=late",
        "training.compile=false",
    ],
    "E3": [
        "model.video_model_name=MCG-NJU/videomae-base",
        "model.use_video=true",
        "model.use_muscle=true",
        "model.fusion_mode=cross_attn",
        "training.compile=false",
    ],
    "E4": [
        "model.video_model_name=MCG-NJU/videomae-base",
        "model.use_video=true",
        "model.use_muscle=false",
        "model.fusion_mode=late",
        "training.compile=false",
    ],
    "E5": [
        "model.video_model_name=MCG-NJU/videomae-base",
        "model.use_video=false",
        "model.use_muscle=true",
        "training.compile=false",
    ],
}

KINEMATIC_FEATURE_NAMES = [
    "com_vertical_displacement",
    "com_vertical_velocity",
    "hip_ankle_distance_min",
    "spine_angular_accel",
    "knee_angular_accel",
    "optical_flow_magnitude",
]


def load_taxonomy(taxonomy_path: Path) -> dict[str, Any]:
    return json.loads(taxonomy_path.read_text())


def flatten_labels(category_dict: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for cat_name, cat_content in category_dict.items():
        if isinstance(cat_content, dict) and "labels" in cat_content:
            labels.update(cat_content["labels"])
    return labels


def build_label_index(taxonomy: dict[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for cls in ("viability_relevant", "viability_neutral"):
        for cat_name, cat_content in taxonomy[cls].get("categories", {}).items():
            for label in cat_content.get("labels", []):
                index[label.lower()] = cls
    for label in taxonomy.get("excluded", {}).get("labels", []):
        index[label.lower()] = "excluded"
    return index


def keyword_match(labels: list[str], label_index: dict[str, str]) -> tuple[str | None, list[str]]:
    text = " ".join(labels).lower().replace("_", " ")
    tokens = set(re.findall(r"[a-z]+", text))
    for keyword, cls in sorted(label_index.items(), key=lambda x: -len(x[0].split())):
        normalized = keyword.lower()
        if " " in normalized:
            if normalized in text:
                return cls, [keyword]
        elif normalized in tokens:
            return cls, [keyword]
    return None, []


def extract_kinematic_features(clip_dir: Path, muscle_dir: Path | None, metadata_path: Path | None) -> dict[str, np.ndarray]:
    features: dict[str, list[float]] = {k: [] for k in KINEMATIC_FEATURE_NAMES}
    clip_ids: list[str] = []

    dataset = EgoMuscleDataset(
        clip_dir=clip_dir,
        muscle_dir=muscle_dir,
        metadata_path=metadata_path,
        n_frames=16,
        image_size=224,
        muscle_dim=32,
        require_muscle=True,
    )

    for record in dataset.records:
        clip_id = record.video_path.stem
        muscle_path = record.muscle_path
        if muscle_path is None or not muscle_path.exists():
            continue

        metadata = record.metadata or {}
        amass_path = metadata.get("amass_path") if isinstance(metadata, dict) else None
        if not amass_path or not Path(amass_path).exists():
            continue

        from egomuscle.data.amass_smpl import load_amass_motion
        try:
            smpl_data = load_amass_motion(amass_path)
        except Exception:
            continue

        if smpl_data is None:
            continue

        clip_ids.append(clip_id)

        betas = smpl_data.get("betas", np.zeros((16, 16)))
        poses = smpl_data.get("poses", np.zeros((16, 156)))

        com_y = poses[:, 1] if poses.shape[1] > 1 else np.zeros(16)
        features["com_vertical_displacement"].append(float(np.ptp(com_y)))
        features["com_vertical_velocity"].append(float(np.mean(np.abs(np.diff(com_y)))) if len(com_y) > 1 else 0.0)

        lhip = poses[:, 6:9] if poses.shape[1] >= 9 else np.zeros((16, 3))
        rhip = poses[:, 12:15] if poses.shape[1] >= 15 else np.zeros((16, 3))
        lankle = poses[:, 24:27] if poses.shape[1] >= 27 else np.zeros((16, 3))
        rankle = poses[:, 30:33] if poses.shape[1] >= 33 else np.zeros((16, 3))

        l_dist = np.mean(np.linalg.norm(lhip - lankle, axis=1))
        r_dist = np.mean(np.linalg.norm(rhip - rankle, axis=1))
        features["hip_ankle_distance_min"].append(float(min(l_dist, r_dist)))

        spine_angle = poses[:, 3:6] if poses.shape[1] >= 6 else np.zeros((16, 3))
        spine_accel = np.mean(np.abs(np.diff(spine_angle, axis=0))) if spine_angle.shape[0] > 1 else 0.0
        features["spine_angular_accel"].append(float(spine_accel))

        lknee = poses[:, 18:21] if poses.shape[1] >= 21 else np.zeros((16, 3))
        rknee = poses[:, 36:39] if poses.shape[1] >= 39 else np.zeros((16, 3))
        lknee_accel = np.mean(np.abs(np.diff(lknee, axis=0))) if lknee.shape[0] > 1 else 0.0
        rknee_accel = np.mean(np.abs(np.diff(rknee, axis=0))) if rknee.shape[0] > 1 else 0.0
        features["knee_angular_accel"].append(float((lknee_accel + rknee_accel) / 2.0))

        features["optical_flow_magnitude"].append(0.0)

    if not clip_ids:
        return {k: np.array([]) for k in KINEMATIC_FEATURE_NAMES}

    return {k: np.array(v) for k, v in features.items()}


def propensity_score_match(
    relevant_ids: list[str],
    neutral_ids: list[str],
    kinematics_relevant: dict[str, np.ndarray],
    kinematics_neutral: dict[str, np.ndarray],
    caliper: float = 0.25,
) -> list[tuple[str, str]]:
    if not relevant_ids or not neutral_ids:
        return []

    feat_matrix_relevant = np.column_stack([
        kinematics_relevant[k] for k in KINEMATIC_FEATURE_NAMES
        if len(kinematics_relevant[k]) == len(relevant_ids)
    ])
    feat_matrix_neutral = np.column_stack([
        kinematics_neutral[k] for k in KINEMATIC_FEATURE_NAMES
        if len(kinematics_neutral[k]) == len(neutral_ids)
    ])

    if feat_matrix_relevant.size == 0 or feat_matrix_neutral.size == 0:
        return []

    r_mean = feat_matrix_relevant.mean(axis=0, keepdims=True)
    r_std = feat_matrix_neutral.std(axis=0, keepdims=True)
    r_std[r_std < 1e-8] = 1.0
    feat_relevant = (feat_matrix_relevant - r_mean) / r_std
    feat_neutral = (feat_matrix_neutral - r_mean) / r_std

    dist_matrix = cdist(feat_relevant, feat_neutral, metric="euclidean")

    matches: list[tuple[str, str]] = []
    matched_neutral: set[int] = set()

    for i in range(len(relevant_ids)):
        distances = dist_matrix[i]
        for j in np.argsort(distances):
            if j in matched_neutral:
                continue
            if distances[j] > caliper * feat_neutral.shape[1]:
                continue
            matches.append((relevant_ids[i], neutral_ids[j]))
            matched_neutral.add(j)
            break

    return matches


def classify_clips(
    taxonomy: dict[str, Any],
    label_index: dict[str, str],
    dataset_root: Path,
    max_clips_per_class: int | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[int]]:
    candidates: list[dict[str, Any]] = []
    offset = 0

    for split in ("train", "val", "test"):
        split_root = dataset_root / split
        clip_dir = split_root / "clips"
        if not clip_dir.exists():
            continue
        try:
            ds = EgoMuscleDataset(
                clip_dir=clip_dir,
                muscle_dir=split_root / "muscles",
                metadata_path=split_root / "metadata.json",
                n_frames=16,
                image_size=224,
                muscle_dim=32,
                require_muscle=False,
                frame_cache_dir=None,
                write_frame_cache=False,
            )
        except (ValueError, FileNotFoundError):
            continue

        for idx, record in enumerate(ds.records):
            metadata = record.metadata or {}
            labels_raw = []
            for value in metadata.values():
                if isinstance(value, str):
                    labels_raw.append(value)
                elif isinstance(value, (list, tuple)):
                    labels_raw.extend(str(v) for v in value)
            cls, hits = keyword_match(labels_raw, label_index)
            candidates.append({
                "global_index": offset + idx,
                "split": split,
                "clip_id": record.video_path.stem,
                "class": cls or "unclassified",
                "labels": "|".join(labels_raw[:20]),
                "matched_keywords": "|".join(hits),
                "dataset_idx": idx,
            })
        offset += len(ds)

    relevant = [c for c in candidates if c["class"] == "viability_relevant"]
    neutral = [c for c in candidates if c["class"] == "viability_neutral"]

    rng = np.random.default_rng(seed)
    if max_clips_per_class is not None:
        if len(relevant) > max_clips_per_class:
            relevant = [relevant[i] for i in rng.choice(len(relevant), size=max_clips_per_class, replace=False)]
        if len(neutral) > max_clips_per_class:
            neutral = [neutral[i] for i in rng.choice(len(neutral), size=max_clips_per_class, replace=False)]

    selected_indices = [c["global_index"] for c in relevant] + [c["global_index"] for c in neutral]
    for c in candidates:
        c["selected"] = c["global_index"] in selected_indices

    return candidates, selected_indices


def load_model_module(
    checkpoint_path: Path,
    config_path: Path,
    overrides: list[str],
    device: torch.device,
) -> EgoMuscleLightningModule:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("hyper_parameters")
    if not isinstance(config, dict):
        config = load_config(config_path)
    else:
        config = copy.deepcopy(config)
    for override in overrides:
        apply_override(config, override)
    config.setdefault("training", {})["compile"] = False

    module = EgoMuscleLightningModule(config)
    state_dict = payload.get("state_dict", payload)
    stripped = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
    try:
        module.load_state_dict(stripped, strict=False)
    except RuntimeError:
        try:
            prefixed = {
                (k.replace("model.", "model._orig_mod.", 1) if k.startswith("model.") and not k.startswith("model._orig_mod.") else k): v
                for k, v in state_dict.items()
            }
            module.load_state_dict(prefixed, strict=False)
        except RuntimeError:
            pass
    module.to(device)
    module.eval()
    return module


def transfer_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    frames = batch["frames"].to(device, non_blocking=True)
    if frames.dtype == torch.uint8:
        from egomuscle.data.dataset import IMAGE_MEAN, IMAGE_STD
        frames = frames.float().div_(255.0)
        frames = (frames - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device)
    moved["frames"] = frames
    moved["activity_id"] = moved["activity_id"].to(device, non_blocking=True)
    if "muscle" in batch and batch["muscle"] is not None:
        moved["muscle"] = batch["muscle"].to(device, non_blocking=True)
    else:
        moved["muscle"] = None
    return moved


def collect_pooled(
    module: EgoMuscleLightningModule,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[str], list[str]]:
    features: list[torch.Tensor] = []
    classes: list[str] = []
    clip_ids: list[str] = []
    module.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = transfer_to_device(batch, device)
            outputs = module.model(
                frames=batch.get("frames"),
                muscle=batch.get("muscle"),
                activity_ids=batch["activity_id"],
                mask_ratio=0.0,
            )
            features.append(outputs.pooled.detach().cpu())
            clip_ids.extend(batch["clip_id"])
    return torch.cat(features, dim=0).numpy(), classes, clip_ids


def compute_cohens_d(
    rdm: np.ndarray,
    relevant_idx: list[int],
    neutral_idx: list[int],
) -> dict[str, float]:
    if len(relevant_idx) < 2 or len(neutral_idx) < 2:
        return {
            "d": float("nan"),
            "across_boundary_distance": float("nan"),
            "within_relevant": float("nan"),
            "within_neutral": float("nan"),
            "pooled_within_sd": float("nan"),
        }

    across = rdm[np.ix_(relevant_idx, neutral_idx)]
    across_mean = float(across.mean())

    rel_sub = rdm[np.ix_(relevant_idx, relevant_idx)]
    rel_tri = rel_sub[np.triu_indices(len(relevant_idx), k=1)]
    within_rel_mean = float(rel_tri.mean())

    neu_sub = rdm[np.ix_(neutral_idx, neutral_idx)]
    neu_tri = neu_sub[np.triu_indices(len(neutral_idx), k=1)]
    within_neu_mean = float(neu_tri.mean())

    pooled_within = np.concatenate([rel_tri, neu_tri])
    pooled_within_sd = float(pooled_within.std(ddof=1)) if len(pooled_within) > 1 else float("nan")

    effect = across_mean - float(np.nanmean([within_rel_mean, within_neu_mean]))
    d = effect / pooled_within_sd if pooled_within_sd > 0 and not math.isnan(pooled_within_sd) else float("nan")

    return {
        "d": d,
        "across_boundary_distance": across_mean,
        "within_relevant": within_rel_mean,
        "within_neutral": within_neu_mean,
        "pooled_within_sd": pooled_within_sd,
        "effect": effect,
    }


def permutation_test_cohens_d(
    rdm: np.ndarray,
    relevant_idx: list[int],
    neutral_idx: list[int],
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    observed = compute_cohens_d(rdm, relevant_idx, neutral_idx)["d"]
    if math.isnan(observed):
        return {"p": float("nan"), "null_mean": float("nan"), "n_permutations": n_permutations}

    all_idx = relevant_idx + neutral_idx
    labels = np.array(["rel"] * len(relevant_idx) + ["neu"] * len(neutral_idx), dtype=object)
    n_rel = len(relevant_idx)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        perm = rng.permutation(len(labels))
        perm_rel = [all_idx[j] for j in range(len(labels)) if labels[perm[j]] == "rel"]
        perm_neu = [all_idx[j] for j in range(len(labels)) if labels[perm[j]] == "neu"]
        null[i] = compute_cohens_d(rdm, perm_rel, perm_neu)["d"]

    p = (float(np.sum(null >= observed)) + 1.0) / (float(n_permutations) + 1.0)
    return {
        "p": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "n_permutations": n_permutations,
    }


def bonferroni_holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = min(1.0, p_values[idx] * (m - rank))
    for i in range(m - 2, -1, -1):
        adjusted[sorted_indices[i]] = min(adjusted[sorted_indices[i]], adjusted[sorted_indices[i + 1]])
    return adjusted


def run_variant_analysis(
    variant_name: str,
    checkpoint: Path,
    config_path: Path,
    dataset: ConcatDataset,
    selected_indices: list[int],
    class_by_index: dict[int, str],
    n_folds: int,
    n_permutations: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, Any]:
    overrides = MODEL_VARIANTS[variant_name]
    module = load_model_module(checkpoint, config_path, overrides, device)

    data_indices = np.array(selected_indices)
    labels = np.array([class_by_index[i] for i in selected_indices])

    relevant_idx = np.where(labels == "viability_relevant")[0]
    neutral_idx = np.where(labels == "viability_neutral")[0]
    fold_labels = np.array(["rel"] * len(relevant_idx) + ["neu"] * len(neutral_idx))
    fold_indices = np.concatenate([relevant_idx, neutral_idx])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results: list[dict[str, Any]] = []
    all_features: list[np.ndarray] = []
    all_fold_ids: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(fold_indices, fold_labels)):
        train_data_idx = fold_indices[val_idx]
        subset = Subset(dataset, [int(data_indices[i]) for i in train_data_idx])
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_egomuscle,
        )

        features, _, clip_ids = collect_pooled(module, loader, device)
        all_features.append(features)
        all_fold_ids.extend([fold] * len(features))

        val_data_indices = train_data_idx
        rel_in_fold = [i for i in range(len(val_data_indices)) if labels[val_data_indices[i]] == "viability_relevant"]
        neu_in_fold = [i for i in range(len(val_data_indices)) if labels[val_data_indices[i]] == "viability_neutral"]

        if len(rel_in_fold) < 2 or len(neu_in_fold) < 2:
            fold_results.append({
                "fold": fold,
                "n_relevant": len(rel_in_fold),
                "n_neutral": len(neu_in_fold),
                "error": "insufficient_samples",
                "d": float("nan"),
                "p": float("nan"),
            })
            continue

        rdm = compute_rdm(features)
        d_metrics = compute_cohens_d(rdm, rel_in_fold, neu_in_fold)
        perm_results = permutation_test_cohens_d(rdm, rel_in_fold, neu_in_fold, n_permutations=n_permutations, seed=seed + fold)

        fold_results.append({
            "fold": fold,
            "n_relevant": len(rel_in_fold),
            "n_neutral": len(neu_in_fold),
            **d_metrics,
            **perm_results,
        })

    combined_features = np.concatenate(all_features, axis=0) if all_features else np.array([])
    d_values = [r["d"] for r in fold_results if not math.isnan(r.get("d", float("nan")))]
    p_values = [r["p"] for r in fold_results if not math.isnan(r.get("p", float("nan")))]

    mean_d = float(np.mean(d_values)) if d_values else float("nan")
    std_d = float(np.std(d_values, ddof=1)) if len(d_values) > 1 else float("nan")
    adjusted_p = bonferroni_holm(p_values) if p_values else []

    return {
        "variant": variant_name,
        "checkpoint": str(checkpoint),
        "n_folds": n_folds,
        "fold_results": fold_results,
        "mean_d": mean_d,
        "std_d": std_d,
        "mean_p": float(np.mean(p_values)) if p_values else float("nan"),
        "min_p": float(np.min(p_values)) if p_values else float("nan"),
        "adjusted_p": adjusted_p,
        "n_total": len(selected_indices),
        "n_relevant_total": int(np.sum(labels == "viability_relevant")),
        "n_neutral_total": int(np.sum(labels == "viability_neutral")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-registered viability boundary experiment with kinematic matching and cross-validation.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--taxonomy", type=Path, default=VIABILITY_TAXONOMY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/viability_boundary_preregistered"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-clips-per-class", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=list(MODEL_VARIANTS.keys()), action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = args.variant or list(MODEL_VARIANTS.keys())

    taxonomy = load_taxonomy(args.taxonomy)
    label_index = build_label_index(taxonomy)

    print("Classifying clips...")
    candidates, selected_indices = classify_clips(
        taxonomy, label_index, args.dataset_root,
        max_clips_per_class=args.max_clips_per_class, seed=args.seed,
    )
    class_by_index = {c["global_index"]: c["class"] for c in candidates}

    if not selected_indices:
        raise ValueError("No clips selected. Check taxonomy and dataset path.")

    datasets = []
    for split in ("train", "val", "test"):
        split_root = args.dataset_root / split
        if (split_root / "clips").exists():
            try:
                ds = EgoMuscleDataset(
                    clip_dir=split_root / "clips",
                    muscle_dir=split_root / "muscles",
                    metadata_path=split_root / "metadata.json",
                    n_frames=16,
                    image_size=224,
                    muscle_dim=32,
                    require_muscle=False,
                    frame_cache_dir=None,
                    write_frame_cache=False,
                )
                datasets.append(ds)
            except (ValueError, FileNotFoundError):
                continue

    if not datasets:
        raise FileNotFoundError(f"No dataset splits found under {args.dataset_root}")
    full_dataset = ConcatDataset(datasets)

    kinematics_relevant: dict[str, np.ndarray] = {k: np.array([]) for k in KINEMATIC_FEATURE_NAMES}
    kinematics_neutral: dict[str, np.ndarray] = {k: np.array([]) for k in KINEMATIC_FEATURE_NAMES}

    print("Extracting kinematic features for propensity matching...")
    for split in ("train", "val", "test"):
        split_root = args.dataset_root / split
        clip_dir = split_root / "clips"
        muscle_dir = split_root / "muscles"
        meta_path = split_root / "metadata.json"
        if not clip_dir.exists():
            continue
        try:
            kf = extract_kinematic_features(clip_dir, muscle_dir if muscle_dir.exists() else None, meta_path if meta_path.exists() else None)
        except Exception as exc:
            print(f"  Skipping kinematics for {split}: {exc}")
            continue
        if kf and len(next(iter(kf.values()))) > 0:
            kinematics_relevant = kf
            kinematics_neutral = kf

    relevant_ids = [c["clip_id"] for c in candidates if c["selected"] and c["class"] == "viability_relevant"]
    neutral_ids = [c["clip_id"] for c in candidates if c["selected"] and c["class"] == "viability_neutral"]

    print(f"Relevant clips: {len(relevant_ids)}, Neutral clips: {len(neutral_ids)}")

    print("Running variant analyses...")
    all_results: dict[str, Any] = {}
    for variant in variants:
        print(f"  Variant: {variant}")
        result = run_variant_analysis(
            variant, args.checkpoint, args.config,
            full_dataset, selected_indices, class_by_index,
            n_folds=args.n_folds, n_permutations=args.n_permutations,
            device=device, batch_size=args.batch_size,
            num_workers=args.num_workers, seed=args.seed,
        )
        all_results[variant] = result
        (args.output_dir / f"{variant}.json").write_text(json.dumps(result, indent=2))

    summary = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "dataset_root": str(args.dataset_root),
            "taxonomy": str(args.taxonomy),
            "n_folds": args.n_folds,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
        },
        "dataset_stats": {
            "n_total_selected": len(selected_indices),
            "n_viability_relevant": int(np.sum([c["class"] == "viability_relevant" for c in candidates if c["selected"]])),
            "n_viability_neutral": int(np.sum([c["class"] == "viability_neutral" for c in candidates if c["selected"]])),
        },
        "variants": {},
    }

    for variant in variants:
        r = all_results[variant]
        summary["variants"][variant] = {
            "mean_d": r["mean_d"],
            "std_d": r["std_d"],
            "mean_p": r["mean_p"],
            "min_p": r["min_p"],
            "adjusted_p": r["adjusted_p"],
            "n_folds_completed": sum(1 for f in r["fold_results"] if not f.get("error")),
        }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_partition_manifest(candidates, args.output_dir / "partition_manifest.csv")
    print(json.dumps(summary, indent=2))
    print("Done.")


def write_partition_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["global_index", "split", "clip_id", "class", "selected", "labels", "matched_keywords"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


if __name__ == "__main__":
    main()

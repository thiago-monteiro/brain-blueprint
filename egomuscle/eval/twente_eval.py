from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.training.train import EgoMuscleLightningModule, apply_override, load_config
from .probes import muscle_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained EgoMuscle checkpoint on real Twente EMG.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--twente-root", type=Path, default=Path("data/processed_real/twente"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/twente_eval.json"))
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-cache-dir", type=Path, default=None)
    parser.add_argument("--target-mode", choices=["mean", "flatten"], default="mean")
    parser.add_argument("--override", action="append", default=[], help="Config override, repeated as needed.")
    return parser.parse_args()


def load_lightning_module(checkpoint_path: Path, config_path: Path, overrides: list[str], device: torch.device) -> EgoMuscleLightningModule:
    payload = torch.load(checkpoint_path, map_location=device)
    if "hyper_parameters" in payload and isinstance(payload["hyper_parameters"], dict):
        import copy
        config = copy.deepcopy(payload["hyper_parameters"])
    else:
        config = load_config(config_path)

    for override in overrides:
        apply_override(config, override)
    state_dict = payload.get("state_dict", payload)

    def strip_compile_segments(candidate: dict[str, Any]) -> dict[str, Any]:
        return {key.replace("._orig_mod", ""): value for key, value in candidate.items()}

    def normalize_legacy_videomae_biases(candidate: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in candidate.items():
            if ".attention.attention.query.bias" in key:
                normalized[key.replace(".attention.attention.query.bias", ".attention.attention.q_bias")] = value
            elif ".attention.attention.value.bias" in key:
                normalized[key.replace(".attention.attention.value.bias", ".attention.attention.v_bias")] = value
            elif ".attention.attention.key.bias" in key:
                continue
            else:
                normalized[key] = value
        return normalized

    normalized_state_dict = strip_compile_segments(state_dict)
    legacy_bias_state_dict = normalize_legacy_videomae_biases(normalized_state_dict)
    model_cfg = config.setdefault("model", {})
    video_name = str(model_cfg.get("video_model_name", ""))
    has_v1_video_keys = any(".video_encoder.encoder.encoder.layer." in key for key in normalized_state_dict)
    if has_v1_video_keys and video_name.startswith("OpenGVLab/VideoMAEv2-"):
        projection = normalized_state_dict.get("model.video_encoder.encoder.embeddings.patch_embeddings.projection.weight")
        hidden_dim = int(projection.shape[0]) if projection is not None and hasattr(projection, "shape") else None
        v1_by_hidden = {
            384: "MCG-NJU/videomae-small",
            768: "MCG-NJU/videomae-base",
            1024: "MCG-NJU/videomae-large",
        }
        if hidden_dim in v1_by_hidden:
            model_cfg["video_model_name"] = v1_by_hidden[hidden_dim]

    # Eval should not require re-creating torch.compile wrappers. Checkpoints
    # saved from compiled training contain ``._orig_mod`` segments in state keys;
    # those are normalized below before strict loading into an uncompiled module.
    config.setdefault("training", {})["compile"] = False
    module = EgoMuscleLightningModule(config)

    candidate_state_dicts: list[tuple[str, dict[str, Any]]] = [
        ("raw", state_dict),
        ("strip_compile_segments", normalized_state_dict),
        ("legacy_videomae_bias_names", legacy_bias_state_dict),
    ]

    errors: list[str] = []
    for name, candidate in candidate_state_dicts:
        try:
            module.load_state_dict(candidate, strict=True)
            break
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")
            if name == "legacy_videomae_bias_names":
                current = module.state_dict()
                shape_mismatches = [
                    key
                    for key, value in candidate.items()
                    if key in current and hasattr(value, "shape") and tuple(value.shape) != tuple(current[key].shape)
                ]
                if shape_mismatches:
                    errors.append(
                        "legacy_videomae_bias_names_non_strict: shape mismatches for "
                        + ", ".join(shape_mismatches[:8])
                    )
                    continue
                incompatible = module.load_state_dict(candidate, strict=False)
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_missing = {
                    "model.video_encoder.encoder.layernorm.weight",
                    "model.video_encoder.encoder.layernorm.bias",
                }
                if not unexpected and set(missing).issubset(allowed_missing):
                    break
    else:
        joined = "\n\n".join(errors)
        raise RuntimeError(f"Could not load checkpoint {checkpoint_path} into Twente eval module.\n\n{joined}")

    module.to(device)
    module.eval()
    return module


def build_dataset(root: Path, *, n_frames: int, image_size: int, frame_cache_dir: Path | None) -> EgoMuscleDataset:
    return EgoMuscleDataset(
        clip_dir=root / "clips",
        muscle_dir=root / "muscles",
        metadata_path=root / "metadata.json",
        n_frames=n_frames,
        image_size=image_size,
        muscle_dim=None,
        require_muscle=True,
        frame_cache_dir=frame_cache_dir,
        write_frame_cache=frame_cache_dir is not None,
    )


def encode_targets(muscle: torch.Tensor, mode: str) -> np.ndarray:
    if mode == "mean":
        return muscle.mean(dim=1).cpu().numpy()
    if mode == "flatten":
        return muscle.reshape(muscle.shape[0], -1).cpu().numpy()
    raise ValueError(f"Unsupported target mode: {mode}")


def infer_twente_subject(clip_id: str, metadata: dict[str, Any] | None) -> str:
    if metadata and metadata.get("subject"):
        return str(metadata["subject"])
    if "__" in clip_id:
        return clip_id.split("__", maxsplit=1)[0]
    return ""


def infer_twente_activity(clip_id: str, metadata: dict[str, Any] | None, batch_activity: str | None) -> str | None:
    if metadata and metadata.get("activity"):
        return str(metadata["activity"])
    if batch_activity:
        activity = str(batch_activity)
        if not activity.startswith("Subj"):
            return activity
    if "__" in clip_id:
        return clip_id.split("__", maxsplit=1)[1]
    return batch_activity


def collect_twente_features(
    module: EgoMuscleLightningModule,
    dataloader: DataLoader,
    device: torch.device,
    *,
    target_mode: str,
) -> dict[str, Any]:
    if not module.model.use_video:
        raise ValueError("Twente evaluation requires a video-enabled checkpoint.")

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    clip_ids: list[str] = []
    subjects: list[str] = []
    activities: list[str | None] = []

    total_clips = len(dataloader.dataset)
    with torch.no_grad():
        pbar = tqdm(total=total_clips, desc="Processing Twente clips", unit="clip", dynamic_ncols=True)
        for batch in dataloader:
            frames = batch["frames"].to(device)
            if frames.dtype == torch.uint8:
                frames = frames.float().div_(255.0)
                frames = (frames - module._image_mean.to(device)) / module._image_std.to(device)
            activity_ids = batch["activity_id"].to(device)
            outputs = module.model(frames=frames, muscle=None, activity_ids=activity_ids, mask_ratio=0.0)
            features.append(outputs.pooled.detach().cpu().numpy())

            if batch["muscle"] is None:
                raise ValueError("Twente evaluation requires EMG targets.")
            targets.append(encode_targets(batch["muscle"], target_mode))
            batch_clip_ids = list(batch["clip_id"])
            clip_ids.extend(batch_clip_ids)
            for clip_id, batch_activity, metadata in zip(batch_clip_ids, batch["activity"], batch["metadata"], strict=True):
                metadata_dict = metadata or {}
                subjects.append(infer_twente_subject(clip_id, metadata_dict))
                activities.append(infer_twente_activity(clip_id, metadata_dict, batch_activity))
            pbar.update(len(batch["clip_id"]))
        pbar.close()

    return {
        "features": np.concatenate(features, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "clip_ids": clip_ids,
        "subjects": np.asarray(subjects),
        "activities": activities,
    }


def evaluate_loso(features: np.ndarray, targets: np.ndarray, subjects: np.ndarray, activities: list[str | None]) -> dict[str, Any]:
    unique_subjects = sorted({str(subject) for subject in subjects if str(subject)})
    if len(unique_subjects) < 2:
        raise ValueError("Twente evaluation requires at least two subjects for leave-one-subject-out probing.")

    activity_names = sorted(list(set([a for a in activities if a is not None])))
    activity_map = {name: i for i, name in enumerate(activity_names)}
    activity_labels = np.array([activity_map[a] if a in activity_map else -1 for a in activities])
    valid_mask = activity_labels != -1

    folds: list[dict[str, Any]] = []
    r2_scores: list[float] = []
    rho_scores: list[float] = []
    acc_scores: list[float] = []

    for subject in unique_subjects:
        test_mask = subjects == subject
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        from sklearn.linear_model import Ridge, LogisticRegression
        from scipy.stats import spearmanr
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, r2_score
        scaler = StandardScaler()
        train_x = scaler.fit_transform(features[train_mask])
        test_x = scaler.transform(features[test_mask])
        reg = Ridge(alpha=100.0)
        reg.fit(train_x, targets[train_mask])
        preds = reg.predict(test_x)
        r2 = float(r2_score(targets[test_mask], preds))
        rhos = []
        for i in range(targets.shape[1]):
            rho, _ = spearmanr(preds[:, i], targets[test_mask, i])
            if not np.isnan(rho):
                rhos.append(rho)
        mean_rho = float(np.mean(rhos)) if rhos else 0.0
        fold_train_mask = train_mask & valid_mask
        fold_test_mask = test_mask & valid_mask
        acc = 0.0
        if fold_train_mask.sum() > 0 and fold_test_mask.sum() > 0:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(scaler.transform(features[fold_train_mask]), activity_labels[fold_train_mask])
            acc_preds = clf.predict(scaler.transform(features[fold_test_mask]))
            acc = float(accuracy_score(activity_labels[fold_test_mask], acc_preds))
        r2_scores.append(r2)
        rho_scores.append(mean_rho)
        acc_scores.append(acc)
        folds.append(
            {
                "subject": subject,
                "train_clips": int(train_mask.sum()),
                "test_clips": int(test_mask.sum()),
                "mean_r2": r2,
                "mean_rho": mean_rho,
                "activity_acc": acc,
            }
        )

    if not folds:
        raise ValueError("No valid LOSO folds were produced for Twente evaluation.")

    return {
        "mean_r2": float(np.mean(r2_scores)),
        "mean_rho": float(np.mean(rho_scores)),
        "mean_acc": float(np.mean(acc_scores)),
        "std_rho": float(np.std(rho_scores)),
        "folds": folds,
    }


def maybe_write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("folds", [])
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    module = load_lightning_module(args.checkpoint, args.config, args.override, device)
    frame_cache_dir = args.frame_cache_dir
    if frame_cache_dir is None:
        frame_cache_dir = args.twente_root / f"frame_cache_{args.n_frames}x{args.image_size}"
    dataset = build_dataset(
        args.twente_root,
        n_frames=args.n_frames,
        image_size=args.image_size,
        frame_cache_dir=frame_cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_egomuscle,
        pin_memory=(device.type == "cuda"),
    )

    collected = collect_twente_features(module, dataloader, device, target_mode=args.target_mode)
    summary = evaluate_loso(collected["features"], collected["targets"], collected["subjects"], collected["activities"])
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "twente_root": str(args.twente_root),
            "frame_cache_dir": str(frame_cache_dir) if frame_cache_dir else None,
            "num_clips": int(len(collected["clip_ids"])),
            "num_subjects": int(len(sorted(set(collected["subjects"].tolist())))),
            "feature_dim": int(collected["features"].shape[1]),
            "target_dim": int(collected["targets"].shape[1]),
            "target_mode": args.target_mode,
            "representation_mode": "video_only",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    if args.csv_output is not None:
        maybe_write_csv(args.csv_output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

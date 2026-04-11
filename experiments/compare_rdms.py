"""Compare internal RDMs across ablation models.

For each trained model checkpoint, extract pooled representations on the
*same* set of AMASS validation clips, compute an N×N RDM (cosine distance),
and then:

1. Compare RDMs between models using Spearman correlation (inter-model RSA).
   - e.g. Does E3 (peripersonal+muscle) organise clips differently from
     E0 (vision-only) or E4 (exocentric+muscle)?

2. Compute within-category vs. between-category dissimilarity ratios.
   - If muscle grounding makes the model cluster "reaching" clips closer
     together than "walking" clips, that shows up as a lower within/between
     ratio for motor-relevant categories.

3. Save all RDMs, features and activity labels so they can be visualised
   downstream (MDS, t-SNE, dendrogrammes) without re-running inference.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.eval.rdm import compute_rdm
from egomuscle.model.egomuscle import EgoMuscleModel
from egomuscle.training.train import EgoMuscleLightningModule, apply_override, load_config

@dataclass
class ModelSpec:
    key: str
    checkpoint: Path
    overrides: tuple[str, ...]
    dataset_root: Path


def discover_models(ablation_root: Path, ego_root: Path, exo_root: Path) -> list[ModelSpec]:
    """Auto-discover completed ablation models from a sweep directory."""
    ablation_map: dict[str, tuple[str, ...]] = {
        "E0": ("model.use_video=true", "model.use_muscle=false", "model.label_conditioning=false"),
        "E1": ("model.use_video=false", "model.use_muscle=true", "model.label_conditioning=false"),
        "E2": ("model.use_video=true", "model.use_muscle=true", "model.fusion_mode=late"),
        "E3": ("model.use_video=true", "model.use_muscle=true", "model.fusion_mode=cross_attn"),
        "E4": ("model.use_video=true", "model.use_muscle=true", "model.fusion_mode=cross_attn"),
        "E5": ("model.use_video=true", "model.use_muscle=true", "model.fusion_mode=cross_attn"),
        "E6": ("model.use_video=true", "model.use_muscle=false", "model.label_conditioning=true"),
    }
    specs: list[ModelSpec] = []
    for key, overrides in ablation_map.items():
        ckpt_dir = ablation_root / key
        if not ckpt_dir.exists():
            continue
        ckpts = sorted(ckpt_dir.rglob("*.ckpt"))
        if not ckpts:
            continue
        dataset_root = exo_root if key == "E4" else ego_root
        specs.append(ModelSpec(
            key=key,
            checkpoint=ckpts[-1],
            overrides=overrides,
            dataset_root=dataset_root,
        ))
    return specs


def load_model(spec: ModelSpec, config_path: Path, device: torch.device) -> EgoMuscleModel:
    """Instantiate and load a model from a checkpoint with the right overrides."""
    config = load_config(config_path)
    for override in spec.overrides:
        apply_override(config, override)

    module = EgoMuscleLightningModule(config)
    payload = torch.load(spec.checkpoint, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload)
    module.load_state_dict(state_dict, strict=True)
    module.to(device)
    module.eval()
    return module.model


def extract_representations(
    model: EgoMuscleModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run the model on a dataset and collect pooled representations + metadata."""
    pooled_list: list[np.ndarray] = []
    activities: list[str] = []
    clip_ids: list[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  extracting", leave=False):
            frames = batch["frames"].to(device) if model.use_video else None
            muscle = batch["muscle"]
            if muscle is not None and model.muscle_encoder is not None:
                muscle = muscle.to(device)
            else:
                muscle = None

            activity_ids = batch["activity_id"].to(device)
            out = model(frames=frames, muscle=muscle, activity_ids=activity_ids, mask_ratio=0.0)
            pooled_list.append(out.pooled.cpu().numpy())
            activities.extend(batch["activity"])
            clip_ids.extend(batch["clip_id"])

    features = np.concatenate(pooled_list, axis=0)
    return {
        "features": features,
        "activities": activities,
        "clip_ids": clip_ids,
    }


def category_dissimilarity_ratio(rdm: np.ndarray, activities: list[str]) -> dict[str, Any]:
    """Compute within-category / between-category dissimilarity for each activity."""
    n = len(activities)
    act_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, act in enumerate(activities):
        if act:
            act_to_idx[act].append(i)

    results: dict[str, Any] = {}
    for act, indices in sorted(act_to_idx.items()):
        if len(indices) < 2:
            continue
        within = []
        between = []
        idx_set = set(indices)
        for i in indices:
            for j in indices:
                if i < j:
                    within.append(rdm[i, j])
            for j in range(n):
                if j not in idx_set:
                    between.append(rdm[i, j])
        if within and between:
            w = float(np.mean(within))
            b = float(np.mean(between))
            results[act] = {
                "within": w,
                "between": b,
                "ratio": w / b if b > 0 else float("inf"),
                "n_clips": len(indices),
            }

    ratios = [v["ratio"] for v in results.values() if v["ratio"] < float("inf")]
    results["__global__"] = {
        "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "n_activities": len(results) - 1,
    }
    return results


def inter_model_rsa(rdms: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute pairwise Spearman correlations between model RDMs."""
    keys = sorted(rdms.keys())
    results: dict[str, float] = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            v1 = squareform(rdms[k1], checks=False)
            v2 = squareform(rdms[k2], checks=False)
            rho, _ = spearmanr(v1, v2)
            results[f"{k1}_vs_{k2}"] = float(rho)
    return results

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare internal RDMs across ablation models.")
    p.add_argument("--ablation-root", type=Path,
                    default=Path("experiments/results/ablations/20260410_203938"))
    p.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    p.add_argument("--ego-root", type=Path, default=Path("data/processed_amass"))
    p.add_argument("--exo-root", type=Path, default=Path("data/processed_exo"))
    p.add_argument("--output", type=Path, default=Path("experiments/results/rdm_comparison.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--only", nargs="+", help="Only run these ablation keys (e.g. E0 E3 E4)")
    p.add_argument("--save-rdms", action="store_true",
                    help="Save the raw RDM matrices as .npy files alongside the JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    specs = discover_models(args.ablation_root, args.ego_root, args.exo_root)
    if args.only:
        specs = [s for s in specs if s.key in set(args.only)]
    if not specs:
        print("No model checkpoints found. Check --ablation-root.")
        return

    print(f"Found {len(specs)} models: {[s.key for s in specs]}")

    val_clip_dir = args.ego_root / "val" / "clips"
    val_muscle_dir = args.ego_root / "val" / "muscles"
    val_metadata = args.ego_root / "val" / "metadata.json"

    print(f"Loading validation set from {val_clip_dir}...")
    val_dataset = EgoMuscleDataset(
        clip_dir=val_clip_dir,
        muscle_dir=val_muscle_dir,
        metadata_path=val_metadata,
        require_muscle=True,
    )
    print(f"  {len(val_dataset)} clips")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_egomuscle,
    )

    all_rdms: dict[str, np.ndarray] = {}
    all_features: dict[str, np.ndarray] = {}
    all_activities: list[str] = []
    all_clip_ids: list[str] = []
    category_results: dict[str, Any] = {}

    for spec in specs:
        print(f"\n[{spec.key}] Loading {spec.checkpoint.name}...")
        model = load_model(spec, args.config, device)

        print(f"[{spec.key}] Extracting representations...")
        data = extract_representations(model, val_loader, device)
        features = data["features"]
        activities = data["activities"]

        if not all_activities:
            all_activities = activities
            all_clip_ids = data["clip_ids"]

        rdm = compute_rdm(features)
        all_rdms[spec.key] = rdm
        all_features[spec.key] = features

        cat = category_dissimilarity_ratio(rdm, activities)
        category_results[spec.key] = cat
        global_ratio = cat["__global__"]["mean_ratio"]
        print(f"[{spec.key}] Category clustering ratio: {global_ratio:.4f} "
              f"(lower = tighter activity clusters)")

        del model
        torch.cuda.empty_cache()

    print("\n--- Inter-model RSA ---")
    rsa_results = inter_model_rsa(all_rdms)
    for pair, rho in sorted(rsa_results.items()):
        print(f"  {pair}: ρ = {rho:.4f}")

    output = {
        "ablation_root": str(args.ablation_root),
        "val_clips": len(val_dataset),
        "models": {
            spec.key: {
                "checkpoint": str(spec.checkpoint),
                "category_clustering_ratio": category_results[spec.key]["__global__"]["mean_ratio"],
                "n_activities": category_results[spec.key]["__global__"]["n_activities"],
            }
            for spec in specs
        },
        "inter_model_rsa": rsa_results,
        "category_details": {
            key: {
                act: vals for act, vals in cat.items()
                if act != "__global__"
            }
            for key, cat in category_results.items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")

    if args.save_rdms:
        rdm_dir = args.output.parent / "rdms"
        rdm_dir.mkdir(parents=True, exist_ok=True)
        for key, rdm in all_rdms.items():
            np.save(rdm_dir / f"{key}_rdm.npy", rdm)
        np.save(rdm_dir / "activities.npy", np.array(all_activities))
        np.save(rdm_dir / "clip_ids.npy", np.array(all_clip_ids))
        for key, feats in all_features.items():
            np.save(rdm_dir / f"{key}_features.npy", feats)
        print(f"Raw RDMs and features saved to {rdm_dir}")


if __name__ == "__main__":
    main()

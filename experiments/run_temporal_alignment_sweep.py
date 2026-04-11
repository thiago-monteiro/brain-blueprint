from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.training.losses import total_loss
from egomuscle.training.smfe_losses import smfe_total_loss
from egomuscle.eval.twente_eval import load_lightning_module
from egomuscle.training.train import apply_override, load_config
from experiments.run_ablations_csv import ABLATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation loss under explicit muscle/video lag offsets.")
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offsets", nargs="+", type=int, default=[-8, -4, -2, -1, 0, 1, 2, 4, 8])
    parser.add_argument("--only", nargs="+", default=["E2", "E3", "E4", "E5"])
    parser.add_argument("--output", type=Path, default=Path("experiments/results/temporal_alignment_sweep.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def build_dataset(config: dict[str, Any], dataset_root: Path, split: str, offset: int) -> EgoMuscleDataset:
    split_cfg = config["data"][split]
    return EgoMuscleDataset(
        clip_dir=dataset_root / split / "clips",
        muscle_dir=dataset_root / split / "muscles",
        metadata_path=dataset_root / split / "metadata.json",
        n_frames=int(config["data"].get("n_frames", 16)),
        image_size=int(config["data"].get("image_size", 224)),
        muscle_dim=config["data"].get("muscle_dim"),
        require_muscle=True,
        scramble_video=bool(split_cfg.get("scramble_video", False)),
        temporal_sample_mode="sparse_uniform",
        muscle_time_offset=offset,
        muscle_noise_std=0.0,
        frame_cache_dir=None,
        full_cache_dir=split_cfg.get("full_cache_dir"),
        write_frame_cache=False,
        is_train=False,
    )


def evaluate_offset(
    module,
    loader: DataLoader,
    *,
    loss_mode: str,
    loss_name: str,
    loss_weights: dict[str, float] | None,
    device: torch.device,
) -> float:
    losses: list[float] = []
    module.eval()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            muscle = None if batch["muscle"] is None else batch["muscle"].to(device)
            activity_ids = batch["activity_id"].to(device)
            outputs = module.model(frames=frames, muscle=muscle, activity_ids=activity_ids, mask_ratio=0.5)
            if outputs.pred is None or outputs.target is None:
                raise ValueError("Temporal alignment sweep requires muscle prediction outputs.")
            if loss_name == "smfe":
                loss, _ = smfe_total_loss(outputs, weights=loss_weights)
            else:
                loss, _ = total_loss(outputs.pred, outputs.target, outputs.fused, loss_mode=loss_mode, weights=loss_weights)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    base_config = load_config(args.config)
    for override in args.override:
        apply_override(base_config, override)

    selected = [ablation for ablation in ABLATIONS if ablation.key in set(args.only)]
    if not selected:
        raise ValueError("No matching ablations selected.")

    payload: dict[str, Any] = {
        "ablation_root": str(args.ablation_root),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "offsets": args.offsets,
        "results": {},
    }
    csv_rows: list[dict[str, Any]] = []

    for ablation in selected:
        checkpoint_paths = sorted((args.ablation_root / ablation.key).rglob("*.ckpt"))
        if not checkpoint_paths:
            continue
        checkpoint = checkpoint_paths[-1]
        module = load_lightning_module(checkpoint, args.config, list(args.override) + list(ablation.overrides), device)
        model_config = load_config(args.config)
        for override in args.override:
            apply_override(model_config, override)
        for override in ablation.overrides:
            apply_override(model_config, override)
        ablation_results: list[dict[str, Any]] = []
        for offset in args.offsets:
            dataset = build_dataset(model_config, args.dataset_root, args.split, offset)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_egomuscle,
            )
            mean_loss = evaluate_offset(
                module,
                loader,
                loss_mode=model_config["training"].get("loss_mode", "mse"),
                loss_name=model_config["training"].get("loss_name", "current"),
                loss_weights=model_config["training"].get("loss_weights"),
                device=device,
            )
            row = {"ablation": ablation.key, "offset": int(offset), "mean_loss": mean_loss}
            ablation_results.append(row)
            csv_rows.append(row)
        best = min(ablation_results, key=lambda row: row["mean_loss"])
        payload["results"][ablation.key] = {
            "checkpoint": str(checkpoint),
            "best_offset": int(best["offset"]),
            "best_mean_loss": float(best["mean_loss"]),
            "curve": ablation_results,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ablation", "offset", "mean_loss"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()

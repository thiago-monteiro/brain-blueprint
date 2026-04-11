from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.eval.twente_eval import load_lightning_module
from egomuscle.model.adapters import AdapterLinear
from egomuscle.training.smfe_losses import smfe_total_loss
from egomuscle.training.train import apply_override, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe slow-adapter long-term memory with few-shot adaptation and interference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--adapt-split", choices=("train", "val"), default="train")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--few-shot-batches", nargs="+", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--interference-batches", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/ltm_probe_suite.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def build_dataset(config: dict[str, Any], dataset_root: Path, split: str) -> EgoMuscleDataset:
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
        temporal_sample_mode=split_cfg.get("temporal_sample_mode", "sparse_uniform") if split == "train" else "sparse_uniform",
        muscle_time_offset=0,
        muscle_noise_std=0.0,
        frame_cache_dir=None,
        full_cache_dir=split_cfg.get("full_cache_dir"),
        write_frame_cache=False,
        is_train=(split == "train"),
    )


def adapter_parameters(module) -> list[torch.nn.Parameter]:
    params = []
    for submodule in module.model.modules():
        if isinstance(submodule, AdapterLinear):
            for name, param in submodule.named_parameters(recurse=False):
                if name.startswith("adapter_"):
                    params.append(param)
    return params


def freeze_except_adapters(module) -> list[torch.nn.Parameter]:
    for param in module.parameters():
        param.requires_grad = False
    params = adapter_parameters(module)
    for param in params:
        param.requires_grad = True
    return params


def evaluate(module, loader: DataLoader, device: torch.device, *, max_batches: int | None) -> dict[str, float]:
    module.eval()
    mses: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            output = module.model(
                frames=batch["frames"].to(device),
                muscle=batch["muscle"].to(device),
                activity_ids=batch["activity_id"].to(device),
                mask_ratio=0.5,
            )
            if output.pred is None or output.target is None:
                raise ValueError("LTM probe requires muscle predictions.")
            mses.append(float(F.mse_loss(output.pred, output.target).detach().cpu().item()))
            if output.pred_mu is not None:
                loss, _ = smfe_total_loss(output)
                losses.append(float(loss.detach().cpu().item()))
    result = {"mse": float(np.mean(mses))}
    if losses:
        result["smfe_loss"] = float(np.mean(losses))
    return result


def train_adapter_batches(module, loader: DataLoader, device: torch.device, optimizer: AdamW, batches: int) -> float:
    module.train()
    seen = 0
    losses: list[float] = []
    while seen < batches:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = module.model(
                frames=batch["frames"].to(device),
                muscle=batch["muscle"].to(device),
                activity_ids=batch["activity_id"].to(device),
                mask_ratio=0.5,
            )
            if output.pred_mu is not None:
                loss, _ = smfe_total_loss(output)
            else:
                if output.pred is None or output.target is None:
                    raise ValueError("LTM probe requires muscle predictions.")
                loss = F.mse_loss(output.pred, output.target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            seen += 1
            if seen >= batches:
                break
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)
    base_module = load_lightning_module(args.checkpoint, args.config, args.override, device)
    adapter_params = freeze_except_adapters(base_module)
    if not adapter_params:
        raise ValueError("No slow adapters found. Train/load with model.slow_adapter.enabled=true.")

    adapt_dataset = build_dataset(config, args.dataset_root, args.adapt_split)
    eval_dataset = build_dataset(config, args.dataset_root, args.eval_split)
    adapt_loader = DataLoader(adapt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_egomuscle)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_egomuscle)

    rows: list[dict[str, Any]] = []
    baseline = evaluate(base_module, eval_loader, device, max_batches=args.max_eval_batches)
    rows.append({"probe": "baseline", "adapt_batches": 0, **baseline})
    base_state = deepcopy(base_module.state_dict())

    for batches in args.few_shot_batches:
        module = deepcopy(base_module)
        freeze_except_adapters(module)
        optimizer = AdamW(adapter_parameters(module), lr=args.learning_rate)
        train_loss = train_adapter_batches(module, adapt_loader, device, optimizer, batches)
        metrics = evaluate(module, eval_loader, device, max_batches=args.max_eval_batches)
        rows.append({"probe": "few_shot", "adapt_batches": batches, "adapt_train_loss": train_loss, **metrics})

    module = deepcopy(base_module)
    module.load_state_dict(base_state)
    freeze_except_adapters(module)
    optimizer = AdamW(adapter_parameters(module), lr=args.learning_rate)
    loss_a = train_adapter_batches(module, adapt_loader, device, optimizer, args.interference_batches)
    after_a = evaluate(module, eval_loader, device, max_batches=args.max_eval_batches)
    loss_b = train_adapter_batches(module, adapt_loader, device, optimizer, args.interference_batches)
    after_b = evaluate(module, eval_loader, device, max_batches=args.max_eval_batches)
    rows.append({"probe": "interference_after_a", "adapt_batches": args.interference_batches, "adapt_train_loss": loss_a, **after_a})
    rows.append({"probe": "interference_after_b", "adapt_batches": args.interference_batches * 2, "adapt_train_loss": loss_b, **after_b})

    payload = {"checkpoint": str(args.checkpoint), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

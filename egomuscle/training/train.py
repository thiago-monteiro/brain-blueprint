from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from egomuscle.data.dataset import IMAGE_MEAN, IMAGE_STD, EgoMuscleDataset, collate_egomuscle, discover_records
from egomuscle.eval.rdm import compute_rdm
from egomuscle.eval.rsa import rsa_score
from egomuscle.model.egomuscle import EgoMuscleModel
from egomuscle.training.losses import total_loss
from egomuscle.training.smfe_losses import smfe_total_loss


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle)


def apply_override(config: dict[str, Any], override: str) -> None:
    key_path, raw_value = override.split("=", maxsplit=1)
    value = yaml.safe_load(raw_value)
    node = config
    keys = key_path.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def validate_configured_data(config: dict[str, Any]) -> None:
    data_cfg = config.get("data", {})
    for split in ("train", "val"):
        split_cfg = data_cfg.get(split)
        if not split_cfg:
            continue
        clip_dir = split_cfg.get("clip_dir")
        if not clip_dir:
            raise ValueError(f"data.{split}.clip_dir is not configured.")
        records = discover_records(
            video_dir=clip_dir,
            muscle_dir=split_cfg.get("muscle_dir"),
            metadata_path=split_cfg.get("metadata_path"),
        )
        if not records:
            raise ValueError(
                f"No video records found under data.{split}.clip_dir={clip_dir}. "
                "Build processed clips first, for example: FULL_BUILD=1 bash scripts/setup_egomuscle.sh"
            )


class PlainTextProgressCallback(Callback):
    def __init__(self, every_n_batches: int = 25) -> None:
        super().__init__()
        self.every_n_batches = max(int(every_n_batches), 1)
        self.fit_start_time: float | None = None

    def _fmt(self, value: Any) -> str:
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    def _step_prefix(self, trainer: pl.Trainer) -> str:
        max_steps = int(trainer.max_steps) if trainer.max_steps is not None and int(trainer.max_steps) > 0 else 0
        if max_steps > 0:
            step = int(trainer.global_step)
            return f"step={step}/{max_steps}"
        return ""

    def _epoch_suffix(self, trainer: pl.Trainer) -> str:
        epoch = trainer.current_epoch + 1
        return f"epoch={epoch}"

    def _run_name(self, trainer: pl.Trainer) -> str:
        logger = trainer.logger
        name = getattr(logger, "name", None)
        if isinstance(name, str) and name:
            return name
        loggers = getattr(trainer, "loggers", []) or []
        for candidate in loggers:
            name = getattr(candidate, "name", None)
            if isinstance(name, str) and name:
                return name
        return "run"

    def _step_progress_parts(self, trainer: pl.Trainer) -> list[str]:
        max_steps = int(trainer.max_steps) if trainer.max_steps is not None and int(trainer.max_steps) > 0 else 0
        if max_steps <= 0:
            return []
        step = int(trainer.global_step)
        pct = 100.0 * min(max(step, 0), max_steps) / float(max_steps)
        parts = [f"run={self._run_name(trainer)}", f"step={step}/{max_steps}", f"complete={pct:.1f}%"]
        if self.fit_start_time is not None and step > 0:
            elapsed = max(time.perf_counter() - self.fit_start_time, 1.0e-6)
            steps_per_sec = float(step) / elapsed
            remaining = max(max_steps - step, 0)
            eta_s = remaining / max(steps_per_sec, 1.0e-9)
            parts.extend([f"steps_per_sec={steps_per_sec:.3f}", f"eta_min={eta_s / 60.0:.1f}"])
        return parts

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.fit_start_time = time.perf_counter()
        max_steps = int(trainer.max_steps) if trainer.max_steps is not None and int(trainer.max_steps) > 0 else -1
        budget_note = (
            f"step_budget={max_steps} (authoritative)"
            if max_steps > 0
            else f"epoch_budget={trainer.max_epochs} (authoritative)"
        )
        print(
            f"[progress] fit_start run={self._run_name(trainer)} {budget_note} max_epochs_cap={trainer.max_epochs} "
            f"train_batches={trainer.num_training_batches} val_batches={trainer.num_val_batches}",
            flush=True,
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        parts = [*self._step_progress_parts(trainer), self._epoch_suffix(trainer)]
        print(f"[progress] epoch_start {' '.join(parts)}", flush=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        total_batches = trainer.num_training_batches
        is_last = batch_idx + 1 >= total_batches
        if ((batch_idx + 1) % self.every_n_batches) != 0 and not is_last:
            return
        loss = None
        if torch.is_tensor(outputs):
            loss = outputs
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        parts = [*self._step_progress_parts(trainer), self._epoch_suffix(trainer)]
        parts.append(f"batch={batch_idx + 1}/{total_batches}")
        parts.append(f"loss={self._fmt(loss) if loss is not None else 'n/a'}")
        print(f"[progress] train {' '.join(parts)}", flush=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        parts = [*self._step_progress_parts(trainer), self._epoch_suffix(trainer)]
        for key in ("val/loss", "val/pred", "val/temp", "val/var", "val/cov"):
            if key in metrics:
                parts.append(f"{key.replace('/', '_')}={self._fmt(metrics[key])}")
        for key in ("val/muscle_nll", "val/mse", "val/precision", "val/video_latent", "val/fast_kl"):
            if key in metrics:
                parts.append(f"{key.replace('/', '_')}={self._fmt(metrics[key])}")
        print(f"[progress] val_end {' '.join(parts)}", flush=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        parts = [*self._step_progress_parts(trainer), self._epoch_suffix(trainer)]
        for key in ("train/loss_epoch", "train/pred", "train/temp", "train/var", "train/cov"):
            if key in metrics:
                parts.append(f"{key.replace('/', '_')}={self._fmt(metrics[key])}")
        for key in ("train/muscle_nll", "train/mse", "train/precision", "train/video_latent", "train/fast_kl"):
            if key in metrics:
                parts.append(f"{key.replace('/', '_')}={self._fmt(metrics[key])}")
        print(f"[progress] epoch_end {' '.join(parts)}", flush=True)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        parts = []
        for key in ("test/loss", "test/pred", "test/temp", "test/var", "test/cov"):
            if key in metrics:
                parts.append(f"{key.replace('/', '_')}={self._fmt(metrics[key])}")
        for key in ("test/muscle_nll", "test/mse", "test/precision", "test/video_latent", "test/fast_kl"):
            if key in metrics:
                parts.append(f"{key.replace('/', '_')}={self._fmt(metrics[key])}")
        if parts:
            print(f"[progress] test_end {' '.join(parts)}", flush=True)


class EgoMuscleDataModule(pl.LightningDataModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"].get("num_workers", 4)
        self.prefetch_factor = config["data"].get("prefetch_factor", 4)
        self.pin_memory = bool(config["data"].get("pin_memory", True)) and not bool(
            config["training"].get("compile", True)
        )
        self.persistent_workers = config["data"].get("persistent_workers", self.num_workers > 0)
        self.datasets: dict[str, EgoMuscleDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        common = {
            "n_frames": self.config["data"].get("n_frames", 16),
            "image_size": self.config["data"].get("image_size", 224),
            "muscle_dim": self.config["data"].get("muscle_dim"),
        }
        sample_mode = self.config["data"].get("temporal_sample_mode", "sparse_uniform")

        def optional_path(value: Any) -> Any:
            return None if value in (None, "") else value

        for split in ("train", "val", "test"):
            split_cfg = self.config["data"].get(split)
            if not split_cfg:
                continue
            self.datasets[split] = EgoMuscleDataset(
                clip_dir=split_cfg["clip_dir"],
                muscle_dir=split_cfg.get("muscle_dir"),
                metadata_path=split_cfg.get("metadata_path"),
                require_muscle=split != "predict",
                scramble_video=bool(split_cfg.get("scramble_video", False)),
                temporal_sample_mode=split_cfg.get("temporal_sample_mode", sample_mode) if split == "train" else "sparse_uniform",
                muscle_time_offset=int(split_cfg.get("muscle_time_offset", 0)),
                muscle_noise_std=float(split_cfg.get("muscle_noise_std", 0.0)) if split == "train" else 0.0,
                frame_cache_dir=optional_path(split_cfg.get("frame_cache_dir")),
                full_cache_dir=optional_path(split_cfg.get("full_cache_dir")),
                write_frame_cache=bool(split_cfg.get("write_frame_cache", True)),
                replacement_sampling=bool(split_cfg.get("replacement_sampling", False)),
                virtual_size=split_cfg.get("virtual_size"),
                threat_correlation_fraction=float(split_cfg.get("threat_correlation_fraction", 0.0)),
                threat_signature_strength=float(split_cfg.get("threat_signature_strength", 0.0)),
                threat_startle_width=float(split_cfg.get("threat_startle_width", 0.08)),
                threat_withdrawal_bias=float(split_cfg.get("threat_withdrawal_bias", 0.75)),
                threat_autonomic_ramp=float(split_cfg.get("threat_autonomic_ramp", 0.25)),
                threat_seed=int(split_cfg.get("threat_seed", self.config.get("seed", 0))),
                is_train=(split == "train"),
                **common,
            )

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        if split not in self.datasets:
            raise ValueError(f"{split} dataset is not configured.")
        loader_kwargs: dict[str, Any] = {
            "dataset": self.datasets[split],
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
            "collate_fn": collate_egomuscle,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            **loader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)


class EgoMuscleLightningModule(pl.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        model_cfg = config["model"]
        data_cfg = config["data"]
        self.model = EgoMuscleModel(
            video_model_name=model_cfg.get("video_model_name", "MCG-NJU/videomae-base"),
            muscle_dim=data_cfg.get("muscle_dim", 20),
            prediction_dim=model_cfg.get("prediction_dim", data_cfg.get("muscle_dim", 20)),
            muscle_hidden_dim=model_cfg.get("muscle_hidden_dim", 128),
            video_trainable_strategy=model_cfg.get("video_trainable_strategy", "frozen"),
            video_trainable_layers=model_cfg.get("video_trainable_layers", 0),
            video_unfreeze_embeddings=model_cfg.get("video_unfreeze_embeddings", False),
            fusion_mode=model_cfg.get("fusion_mode", "cross_attn"),
            use_video=model_cfg.get("use_video", True),
            use_muscle=model_cfg.get("use_muscle", True),
            label_conditioning=model_cfg.get("label_conditioning", False),
            label_vocab_size=model_cfg.get("label_vocab_size", 512),
            scramble_video=model_cfg.get("scramble_video", False),
            fusion_dropout=model_cfg.get("fusion_dropout", 0.1),
            pred_dropout=model_cfg.get("pred_dropout", 0.1),
            predictive_distribution=model_cfg.get("predictive_distribution", "point"),
            video_latent_prediction=model_cfg.get("video_latent_prediction", False),
            fast_memory=model_cfg.get("fast_memory"),
            pbit=model_cfg.get("pbit"),
            slow_adapter=model_cfg.get("slow_adapter"),
        )
        self.training_cfg = config["training"]
        # (1, 3, 1, 1) broadcasts to (B, T, C, H, W). Non-persistent for checkpoint compat.
        self.register_buffer("_image_mean", IMAGE_MEAN.clone(), persistent=False)
        self.register_buffer("_image_std", IMAGE_STD.clone(), persistent=False)
        if self.training_cfg.get("compile", True):
            self.model = self._compile_training_model(self.model)
        self.eval_cfg = config.get("evaluation", {})
        self.scaling_cfg = config.get("scaling", {})
        self.validation_representations: list[torch.Tensor] = []

    def _compile_training_model(self, model: EgoMuscleModel) -> EgoMuscleModel:
        compile_mode = str(self.training_cfg.get("compile_mode", "default"))
        if compile_mode == "reduce-overhead":
            # reduce-overhead captures CUDA graphs; Lightning's train/backward loop reuses
            # tensors across steps and triggers "CUDAGraphs overwritten" errors.
            print(
                "compile_mode=reduce-overhead is not compatible with Lightning training; "
                "using compile_mode=default instead.",
                flush=True,
            )
            compile_mode = "default"
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
        print(f"EgoMuscleModel compiled for speed (mode={compile_mode}).", flush=True)
        return compiled

    def _mark_compile_step(self) -> None:
        if not self.training_cfg.get("compile", True):
            return
        mark = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        if callable(mark):
            mark()

    def on_fit_start(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[progress] model_params total={total} trainable={trainable}", flush=True)
        if self.trainer is None:
            return
        metrics = {
            "model/total_params": float(total),
            "model/trainable_params": float(trainable),
        }
        for key, value in self.scaling_cfg.items():
            if isinstance(value, bool):
                metrics[f"scaling/{key}"] = float(value)
            elif isinstance(value, (int, float)):
                metrics[f"scaling/{key}"] = float(value)
        for logger in self.trainer.loggers:
            logger.log_metrics(metrics, step=0)

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        batch_size = 0
        frames = batch.get("frames")
        muscle = batch.get("muscle")
        if torch.is_tensor(frames):
            batch_size = int(frames.shape[0])
        elif torch.is_tensor(muscle):
            batch_size = int(muscle.shape[0])

        outputs = self.model(
            frames=frames,
            muscle=muscle,
            activity_ids=batch["activity_id"],
            mask_ratio=self.training_cfg.get("mask_ratio", 0.5),
        )
        if outputs.pred is None or outputs.target is None:
            raise ValueError("Training and evaluation require muscle supervision.")

        if self.training_cfg.get("loss_name", "current") == "smfe":
            loss, loss_parts = smfe_total_loss(
                outputs,
                weights=self.training_cfg.get("loss_weights"),
            )
        else:
            loss, loss_parts = total_loss(
                outputs.pred,
                outputs.target,
                outputs.fused,
                loss_mode=self.training_cfg.get("loss_mode", "mse"),
                weights=self.training_cfg.get("loss_weights"),
            )

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=(stage != "train"),
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=batch_size,
        )
        for name, value in loss_parts.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True, batch_size=batch_size)

        if stage == "val":
            self.validation_representations.append(outputs.pooled.detach().cpu())

        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self._mark_compile_step()
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self._mark_compile_step()
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def transfer_batch_to_device(self, batch: dict[str, Any], device: torch.device, dataloader_idx: int) -> dict[str, Any]:
        non_blocking = device.type == "cuda" and not self.training_cfg.get("compile", True)
        frames = batch.get("frames")
        if torch.is_tensor(frames):
            frames = frames.to(device, non_blocking=non_blocking)
            if frames.dtype == torch.uint8:
                frames = frames.float().div_(255.0)
                frames = (frames - self._image_mean) / self._image_std
            batch["frames"] = frames

        for key in ("muscle", "activity_id"):
            value = batch.get(key)
            if torch.is_tensor(value):
                batch[key] = value.to(device, non_blocking=non_blocking)
        return batch

    def on_validation_epoch_start(self) -> None:
        self.validation_representations = []

    def on_validation_epoch_end(self) -> None:
        neural_dir = self.eval_cfg.get("neural_rdms_dir")
        if not neural_dir or not self.validation_representations:
            return

        pooled = torch.cat(self.validation_representations, dim=0).numpy()
        subset_size = min(len(pooled), int(self.eval_cfg.get("rsa_monitor_subset", len(pooled))))
        if subset_size < 2:
            return
        pooled = pooled[:subset_size]
        model_rdm = compute_rdm(pooled)

        for path in sorted(Path(neural_dir).glob("*.npy")):
            neural_rdm = np.load(path)
            if neural_rdm.shape != model_rdm.shape:
                continue
            score, _ = rsa_score(model_rdm, neural_rdm)
            region_name = path.stem
            self.log(f"val/rsa_{region_name}", score, prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> dict[str, Any]:
        weight_decay = float(self.training_cfg.get("weight_decay", 0.05))
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or ".norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer_grouped_parameters = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(self.training_cfg.get("learning_rate", 3e-4)),
        )

        warmup_epochs = self.training_cfg.get("warmup_epochs", 5)
        max_epochs = self.training_cfg.get("max_epochs", 100)
        max_steps_cfg = self.training_cfg.get("max_steps")
        warmup_steps_cfg = self.training_cfg.get("warmup_steps")
        min_lr_scale = self.training_cfg.get("min_lr_scale", 1e-5 / 3e-4)
        if max_steps_cfg is not None and int(max_steps_cfg) > 0:
            total_steps = int(max_steps_cfg)
            if warmup_steps_cfg is not None:
                warmup_steps = int(warmup_steps_cfg)
            else:
                warmup_ratio = self.training_cfg.get("warmup_ratio", 0.1)
                warmup_steps = int(round(warmup_ratio * total_steps))
            warmup_cap = self.training_cfg.get("warmup_steps_cap")
            if warmup_cap is not None:
                warmup_steps = min(warmup_steps, int(warmup_cap))
            warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(warmup_steps, 1))
                progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(min_lr_scale, cosine)

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(warmup_epochs, 1))
            progress = (epoch - warmup_epochs) / float(max(max_epochs - warmup_epochs, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_scale, cosine)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def build_loggers(config: dict[str, Any]) -> list[Any]:
    logging_cfg = config.get("logging", {})
    save_dir = logging_cfg.get("save_dir", "lightning_logs")
    run_name = logging_cfg.get("run_name", "run")
    loggers: list[Any] = [CSVLogger(save_dir=save_dir, name=run_name)]

    if not logging_cfg.get("use_wandb", False):
        return loggers

    from pytorch_lightning.loggers import WandbLogger

    tags = logging_cfg.get("tags")
    if isinstance(tags, str):
        tags = [tags]

    wandb_kwargs: dict[str, Any] = {
        "project": logging_cfg.get("project", "egomuscle"),
        "name": run_name,
        "save_dir": save_dir,
        "log_model": bool(logging_cfg.get("wandb_log_model", False)),
    }
    optional_fields = {
        "entity": logging_cfg.get("entity"),
        "group": logging_cfg.get("group"),
        "job_type": logging_cfg.get("job_type"),
        "notes": logging_cfg.get("notes"),
        "mode": logging_cfg.get("wandb_mode"),
        "id": logging_cfg.get("wandb_run_id"),
        "resume": logging_cfg.get("wandb_resume"),
        "tags": tags,
    }
    for key, value in optional_fields.items():
        if value not in (None, "", []):
            wandb_kwargs[key] = value

    logger = WandbLogger(**wandb_kwargs)
    logger.experiment.config.update(config, allow_val_change=True)
    loggers.append(logger)
    return loggers


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EgoMuscle models.")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--override", action="append", default=[], help="Override config values: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if config["data"].get("num_workers") is None:
        config["data"]["num_workers"] = min(8, os.cpu_count() or 1)

    pl.seed_everything(config.get("seed", 0), workers=True)
    validate_configured_data(config)

    max_steps_cfg = config["training"].get("max_steps")
    max_epochs_cfg = int(config["training"].get("max_epochs", 100))
    if max_steps_cfg is not None and int(max_steps_cfg) > 0:
        accum = int(config["training"].get("accumulate_grad_batches", 1))
        batch_size = int(config["data"].get("batch_size", 1))
        devices_raw = config["trainer"].get("devices", 1)
        device_count = 1 if devices_raw in (None, "auto") else (
            len(devices_raw) if isinstance(devices_raw, (list, tuple)) else int(devices_raw)
        )
        try:
            from egomuscle.data.dataset import discover_records

            train_cfg = config["data"]["train"]
            n_train = len(
                discover_records(
                    video_dir=train_cfg["clip_dir"],
                    muscle_dir=train_cfg.get("muscle_dir"),
                    metadata_path=train_cfg.get("metadata_path"),
                )
            )
            micro_per_epoch = max(1, math.ceil(n_train / max(1, batch_size * device_count)))
            steps_per_epoch = max(1, math.ceil(micro_per_epoch / max(1, accum)))
            epoch_step_budget = max_epochs_cfg * steps_per_epoch
            if epoch_step_budget > int(max_steps_cfg) * 10:
                print(
                    f"[progress] WARN: max_epochs={max_epochs_cfg} implies ~{epoch_step_budget} "
                    f"optimizer steps but max_steps={int(max_steps_cfg)}; epoch cap may be misleading in logs.",
                    flush=True,
                )
        except (ValueError, KeyError, TypeError):
            pass

    data_module = EgoMuscleDataModule(config)
    lightning_module = EgoMuscleLightningModule(config)

    output_root = Path(config.get("output_dir", "checkpoints"))
    run_name = str(config.get("logging", {}).get("run_name", "run"))
    ckpt_dir = output_root / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch={epoch:02d}-step={step}",
        dirpath=str(ckpt_dir),
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        PlainTextProgressCallback(every_n_batches=config["trainer"].get("progress_every_n_batches", 25)),
    ]
    early_stopping_patience = config["training"].get("early_stopping_patience")
    if early_stopping_patience is not None:
        callbacks.append(EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=int(early_stopping_patience),
            verbose=True,
        ))
    trainer = pl.Trainer(
        accelerator=config["trainer"].get("accelerator", "auto"),
        devices=config["trainer"].get("devices", "auto"),
        max_epochs=config["training"].get("max_epochs", 100),
        max_steps=config["training"].get("max_steps", -1) if config["training"].get("max_steps") is not None else -1,
        precision=config["training"].get("precision", "16-mixed"),
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 4),
        gradient_clip_val=config["training"].get("grad_clip_norm", 1.0),
        num_sanity_val_steps=config["trainer"].get("num_sanity_val_steps", 0),
        logger=build_loggers(config),
        callbacks=callbacks,
        default_root_dir=config.get("output_dir", "checkpoints"),
        log_every_n_steps=config["trainer"].get("log_every_n_steps", 10),
        check_val_every_n_epoch=config["trainer"].get("check_val_every_n_epoch", 1),
        val_check_interval=config["trainer"].get("val_check_interval", 1.0),
        limit_val_batches=config["trainer"].get("limit_val_batches", 1.0),
    )
    trainer.fit(lightning_module, datamodule=data_module)
    if config["data"].get("test"):
        ckpt_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or None
        if ckpt_path:
            print(f"[progress] test_start ckpt_path={ckpt_path}", flush=True)
        else:
            print("[progress] test_start ckpt_path=current_weights", flush=True)
        trainer.test(lightning_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

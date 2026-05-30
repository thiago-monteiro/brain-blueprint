from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmfeCondition:
    key: str
    run_name: str
    train_overrides: tuple[str, ...]
    eval_overrides: tuple[str, ...]
    smfe: bool
    has_pbit: bool
    has_slow_adapter: bool


BASE_CURRENT_OVERRIDES = (
    "training.loss_name=current",
    "training.loss_mode=mse",
    "model.use_video=true",
    "model.use_muscle=true",
    "model.fusion_mode=cross_attn",
    "model.predictive_distribution=point",
    "model.video_latent_prediction=false",
    "model.fast_memory.enabled=false",
    "model.pbit.enabled=false",
    "model.slow_adapter.enabled=false",
)

SMFE_BASE_OVERRIDES = (
    "training.loss_name=smfe",
    "model.predictive_distribution=gaussian",
    "model.video_latent_prediction=true",
)

SMFE_CONDITIONS: tuple[SmfeCondition, ...] = (
    SmfeCondition(
        key="B0",
        run_name="B0_E3_current",
        train_overrides=BASE_CURRENT_OVERRIDES,
        eval_overrides=BASE_CURRENT_OVERRIDES,
        smfe=False,
        has_pbit=False,
        has_slow_adapter=False,
    ),
    SmfeCondition(
        key="B1",
        run_name="B1_E4_exocentric_current",
        train_overrides=(
            *BASE_CURRENT_OVERRIDES,
            "data.train.clip_dir=data/processed_exo/train/clips",
            "data.val.clip_dir=data/processed_exo/val/clips",
            "data.test.clip_dir=data/processed_exo/test/clips",
            "data.train.metadata_path=data/processed_exo/train/metadata.json",
            "data.val.metadata_path=data/processed_exo/val/metadata.json",
            "data.test.metadata_path=data/processed_exo/test/metadata.json",
            "data.train.frame_cache_dir=data/processed_exo/train/frames",
            "data.val.frame_cache_dir=data/processed_exo/val/frames",
            "data.test.frame_cache_dir=data/processed_exo/test/frames",
        ),
        eval_overrides=BASE_CURRENT_OVERRIDES,
        smfe=False,
        has_pbit=False,
        has_slow_adapter=False,
    ),
    SmfeCondition(
        key="B2",
        run_name="B2_E5_scrambled_current",
        train_overrides=(
            *BASE_CURRENT_OVERRIDES,
            "data.train.scramble_video=true",
            "data.val.scramble_video=true",
            "data.test.scramble_video=true",
        ),
        eval_overrides=BASE_CURRENT_OVERRIDES,
        smfe=False,
        has_pbit=False,
        has_slow_adapter=False,
    ),
    SmfeCondition(
        key="S0",
        run_name="S0_smfe_no_memory",
        train_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=false",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
            "training.loss_weights.muscle_nll=1.0",
            "training.loss_weights.video_latent=0.1",
            "training.loss_weights.precision=0.05",
        ),
        eval_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=false",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
        ),
        smfe=True,
        has_pbit=False,
        has_slow_adapter=False,
    ),
    SmfeCondition(
        key="S1",
        run_name="S1_smfe_fast_memory",
        train_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
            "training.loss_weights.muscle_nll=1.0",
            "training.loss_weights.video_latent=0.1",
            "training.loss_weights.fast_kl=0.01",
            "training.loss_weights.precision=0.05",
        ),
        eval_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
        ),
        smfe=True,
        has_pbit=False,
        has_slow_adapter=False,
    ),
    SmfeCondition(
        key="S2",
        run_name="S2_smfe_fast_memory_pbit",
        train_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=stochastic_straight_through",
            "model.slow_adapter.enabled=false",
            "training.loss_weights.entropy=0.01",
            "training.loss_weights.homeostasis=0.01",
        ),
        eval_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=stochastic_straight_through",
            "model.slow_adapter.enabled=false",
        ),
        smfe=True,
        has_pbit=True,
        has_slow_adapter=False,
    ),
    SmfeCondition(
        key="S3",
        run_name="S3_smfe_fast_memory_slow_adapter",
        train_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=null",
            "model.slow_adapter.quantization_mode=none",
            "training.loss_weights.capacity=0.0",
            "training.loss_weights.plastic=0.001",
        ),
        eval_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=null",
            "model.slow_adapter.quantization_mode=none",
        ),
        smfe=True,
        has_pbit=False,
        has_slow_adapter=True,
    ),
    SmfeCondition(
        key="S4",
        run_name="S4_full_smfe_memory",
        train_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=stochastic_straight_through",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=26",
            "model.slow_adapter.quantization_mode=qat",
            "training.precision=32-true",
            "training.loss_weights.entropy=0.01",
            "training.loss_weights.homeostasis=0.01",
            "training.loss_weights.capacity=0.001",
            "training.loss_weights.plastic=0.001",
        ),
        eval_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=stochastic_straight_through",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=26",
            "model.slow_adapter.quantization_mode=qat",
            "training.precision=32-true",
        ),
        smfe=True,
        has_pbit=True,
        has_slow_adapter=True,
    ),
    SmfeCondition(
        key="S5",
        run_name="S5_full_deterministic_control",
        train_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=deterministic_sigmoid",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=26",
            "model.slow_adapter.quantization_mode=qat",
            "training.precision=32-true",
        ),
        eval_overrides=(
            *SMFE_BASE_OVERRIDES,
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=deterministic_sigmoid",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=26",
            "model.slow_adapter.quantization_mode=qat",
            "training.precision=32-true",
        ),
        smfe=True,
        has_pbit=True,
        has_slow_adapter=True,
    ),
)


def parse_keyed_path(value: str) -> tuple[str, Path]:
    key, sep, path = value.partition("=")
    if not sep or not key or not path:
        raise argparse.ArgumentTypeError("Expected KEY=/path/to/checkpoint.ckpt")
    return key, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate SMFE conditions from one entry point.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--only", nargs="+", default=["S0", "S1"])
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("experiments/results/smfe"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint", action="append", type=parse_keyed_path, default=[])
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--twente-root", type=Path, default=Path("data/processed_real/twente"))
    parser.add_argument("--neural-dir", type=Path, default=Path("egomuscle/eval/neural_rdms"))
    parser.add_argument("--stimuli-dir", type=Path)
    parser.add_argument("--stimuli-list", type=Path, default=Path("experiments/results/bold5000_stimuli_list_neural_order.txt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-max-batches", type=int)
    parser.add_argument("--skip-twente", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-pbit", action="store_true")
    parser.add_argument("--skip-agency", action="store_true")
    parser.add_argument("--skip-ltm", action="store_true")
    parser.add_argument("--skip-layerwise", action="store_true")
    return parser.parse_args()


def selected_conditions(keys: list[str]) -> list[SmfeCondition]:
    selected = [condition for condition in SMFE_CONDITIONS if condition.key in set(keys)]
    if not selected:
        valid = ", ".join(condition.key for condition in SMFE_CONDITIONS)
        raise ValueError(f"No matching SMFE conditions selected. Valid keys: {valid}")
    return selected


def run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def add_overrides(command: list[str], overrides: tuple[str, ...] | list[str]) -> list[str]:
    for override in overrides:
        command.extend(["--override", override])
    return command


def run_train(args: argparse.Namespace, conditions: list[SmfeCondition]) -> None:
    for condition in conditions:
        command = [
            sys.executable,
            "-m",
            "egomuscle.training.train",
            "--config",
            str(args.config),
            "--override",
            f"logging.run_name={condition.run_name}",
        ]
        add_overrides(command, (*condition.train_overrides, *args.override))
        run(command, dry_run=args.dry_run)


def checkpoint_overrides(args: argparse.Namespace) -> dict[str, Path]:
    return dict(args.checkpoint)


def resolve_checkpoint(condition: SmfeCondition, args: argparse.Namespace, explicit: dict[str, Path]) -> Path | None:
    if condition.key in explicit:
        return explicit[condition.key]
    candidates = sorted((args.checkpoint_root / condition.run_name).rglob("*.ckpt"))
    if not candidates:
        candidates = sorted((args.checkpoint_root / condition.key).rglob("*.ckpt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_eval(args: argparse.Namespace, conditions: list[SmfeCondition]) -> None:
    args.root.mkdir(parents=True, exist_ok=True)
    explicit = checkpoint_overrides(args)
    for condition in conditions:
        checkpoint = resolve_checkpoint(condition, args, explicit)
        if checkpoint is None:
            print(f"[skip] {condition.key}: no checkpoint under {args.checkpoint_root / condition.run_name}", flush=True)
            continue
        condition_root = args.root / condition.key
        condition_root.mkdir(parents=True, exist_ok=True)
        overrides = (*condition.eval_overrides, *args.override)
        if not args.skip_twente:
            command = [
                sys.executable,
                "-m",
                "egomuscle.eval.twente_eval",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.config),
                "--twente-root",
                str(args.twente_root),
                "--output",
                str(condition_root / "twente_eval.json"),
                "--csv-output",
                str(condition_root / "twente_eval_folds.csv"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
            run(add_overrides(command, overrides), dry_run=args.dry_run)
        if not args.skip_memory and condition.smfe:
            command = [
                sys.executable,
                "experiments/run_memory_probe_suite.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.config),
                "--dataset-root",
                str(args.dataset_root),
                "--output",
                str(condition_root / "memory_probe_suite.json"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.probe_max_batches is not None:
                command.extend(["--max-probe-batches", str(args.probe_max_batches)])
            run(add_overrides(command, overrides), dry_run=args.dry_run)
        if not args.skip_pbit and condition.has_pbit:
            command = [
                sys.executable,
                "experiments/run_pbit_quantization_sweep.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.config),
                "--dataset-root",
                str(args.dataset_root),
                "--output",
                str(condition_root / "pbit_quantization_sweep.json"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--stochastic-samples",
                "8",
            ]
            if args.probe_max_batches is not None:
                command.extend(["--max-batches", str(args.probe_max_batches)])
            run(add_overrides(command, overrides), dry_run=args.dry_run)
        if not args.skip_agency and condition.smfe:
            command = [
                sys.executable,
                "experiments/run_agency_boundary_probe.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.config),
                "--dataset-root",
                str(args.dataset_root),
                "--output",
                str(condition_root / "agency_boundary_probe.json"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.probe_max_batches is not None:
                command.extend(["--max-batches", str(args.probe_max_batches)])
            run(add_overrides(command, overrides), dry_run=args.dry_run)
        if not args.skip_ltm and condition.has_slow_adapter:
            command = [
                sys.executable,
                "experiments/run_ltm_probe_suite.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.config),
                "--dataset-root",
                str(args.dataset_root),
                "--output",
                str(condition_root / "ltm_probe_suite.json"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.probe_max_batches is not None:
                command.extend(["--max-eval-batches", str(args.probe_max_batches)])
            run(add_overrides(command, overrides), dry_run=args.dry_run)
        if not args.skip_layerwise:
            command = [
                sys.executable,
                "experiments/run_layerwise_hierarchy.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.config),
                "--neural-dir",
                str(args.neural_dir),
                "--output",
                str(condition_root / "layerwise_hierarchy.json"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.stimuli_dir is not None:
                command.extend(["--stimuli-dir", str(args.stimuli_dir)])
                if args.stimuli_list.exists():
                    command.extend(["--stimuli-list", str(args.stimuli_list)])
            run(add_overrides(command, overrides), dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    conditions = selected_conditions(args.only)
    if args.mode == "train":
        run_train(args, conditions)
    else:
        run_eval(args, conditions)


if __name__ == "__main__":
    main()

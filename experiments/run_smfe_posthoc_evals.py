from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Condition:
    key: str
    checkpoint: Path
    overrides: tuple[str, ...]
    smfe: bool
    has_pbit: bool
    has_slow_adapter: bool


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        key="B0",
        checkpoint=Path("checkpoints/B0_E3_current/epoch=epoch=09-step=step=170.ckpt"),
        overrides=(
            "training.loss_name=current",
            "training.loss_mode=mse",
            "model.predictive_distribution=point",
            "model.video_latent_prediction=false",
            "model.fast_memory.enabled=false",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
        ),
        smfe=False,
        has_pbit=False,
        has_slow_adapter=False,
    ),
    Condition(
        key="S0",
        checkpoint=Path("checkpoints/S0_smfe_no_memory/epoch=epoch=09-step=step=170.ckpt"),
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=false",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
        ),
        smfe=True,
        has_pbit=False,
        has_slow_adapter=False,
    ),
    Condition(
        key="S1",
        checkpoint=Path("checkpoints/S1_smfe_fast_memory/epoch=epoch=09-step=step=170.ckpt"),
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
        ),
        smfe=True,
        has_pbit=False,
        has_slow_adapter=False,
    ),
    Condition(
        key="S2",
        checkpoint=Path("checkpoints/S2_smfe_fast_memory_pbit/epoch=epoch=09-step=step=170.ckpt"),
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=stochastic_straight_through",
            "model.slow_adapter.enabled=false",
        ),
        smfe=True,
        has_pbit=True,
        has_slow_adapter=False,
    ),
    Condition(
        key="S4",
        checkpoint=Path("checkpoints/S4_full_smfe_memory/epoch=epoch=09-step=step=170.ckpt"),
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
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
    Condition(
        key="S5",
        checkpoint=Path("checkpoints/S5_full_deterministic_control/epoch=epoch=09-step=step=170.ckpt"),
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run posthoc SMFE-Memory pilot evaluations and summary tables.")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--root", type=Path, default=Path("experiments/results/smfe_memory_pilot"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--twente-root", type=Path, default=Path("data/processed_real/twente"))
    parser.add_argument("--neural-dir", type=Path, default=Path("egomuscle/eval/neural_rdms"))
    parser.add_argument("--stimuli-dir", type=Path)
    parser.add_argument("--stimuli-list", type=Path, default=Path("experiments/results/bold5000_stimuli_list_neural_order.txt"))
    parser.add_argument("--only", nargs="+", default=[c.key for c in CONDITIONS])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-max-batches", type=int)
    parser.add_argument("--skip-twente", action="store_true")
    parser.add_argument("--skip-layerwise", action="store_true")
    parser.add_argument("--run-temporal", action="store_true", help="Also run the legacy E-ablation temporal sweep.")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-pbit", action="store_true")
    parser.add_argument("--skip-agency", action="store_true")
    parser.add_argument("--skip-ltm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add_overrides(command: list[str], overrides: tuple[str, ...]) -> list[str]:
    for override in overrides:
        command.extend(["--override", override])
    return command


def run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def checkpoint_exists(condition: Condition) -> bool:
    return condition.checkpoint.exists()


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    selected = [condition for condition in CONDITIONS if condition.key in set(args.only)]
    if not selected:
        raise ValueError("No matching conditions selected.")

    for condition in selected:
        if not checkpoint_exists(condition):
            print(f"[skip] {condition.key}: missing checkpoint {condition.checkpoint}", flush=True)
            continue
        condition_root = args.root / condition.key
        condition_root.mkdir(parents=True, exist_ok=True)

        if not args.skip_twente:
            command = [
                sys.executable,
                "-m",
                "egomuscle.eval.twente_eval",
                "--checkpoint",
                str(condition.checkpoint),
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
            run(add_overrides(command, condition.overrides), dry_run=args.dry_run)

        if not args.skip_memory and condition.smfe:
            command = [
                sys.executable,
                "experiments/run_memory_probe_suite.py",
                "--checkpoint",
                str(condition.checkpoint),
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
            run(add_overrides(command, condition.overrides), dry_run=args.dry_run)

        if not args.skip_pbit and condition.has_pbit:
            command = [
                sys.executable,
                "experiments/run_pbit_quantization_sweep.py",
                "--checkpoint",
                str(condition.checkpoint),
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
            run(add_overrides(command, condition.overrides), dry_run=args.dry_run)

        if not args.skip_agency and condition.smfe:
            command = [
                sys.executable,
                "experiments/run_agency_boundary_probe.py",
                "--checkpoint",
                str(condition.checkpoint),
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
            run(add_overrides(command, condition.overrides), dry_run=args.dry_run)

        if not args.skip_ltm and condition.has_slow_adapter:
            command = [
                sys.executable,
                "experiments/run_ltm_probe_suite.py",
                "--checkpoint",
                str(condition.checkpoint),
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
            run(add_overrides(command, condition.overrides), dry_run=args.dry_run)

        if not args.skip_layerwise:
            command = [
                sys.executable,
                "experiments/run_layerwise_hierarchy.py",
                "--checkpoint",
                str(condition.checkpoint),
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
            run(add_overrides(command, condition.overrides), dry_run=args.dry_run)

    if args.run_temporal:
        command = [
            sys.executable,
            "experiments/run_temporal_alignment_sweep.py",
            "--ablation-root",
            "checkpoints",
            "--config",
            str(args.config),
            "--dataset-root",
            str(args.dataset_root),
            "--only",
            "E3",
            "--output",
            str(args.root / "temporal_alignment_sweep.json"),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        run(command, dry_run=args.dry_run)

    command = [
        sys.executable,
        "experiments/summarize_blueprint_v2.py",
        "--root",
        str(args.root),
        "--output",
        str(args.root / "blueprint_v2_summary.json"),
        "--csv-output",
        str(args.root / "blueprint_v2_summary.csv"),
    ]
    run(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

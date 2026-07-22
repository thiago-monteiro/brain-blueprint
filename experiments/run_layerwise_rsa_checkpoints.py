from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layerwise RSA for multiple checkpoint groups.")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/layerwise_all"))
    parser.add_argument("--stimuli-dir", type=Path, default=None)
    parser.add_argument("--stimuli-list", type=Path, default=None)
    parser.add_argument("--neural-dir", type=Path, default=Path("egomuscle/eval/algonauts2025_rdms"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=30)
    parser.add_argument("--perm-thread-workers", type=int, default=8)
    parser.add_argument("--rdm-workers", type=int, default=0)
    parser.add_argument("--permutation-mode", choices=("pair_shuffle", "mantel"), default="pair_shuffle")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument(
        "--stats-workers",
        type=int,
        default=0,
        help="Parallel workers for stats (0 = conservative memory-aware default). Uses layers×regions tasks.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip checkpoints whose output JSON already exists (default: true).",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even when output JSON exists.")
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run each checkpoint in a new Python process (slower startup; isolates GPU crashes).",
    )
    parser.add_argument(
        "--gpu-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="First extract cached layerwise features for every checkpoint on GPU, then run CPU RDM/stats (default: true).",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Only run the GPU feature extraction phase for all checkpoints, then exit.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU inference and run CPU RDM/stats from cached features.",
    )
    parser.add_argument(
        "--cpu-rdm-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In --cpu-only mode, build all checkpoint RDMs first, then run stats from cached RDMs (default: true).",
    )
    parser.add_argument(
        "--checkpoint-workers",
        type=int,
        default=1,
        help="Checkpoint-level workers for the CPU RDM prepass. Use with small --rdm-workers to avoid oversubscription.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Group spec LABEL=glob. Repeat for E2/E3/E4/E5. If omitted, common clean-run ablation globs are used.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-every-n-batches", type=int, default=25)
    parser.add_argument("--log-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stats-log-per-task", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def default_groups() -> list[str]:
    return [
        "E2=experiments/results/clean_runs/*/ablations_epochs100/seed_*/E2/checkpoints/**/*.ckpt",
        "E3=experiments/results/clean_runs/*/ablations_epochs100/seed_*/E3/checkpoints/**/*.ckpt",
        "E4=experiments/results/clean_runs/*/ablations_epochs100/seed_*/E4/checkpoints/**/*.ckpt",
        "E5=experiments/results/clean_runs/*/ablations_epochs100/seed_*/E5/checkpoints/**/*.ckpt",
    ]


def parse_group(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=glob group spec, got {spec}")
    label, pattern = spec.split("=", 1)
    return label.strip(), pattern.strip()


def output_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("layers") and payload.get("best_by_region"))


def run_subprocess_row(cmd: list[str], label: str, checkpoint: Path, output: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "group": label,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "command": cmd,
    }
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        row["status"] = "ok"
    except subprocess.CalledProcessError as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
    return row


def hierarchy_args_for_checkpoint(runner: argparse.Namespace, checkpoint: Path, output: Path) -> Namespace:
    import torch

    device = runner.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return Namespace(
        checkpoint=checkpoint,
        config=runner.config,
        split="val",
        stimuli_dir=runner.stimuli_dir,
        stimuli_list=runner.stimuli_list,
        neural_dir=runner.neural_dir,
        output=output,
        n_permutations=runner.n_permutations,
        permutation_mode=runner.permutation_mode,
        n_bootstrap=runner.n_bootstrap,
        stats_seed=0,
        stats_workers=runner.stats_workers,
        rdm_workers=runner.rdm_workers,
        max_stat_regions=None,
        device=device,
        batch_size=runner.batch_size,
        num_workers=runner.num_workers,
        override=list(runner.override),
        log_level=runner.log_level,
        log_every_n_batches=runner.log_every_n_batches,
        log_memory=runner.log_memory,
        stats_log_per_task=runner.stats_log_per_task,
        perm_thread_workers=runner.perm_thread_workers,
        features_only=False,
        use_cached_features=False,
        rdms_only=False,
        use_cached_rdms=False,
    )


def build_subprocess_cmd(
    runner: argparse.Namespace,
    checkpoint: Path,
    output: Path,
    *,
    features_only: bool = False,
    use_cached_features: bool = False,
    rdms_only: bool = False,
    use_cached_rdms: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/run_layerwise_hierarchy.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(runner.config),
        "--stimuli-dir",
        str(runner.stimuli_dir),
        "--stimuli-list",
        str(runner.stimuli_list),
        "--neural-dir",
        str(runner.neural_dir),
        "--output",
        str(output),
        "--batch-size",
        str(runner.batch_size),
        "--num-workers",
        str(runner.num_workers),
        "--n-permutations",
        str(runner.n_permutations),
        "--permutation-mode",
        runner.permutation_mode,
        "--n-bootstrap",
        str(runner.n_bootstrap),
        "--stats-workers",
        str(runner.stats_workers),
        "--rdm-workers",
        str(runner.rdm_workers),
        "--log-level",
        runner.log_level,
        "--log-every-n-batches",
        str(runner.log_every_n_batches),
        "--perm-thread-workers",
        str(runner.perm_thread_workers),
    ]
    if runner.log_memory:
        cmd.append("--log-memory")
    else:
        cmd.append("--no-log-memory")
    if runner.stats_log_per_task:
        cmd.append("--stats-log-per-task")
    if features_only:
        cmd.append("--features-only")
    if use_cached_features:
        cmd.append("--use-cached-features")
    if rdms_only:
        cmd.append("--rdms-only")
    if use_cached_rdms:
        cmd.append("--use-cached-rdms")
    if runner.device:
        cmd.extend(["--device", runner.device])
    for override in runner.override:
        cmd.extend(["--override", override])
    return cmd


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    specs = args.group or default_groups()
    args.output_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("Starting layerwise RSA checkpoint processing")
    logger.info("Output root: %s", args.output_root)
    logger.info("In-process mode: %s", not args.subprocess)

    manifest = []
    failures = []
    total_checkpoints = 0
    completed_checkpoints = 0
    skipped_checkpoints = 0
    skip_existing = args.skip_existing and not args.force

    shared_ctx = None
    if not args.subprocess:
        from experiments.run_layerwise_hierarchy import build_shared_context, run_layerwise_analysis

        logger.info("Preloading neural RDMs, rank vectors, and stimulus paths...")
        shared_ctx = build_shared_context(args)

    if args.cpu_only and int(args.checkpoint_workers) > 1:
        all_jobs: list[tuple[str, int, int, Path, Path]] = []
        for spec_idx, spec in enumerate(specs, 1):
            label, pattern = parse_group(spec)
            logger.info("[%s/%s] Collecting '%s'", spec_idx, len(specs), label)
            checkpoints = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
            if not checkpoints:
                logger.warning("[%s/%s] No checkpoints matched for '%s'", spec_idx, len(specs), label)
                failures.append({"group": label, "pattern": pattern, "error": "no checkpoints matched"})
                continue
            total_checkpoints += len(checkpoints)
            out_dir = args.output_root / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, checkpoint in enumerate(checkpoints, 1):
                output = out_dir / f"{checkpoint.stem}_{idx:03d}.json"
                if skip_existing and output_is_complete(output):
                    skipped_checkpoints += 1
                    manifest.append({"group": label, "checkpoint": str(checkpoint), "output": str(output), "status": "skipped"})
                    continue
                all_jobs.append((label, idx, len(checkpoints), checkpoint, output))

        checkpoint_workers = max(1, int(args.checkpoint_workers))
        logger.info(
            "CPU-only global queue: jobs=%s checkpoint_workers=%s rdm_workers_per_checkpoint=%s stats_workers_per_checkpoint=%s",
            len(all_jobs),
            checkpoint_workers,
            args.rdm_workers,
            args.stats_workers,
        )

        if args.cpu_rdm_first and all_jobs:
            from joblib import Parallel, delayed

            def run_global_rdm_prepass(job: tuple[str, int, int, Path, Path]) -> dict[str, object]:
                label, _idx, _count, checkpoint, output = job
                try:
                    if args.subprocess:
                        cmd = build_subprocess_cmd(args, checkpoint, output, use_cached_features=True, rdms_only=True)
                        subprocess.run(cmd, check=True, cwd=ROOT)
                    else:
                        hargs = hierarchy_args_for_checkpoint(args, checkpoint, output)
                        hargs.use_cached_features = True
                        hargs.rdms_only = True
                        run_layerwise_analysis(hargs, shared=shared_ctx)
                    return {"status": "rdms_cached", "group": label, "checkpoint": str(checkpoint), "output": str(output)}
                except Exception as exc:
                    return {"status": "failed", "group": label, "checkpoint": str(checkpoint), "output": str(output), "error": str(exc)}

            logger.info("Global CPU RDM prepass starting")
            rdm_rows = Parallel(n_jobs=checkpoint_workers, backend="threading")(
                delayed(run_global_rdm_prepass)(job) for job in all_jobs
            )
            for row in rdm_rows:
                if row["status"] == "failed":
                    failures.append(row)
                    manifest.append(row)
                    logger.error("Global RDM prepass failed for %s: %s", row["checkpoint"], row.get("error"))
            if failures and not args.keep_going:
                payload = {"runs": manifest, "failures": failures}
                (args.output_root / "manifest.json").write_text(json.dumps(payload, indent=2))
                print(json.dumps(payload, indent=2))
                raise SystemExit(1)

        if all_jobs:
            from joblib import Parallel, delayed

            logger.info("Global CPU stats/scoring phase starting")
            commands = [
                (
                    build_subprocess_cmd(
                        args,
                        checkpoint,
                        output,
                        use_cached_features=True,
                        use_cached_rdms=args.cpu_rdm_first,
                    ),
                    label,
                    checkpoint,
                    output,
                )
                for label, _idx, _count, checkpoint, output in all_jobs
            ]
            rows = Parallel(n_jobs=checkpoint_workers, backend="threading")(
                delayed(run_subprocess_row)(cmd, row_label, checkpoint, output)
                for cmd, row_label, checkpoint, output in commands
            )
            for row in rows:
                manifest.append(row)
                if row.get("status") == "ok":
                    completed_checkpoints += 1
                    logger.info("[%s] Completed: %s", row["group"], row["output"])
                else:
                    failures.append(row)
                    logger.error("[%s] Failed: %s", row["group"], row.get("error"))

        logger.info(
            "Processing complete: %s/%s succeeded, %s skipped",
            completed_checkpoints,
            total_checkpoints,
            skipped_checkpoints,
        )
        payload = {"runs": manifest, "failures": failures}
        (args.output_root / "manifest.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        raise SystemExit(1 if failures else 0)

    for spec_idx, spec in enumerate(specs, 1):
        label, pattern = parse_group(spec)
        logger.info("[%s/%s] Processing '%s'", spec_idx, len(specs), label)

        checkpoints = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
        if not checkpoints:
            logger.warning("[%s/%s] No checkpoints matched for '%s'", spec_idx, len(specs), label)
            failures.append({"group": label, "pattern": pattern, "error": "no checkpoints matched"})
            continue

        logger.info("[%s/%s] Found %s checkpoint(s)", spec_idx, len(specs), len(checkpoints))
        total_checkpoints += len(checkpoints)

        jobs: list[tuple[int, Path, Path]] = []
        for idx, checkpoint in enumerate(checkpoints, 1):
            out_dir = args.output_root / label
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"{checkpoint.stem}_{idx:03d}.json"

            if skip_existing and output_is_complete(output):
                skipped_checkpoints += 1
                logger.info("[%s %s/%s] Skipping (exists): %s", label, idx, len(checkpoints), output.name)
                manifest.append(
                    {
                        "group": label,
                        "checkpoint": str(checkpoint),
                        "output": str(output),
                        "status": "skipped",
                    }
                )
                continue
            jobs.append((idx, checkpoint, output))

        if (args.gpu_first or args.features_only) and not args.cpu_only and jobs:
            logger.info("[%s] GPU feature extraction phase: %s checkpoint(s)", label, len(jobs))
            for idx, checkpoint, output in tqdm(jobs, desc=f"{label} gpu", leave=True):
                logger.info("[%s %s/%s] Extracting features: %s", label, idx, len(checkpoints), checkpoint.name)
                try:
                    if args.subprocess:
                        cmd = build_subprocess_cmd(args, checkpoint, output, features_only=True)
                        subprocess.run(cmd, check=True, cwd=ROOT)
                    else:
                        hargs = hierarchy_args_for_checkpoint(args, checkpoint, output)
                        hargs.features_only = True
                        run_layerwise_analysis(hargs, shared=shared_ctx)
                except Exception as exc:
                    row = {"group": label, "checkpoint": str(checkpoint), "output": str(output), "status": "failed", "error": str(exc)}
                    logger.error("[%s %s/%s] Feature extraction failed: %s", label, idx, len(checkpoints), exc)
                    failures.append(row)
                    manifest.append(row)
                    if not args.keep_going:
                        break
            if failures and not args.keep_going:
                break

        if args.features_only:
            for idx, checkpoint, output in jobs:
                manifest.append(
                    {
                        "group": label,
                        "checkpoint": str(checkpoint),
                        "output": str(output),
                        "status": "features_cached",
                    }
            )
            continue

        if args.cpu_only and args.cpu_rdm_first and jobs:
            from joblib import Parallel, delayed

            checkpoint_workers = max(1, int(args.checkpoint_workers))
            logger.info(
                "[%s] CPU RDM prepass: %s checkpoint(s), checkpoint_workers=%s, rdm_workers_per_checkpoint=%s",
                label,
                len(jobs),
                checkpoint_workers,
                args.rdm_workers,
            )

            def run_rdm_prepass(job: tuple[int, Path, Path]) -> dict[str, str]:
                idx, checkpoint, output = job
                try:
                    if args.subprocess:
                        cmd = build_subprocess_cmd(
                            args,
                            checkpoint,
                            output,
                            use_cached_features=True,
                            rdms_only=True,
                        )
                        subprocess.run(cmd, check=True, cwd=ROOT)
                    else:
                        hargs = hierarchy_args_for_checkpoint(args, checkpoint, output)
                        hargs.use_cached_features = True
                        hargs.rdms_only = True
                        run_layerwise_analysis(hargs, shared=shared_ctx)
                    return {"status": "rdms_cached", "group": label, "checkpoint": str(checkpoint), "output": str(output)}
                except Exception as exc:
                    return {"status": "failed", "group": label, "checkpoint": str(checkpoint), "output": str(output), "error": str(exc)}

            rdm_rows = Parallel(n_jobs=checkpoint_workers, backend="threading")(
                delayed(run_rdm_prepass)(job) for job in jobs
            )
            for row in rdm_rows:
                if row["status"] == "failed":
                    failures.append(row)
                    manifest.append(row)
                    logger.error("[%s] RDM prepass failed for %s: %s", label, row["checkpoint"], row.get("error"))
            if failures and not args.keep_going:
                break

        if args.cpu_only and int(args.checkpoint_workers) > 1 and jobs:
            from joblib import Parallel, delayed

            checkpoint_workers = max(1, int(args.checkpoint_workers))
            logger.info(
                "[%s] CPU stats/scoring phase in parallel: %s checkpoint(s), checkpoint_workers=%s, stats_workers_per_checkpoint=%s",
                label,
                len(jobs),
                checkpoint_workers,
                args.stats_workers,
            )
            commands = [
                (
                    build_subprocess_cmd(
                        args,
                        checkpoint,
                        output,
                        use_cached_features=True,
                        use_cached_rdms=args.cpu_rdm_first,
                    ),
                    label,
                    checkpoint,
                    output,
                )
                for _idx, checkpoint, output in jobs
            ]
            rows = Parallel(n_jobs=checkpoint_workers, backend="threading")(
                delayed(run_subprocess_row)(cmd, row_label, checkpoint, output)
                for cmd, row_label, checkpoint, output in commands
            )
            for row in rows:
                manifest.append(row)
                if row.get("status") == "ok":
                    completed_checkpoints += 1
                    logger.info("[%s] Completed: %s", label, row["output"])
                else:
                    failures.append(row)
                    logger.error("[%s] Failed: %s", label, row.get("error"))
            if failures and not args.keep_going:
                break
            continue

        for idx, checkpoint, output in tqdm(jobs, desc=f"{label} cpu" if args.gpu_first else label, leave=True):
            logger.info("[%s %s/%s] Processing: %s", label, idx, len(checkpoints), checkpoint.name)
            row = {"group": label, "checkpoint": str(checkpoint), "output": str(output)}
            try:
                if args.subprocess:
                    cmd = build_subprocess_cmd(
                        args,
                        checkpoint,
                        output,
                        use_cached_features=args.gpu_first or args.cpu_only,
                        use_cached_rdms=args.cpu_only and args.cpu_rdm_first,
                    )
                    row["command"] = cmd
                    subprocess.run(cmd, check=True, cwd=ROOT)
                else:
                    hargs = hierarchy_args_for_checkpoint(args, checkpoint, output)
                    hargs.use_cached_features = args.gpu_first or args.cpu_only
                    hargs.use_cached_rdms = args.cpu_only and args.cpu_rdm_first
                    run_layerwise_analysis(hargs, shared=shared_ctx)
                row["status"] = "ok"
                completed_checkpoints += 1
                logger.info("[%s %s/%s] Completed: %s", label, idx, len(checkpoints), output)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                logger.error("[%s %s/%s] Failed: %s", label, idx, len(checkpoints), exc)
                failures.append(row)
                if not args.keep_going:
                    manifest.append(row)
                    logger.error("Stopping due to failure (--keep-going not set)")
                    break
            manifest.append(row)
        if failures and not args.keep_going:
            break

    logger.info(
        "Processing complete: %s/%s succeeded, %s skipped",
        completed_checkpoints,
        total_checkpoints,
        skipped_checkpoints,
    )
    if failures:
        logger.warning("Failures recorded: %s", len(failures))

    payload = {"runs": manifest, "failures": failures}
    (args.output_root / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

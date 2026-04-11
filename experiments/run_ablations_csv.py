from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AblationRun:
    key: str
    run_name: str
    overrides: tuple[str, ...]


ABLATIONS: tuple[AblationRun, ...] = (
    AblationRun(
        key="E0",
        run_name="E0_vision_only",
        overrides=(
            "model.use_video=true",
            "model.use_muscle=false",
            "model.label_conditioning=false",
        ),
    ),
    AblationRun(
        key="E1",
        run_name="E1_muscle_only",
        overrides=(
            "model.use_video=false",
            "model.use_muscle=true",
            "model.label_conditioning=false",
        ),
    ),
    AblationRun(
        key="E2",
        run_name="E2_late_fusion",
        overrides=(
            "model.use_video=true",
            "model.use_muscle=true",
            "model.fusion_mode=late",
        ),
    ),
    AblationRun(
        key="E3",
        run_name="E3_egomuscle",
        overrides=(
            "model.use_video=true",
            "model.use_muscle=true",
            "model.fusion_mode=cross_attn",
        ),
    ),
    AblationRun(
        key="E4",
        run_name="E4_exocentric",
        overrides=(
            "model.use_video=true",
            "model.use_muscle=true",
            "model.fusion_mode=cross_attn",
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
    ),
    AblationRun(
        key="E5",
        run_name="E5_scrambled_temporal",
        overrides=(
            "model.use_video=true",
            "model.use_muscle=true",
            "model.fusion_mode=cross_attn",
            "data.train.scramble_video=true",
            "data.val.scramble_video=true",
            "data.test.scramble_video=true",
        ),
    ),
    AblationRun(
        key="E6",
        run_name="E6_coarse_labels",
        overrides=(
            "model.use_video=true",
            "model.use_muscle=false",
            "model.label_conditioning=true",
        ),
    ),
)


def coerce_number(value: str) -> int | float | str:
    value = value.strip()
    if value == "":
        return ""
    try:
        parsed = float(value)
    except ValueError:
        return value
    if parsed.is_integer():
        return int(parsed)
    return parsed


def load_metrics_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def last_non_empty_row(rows: list[dict[str, str]], prefix: str) -> dict[str, Any]:
    matching = [row for row in rows if any((key.startswith(prefix) and row.get(key, "").strip()) for key in row)]
    if not matching:
        return {}
    row = matching[-1]
    return {key: coerce_number(value) for key, value in row.items() if key.startswith(prefix) and value.strip()}


def best_val_row(rows: list[dict[str, str]]) -> dict[str, Any]:
    candidates = []
    for row in rows:
        raw_loss = row.get("val/loss", "").strip()
        if not raw_loss:
            continue
        try:
            loss = float(raw_loss)
        except ValueError:
            continue
        candidates.append((loss, row))
    if not candidates:
        return {}
    _, row = min(candidates, key=lambda item: item[0])
    return {key: coerce_number(value) for key, value in row.items() if key.startswith("val/") and value.strip()}


def find_latest_version_dir(log_root: Path, run_name: str) -> Path | None:
    run_root = log_root / run_name
    versions = sorted(path for path in run_root.glob("version_*") if path.is_dir())
    return versions[-1] if versions else None


def find_best_checkpoint(search_root: Path) -> str:
    matches = sorted(search_root.rglob("*.ckpt"))
    return str(matches[0]) if matches else ""


def dataset_overrides(root: Path) -> list[str]:
    overrides: list[str] = []
    for split in ("train", "val", "test"):
        overrides.extend(
            [
                f"data.{split}.clip_dir={root / split / 'clips'}",
                f"data.{split}.muscle_dir={root / split / 'muscles'}",
                f"data.{split}.metadata_path={root / split / 'metadata.json'}",
                f"data.{split}.frame_cache_dir={root / split / 'frames'}",
            ]
        )
    return overrides


def preflight_dataset_root(root: Path, *, label: str) -> None:
    required = (
        root / "train" / "clips",
        root / "train" / "muscles",
        root / "train" / "metadata.json",
        root / "val" / "clips",
        root / "test" / "clips",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"{label} dataset root is incomplete: {joined}")


def flatten_metrics(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key.replace('/', '_').replace('-', '_')}": value for key, value in values.items()}


def write_summary(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass


def stream_command(command: list[str], *, log_path: Path, prefix: str) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[{prefix}] log={log_path}", flush=True)
    print(f"[{prefix}] cmd={' '.join(command)}", flush=True)

    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(f"[{prefix}] {line}")
                log_handle.write(line)
                sys.stdout.flush()
                log_handle.flush()
            return process.wait()
        except KeyboardInterrupt:
            log_handle.write("\n[runner] interrupted by user\n")
            log_handle.flush()
            _terminate_process_group(process)
            raise


def build_twente_command(
    checkpoint_path: Path,
    *,
    config: Path,
    twente_root: Path,
    output_path: Path,
    csv_output_path: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    target_mode: str,
    overrides: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "egomuscle.eval.twente_eval",
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(config),
        "--twente-root",
        str(twente_root),
        "--output",
        str(output_path),
        "--csv-output",
        str(csv_output_path),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--target-mode",
        target_mode,
    ]
    for override in overrides:
        command.extend(["--override", override])
    return command


def flatten_twente_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mean_rho",
        "std_rho",
        "mean_acc",
        "mean_r2",
        "num_clips",
        "num_subjects",
        "feature_dim",
        "target_dim",
        "target_mode",
        "representation_mode",
    )
    return {f"twente_{key}": summary[key] for key in keys if key in summary}


def run_twente_eval(
    *,
    checkpoint_path: Path,
    config: Path,
    twente_root: Path,
    run_root: Path,
    overrides: list[str],
    device: str,
    batch_size: int,
    num_workers: int,
    target_mode: str,
) -> dict[str, Any]:
    output_path = run_root / "twente_eval.json"
    csv_output_path = run_root / "twente_eval_folds.csv"
    log_path = run_root / "twente_eval.log"
    command = build_twente_command(
        checkpoint_path,
        config=config,
        twente_root=twente_root,
        output_path=output_path,
        csv_output_path=csv_output_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        target_mode=target_mode,
        overrides=overrides,
    )
    started_at = datetime.now().astimezone()
    returncode = stream_command(command, log_path=log_path, prefix=f"{run_root.name}:twente")
    finished_at = datetime.now().astimezone()

    row: dict[str, Any] = {
        "twente_status": "ok" if returncode == 0 else "failed",
        "twente_returncode": returncode,
        "twente_started_at": started_at.isoformat(),
        "twente_finished_at": finished_at.isoformat(),
        "twente_duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "twente_output_path": str(output_path),
        "twente_folds_csv_path": str(csv_output_path),
        "twente_log": str(log_path),
    }
    if returncode == 0 and output_path.exists():
        summary = json.loads(output_path.read_text())
        row.update(flatten_twente_summary(summary))
    return row


def build_command(
    ablation: AblationRun,
    *,
    config: Path,
    ckpt_dir: Path,
    log_dir: Path,
    extra_overrides: list[str],
    dataset_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "egomuscle.training.train",
        "--config",
        str(config),
        "--override",
        f"output_dir={ckpt_dir}",
        "--override",
        f"logging.save_dir={log_dir}",
        "--override",
        f"logging.run_name={ablation.run_name}",
    ]
    for override in ablation.overrides:
        command.extend(["--override", override])
    for override in dataset_overrides(dataset_root):
        command.extend(["--override", override])
    for override in extra_overrides:
        command.extend(["--override", override])
    return command


def run_one(
    ablation: AblationRun,
    *,
    config: Path,
    root: Path,
    extra_overrides: list[str],
    ego_root: Path,
    exo_root: Path,
    twente_root: Path | None,
    skip_twente_eval: bool,
    twente_device: str,
    twente_batch_size: int,
    twente_num_workers: int,
    twente_target_mode: str,
    reuse_existing_train: bool,
) -> dict[str, Any]:
    run_root = root / ablation.key
    ckpt_dir = run_root / "checkpoints"
    log_dir = run_root / "logs"
    train_log = run_root / "train.log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = exo_root if ablation.key == "E4" else ego_root
    version_dir = find_latest_version_dir(log_dir, ablation.run_name)
    checkpoint_path = find_best_checkpoint(run_root)
    if reuse_existing_train and version_dir is not None and checkpoint_path:
        started_at = datetime.now().astimezone()
        finished_at = started_at
        returncode = 0
    else:
        command = build_command(
            ablation,
            config=config,
            ckpt_dir=ckpt_dir,
            log_dir=log_dir,
            extra_overrides=extra_overrides,
            dataset_root=dataset_root,
        )
        started_at = datetime.now().astimezone()
        returncode = stream_command(command, log_path=train_log, prefix=ablation.key)
        finished_at = datetime.now().astimezone()
        version_dir = find_latest_version_dir(log_dir, ablation.run_name)
        checkpoint_path = find_best_checkpoint(run_root)

    metrics_path = version_dir / "metrics.csv" if version_dir is not None else None
    metrics_rows = load_metrics_rows(metrics_path) if metrics_path is not None and metrics_path.exists() else []
    best_val = best_val_row(metrics_rows)
    last_test = last_non_empty_row(metrics_rows, "test/")
    last_train = last_non_empty_row(metrics_rows, "train/")

    row: dict[str, Any] = {
        "ablation": ablation.key,
        "run_name": ablation.run_name,
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "config": str(config),
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "train_log": str(train_log),
        "metrics_path": str(metrics_path) if metrics_path is not None else "",
        "checkpoint_path": checkpoint_path,
        "overrides": json.dumps(list(ablation.overrides + tuple(extra_overrides))),
    }
    row.update(flatten_metrics("best", best_val))
    row.update(flatten_metrics("last_test", last_test))
    row.update(flatten_metrics("last_train", last_train))

    twente_overrides = list(ablation.overrides + tuple(extra_overrides))
    twente_overrides.append("training.compile=true")
    video_disabled = "model.use_video=false" in ablation.overrides or "model.use_video=false" in extra_overrides
    label_conditioned = "model.label_conditioning=true" in ablation.overrides or "model.label_conditioning=true" in extra_overrides
    if row["status"] != "ok":
        row.update({"twente_status": "skipped_train_failed", "twente_returncode": ""})
    elif video_disabled:
        row.update({"twente_status": "skipped_video_disabled", "twente_returncode": ""})
    elif label_conditioned:
        row.update({"twente_status": "skipped_label_conditioning", "twente_returncode": ""})
    elif skip_twente_eval:
        row.update({"twente_status": "skipped_disabled", "twente_returncode": ""})
    elif twente_root is None or not twente_root.exists():
        row.update({"twente_status": "skipped_missing_root", "twente_returncode": ""})
    elif not checkpoint_path:
        row.update({"twente_status": "skipped_missing_checkpoint", "twente_returncode": ""})
    else:
        row.update(
            run_twente_eval(
                checkpoint_path=Path(checkpoint_path),
                config=config,
                twente_root=twente_root,
                run_root=run_root,
                overrides=twente_overrides,
                device=twente_device,
                batch_size=twente_batch_size,
                num_workers=twente_num_workers,
                target_mode=twente_target_mode,
            )
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all EgoMuscle ablations and summarize metrics to CSV.")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--root", type=Path, help="Sweep output root. Defaults to experiments/results/ablations/<timestamp>.")
    parser.add_argument("--ego-root", type=Path, default=Path("data/processed_amass"))
    parser.add_argument("--exo-root", type=Path, default=Path("data/processed_exo"))
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--jsonl-name", default="summary.jsonl")
    parser.add_argument("--only", nargs="+", choices=[ablation.key for ablation in ABLATIONS])
    parser.add_argument("--override", action="append", default=[], help="Extra training override, repeated as needed.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed ablation instead of stopping.")
    parser.add_argument("--reuse-existing-train", action="store_true")
    parser.add_argument("--twente-root", type=Path, default=Path("data/processed_real/twente"))
    parser.add_argument("--skip-twente-eval", action="store_true")
    parser.add_argument("--twente-device", default="cuda")
    parser.add_argument("--twente-batch-size", type=int, default=8)
    parser.add_argument("--twente-num-workers", type=int, default=0)
    parser.add_argument("--twente-target-mode", choices=["mean", "flatten"], default="mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.root or Path("experiments/results/ablations") / stamp
    root.mkdir(parents=True, exist_ok=True)

    selected = [ablation for ablation in ABLATIONS if args.only is None or ablation.key in set(args.only)]
    needs_ego = any(ablation.key != "E4" for ablation in selected)
    needs_exo = any(ablation.key == "E4" for ablation in selected)
    if needs_ego:
        preflight_dataset_root(args.ego_root, label="ego")
    if needs_exo:
        preflight_dataset_root(args.exo_root, label="exo")
    summary_csv = root / args.summary_name
    summary_jsonl = root / args.jsonl_name
    rows: list[dict[str, Any]] = []

    for ablation in selected:
        print(f"[run] {ablation.key} -> {ablation.run_name}", flush=True)
        row = run_one(
            ablation,
            config=args.config,
            root=root,
            extra_overrides=args.override,
            ego_root=args.ego_root,
            exo_root=args.exo_root,
            twente_root=args.twente_root,
            skip_twente_eval=args.skip_twente_eval,
            twente_device=args.twente_device,
            twente_batch_size=args.twente_batch_size,
            twente_num_workers=args.twente_num_workers,
            twente_target_mode=args.twente_target_mode,
            reuse_existing_train=args.reuse_existing_train,
        )
        rows.append(row)
        append_jsonl(summary_jsonl, row)
        write_summary(summary_csv, rows)
        print(
            f"[done] {ablation.key} status={row['status']} best_val_loss={row.get('best_val_loss', '')} "
            f"test_loss={row.get('last_test_test_loss', '')} twente={row.get('twente_mean_rho', row.get('twente_status', ''))}",
            flush=True,
        )
        if row["status"] != "ok" and not args.keep_going:
            raise SystemExit(row["returncode"])

    print(f"summary_csv={summary_csv}")
    print(f"summary_jsonl={summary_jsonl}")


if __name__ == "__main__":
    main()

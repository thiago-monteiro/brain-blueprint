from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Candidate:
    generation: int
    index: int
    vector: np.ndarray
    overrides: list[str]
    run_dir: Path


WEIGHT_NAMES = (
    "muscle_nll",
    "video_latent",
    "fast_kl",
    "precision",
    "entropy",
    "homeostasis",
    "capacity",
    "plastic",
)
SLOT_CHOICES = (2, 4, 8, 16)
PBIT_CHOICES = (64, 128, 256, 512)
RANK_CHOICES = (0, 2, 4, 8)
QUANT_CHOICES = ("null", "26", "16", "8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenES search over compact SMFE loss/design vectors.")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--root", type=Path, default=Path("experiments/results/es_loss_meta_search"))
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", action="store_true", help="Actually launch candidate training jobs.")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--fitness-metric", default="val/loss", help="Metric column to minimize from metrics.csv.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def choice(values: tuple[Any, ...], scalar: float) -> Any:
    idx = int(np.clip(round((math.tanh(float(scalar)) + 1.0) * 0.5 * (len(values) - 1)), 0, len(values) - 1))
    return values[idx]


def vector_to_overrides(vector: np.ndarray) -> list[str]:
    overrides = [
        "training.loss_name=smfe",
        "model.predictive_distribution=gaussian",
        "model.video_latent_prediction=true",
        "model.fast_memory.enabled=true",
        "model.pbit.enabled=true",
        "model.slow_adapter.enabled=true",
        "training.precision=32-true",
    ]
    for name, raw in zip(WEIGHT_NAMES, vector[: len(WEIGHT_NAMES)], strict=True):
        value = float(np.exp(raw))
        overrides.append(f"training.loss_weights.{name}={value:.8g}")
    overrides.extend(
        [
            f"model.fast_memory.num_slots={choice(SLOT_CHOICES, vector[8])}",
            f"model.pbit.num_bits={choice(PBIT_CHOICES, vector[9])}",
            f"model.pbit.temperature={float(np.clip(np.exp(vector[10]), 0.2, 2.0)):.8g}",
            f"model.slow_adapter.rank={choice(RANK_CHOICES, vector[11])}",
            f"model.slow_adapter.quantization_levels={choice(QUANT_CHOICES, vector[12])}",
            "model.slow_adapter.quantization_mode=qat",
        ]
    )
    return overrides


def read_metric(run_dir: Path, metric_name: str) -> float | None:
    metric_paths = sorted((run_dir / "logs").rglob("metrics.csv"))
    if not metric_paths:
        return None
    best: float | None = None
    with metric_paths[-1].open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(metric_name, "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            best = value if best is None else min(best, value)
    return best


def build_candidate(generation: int, index: int, vector: np.ndarray, root: Path) -> Candidate:
    run_dir = root / f"gen_{generation:03d}" / f"cand_{index:03d}"
    return Candidate(
        generation=generation,
        index=index,
        vector=vector,
        overrides=vector_to_overrides(vector),
        run_dir=run_dir,
    )


def run_candidate(candidate: Candidate, args: argparse.Namespace) -> int:
    candidate.run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "egomuscle.training.train",
        "--config",
        str(args.config),
        "--override",
        f"output_dir={candidate.run_dir / 'checkpoints'}",
        "--override",
        f"logging.save_dir={candidate.run_dir / 'logs'}",
        "--override",
        f"logging.run_name=es_g{candidate.generation:03d}_c{candidate.index:03d}",
        "--override",
        f"training.max_epochs={args.max_epochs}",
    ]
    if args.max_steps is not None:
        command.extend(["--override", f"training.max_steps={args.max_steps}"])
    for override in (*candidate.overrides, *args.override):
        command.extend(["--override", override])
    (candidate.run_dir / "command.json").write_text(json.dumps({"command": command, "overrides": candidate.overrides}, indent=2))
    with (candidate.run_dir / "train.log").open("w") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode


def write_generation(root: Path, generation: int, rows: list[dict[str, Any]]) -> None:
    path = root / f"generation_{generation:03d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.root.mkdir(parents=True, exist_ok=True)
    mean = np.array([0.0, -2.3, -4.6, -3.0, -4.6, -4.6, -6.9, -6.9, 0.0, 0.0, math.log(0.67), 0.0, 0.0], dtype=np.float64)
    all_rows: list[dict[str, Any]] = []

    for generation in range(args.generations):
        noise = rng.normal(size=(args.population, mean.size))
        candidates = [build_candidate(generation, idx, mean + args.sigma * noise[idx], args.root) for idx in range(args.population)]
        rows: list[dict[str, Any]] = []
        fitness = np.zeros(args.population, dtype=np.float64)
        for idx, candidate in enumerate(candidates):
            if args.train:
                returncode = run_candidate(candidate, args)
                metric = read_metric(candidate.run_dir, args.fitness_metric)
            else:
                returncode = 0
                metric = float(np.linalg.norm(candidate.vector - mean))
            if metric is None:
                metric = float("inf")
            score = -float(metric)
            fitness[idx] = score
            row = {
                "generation": generation,
                "candidate": idx,
                "returncode": returncode,
                "metric_name": args.fitness_metric,
                "metric": metric,
                "fitness": score,
                "run_dir": str(candidate.run_dir),
                "overrides": json.dumps(candidate.overrides),
            }
            rows.append(row)
            all_rows.append(row)

        finite = np.isfinite(fitness)
        if finite.any():
            ranked = fitness.copy()
            ranked[~finite] = np.min(ranked[finite]) - 1.0
            normalized = (ranked - ranked.mean()) / (ranked.std() + 1e-8)
            gradient = (noise.T @ normalized) / (args.population * args.sigma)
            mean = mean + args.lr * gradient

        write_generation(args.root, generation, rows)
        (args.root / "mean_vector.json").write_text(json.dumps({"generation": generation, "mean": mean.tolist()}, indent=2))

    summary_path = args.root / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    best = max(all_rows, key=lambda row: float(row["fitness"])) if all_rows else None
    selected = {"best": best, "final_mean_overrides": vector_to_overrides(mean)}
    (args.root / "selected_config.json").write_text(json.dumps(selected, indent=2))
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()

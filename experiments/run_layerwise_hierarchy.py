from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import squareform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.layerwise_cache import (
    layerwise_run_cache_dir,
    load_or_build,
    neural_rank_cache_path,
    stimuli_list_cache_path,
)
from experiments.layerwise_stats import (
    fast_spearman_vec,
    permutation_test_rdm,
    rank_vector,
)
from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.eval.rdm import compute_rdm
from egomuscle.eval.twente_eval import load_lightning_module
from egomuscle.training.train import apply_override, load_config

logger = logging.getLogger("layerwise_rsa")


def configure_logging(level: str = "INFO", log_memory: bool = True) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("layerwise_rsa").disabled = False


def rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def log_phase(message: str, *, log_memory: bool = True) -> None:
    suffix = f" rss_mb={rss_mb():.1f}" if log_memory and rss_mb() is not None else ""
    logger.info("%s%s", message, suffix)


class StaticImageDataset(Dataset):
    """Wrap image stimuli as static videos for the VideoMAE encoder."""

    def __init__(self, image_paths: list[Path], n_frames: int, image_size: int) -> None:
        self.image_paths = image_paths
        self.n_frames = n_frames
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        frames = img_tensor.unsqueeze(0).repeat(self.n_frames, 1, 1, 1)
        return {
            "frames": frames,
            "muscle": None,
            "activity_id": 0,
            "clip_id": path.stem,
            "activity": None,
        }


def collate_static_images(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frames": torch.stack([item["frames"] for item in items], dim=0),
        "muscle": None,
        "activity_id": torch.tensor([item["activity_id"] for item in items], dtype=torch.long),
        "clip_id": [item["clip_id"] for item in items],
        "activity": [item["activity"] for item in items],
        "metadata": [None for _ in items],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layerwise RSA for a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--stimuli-dir", type=Path, help="Directory containing benchmark stimuli.")
    parser.add_argument("--stimuli-list", type=Path, help="Ordered text file of stimulus filenames; keeps model/neural ordering aligned.")
    parser.add_argument("--neural-dir", type=Path, default=Path("egomuscle/eval/algonauts2025_rdms"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/layerwise_hierarchy.json"))
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=30,
        help="Permutation count per layer×ROI (30 is the default; p-values stabilize slowly above this).",
    )
    parser.add_argument(
        "--perm-thread-workers",
        type=int,
        default=8,
        help="Thread pool size for null permutations when outer stats joblib is active.",
    )
    parser.add_argument("--permutation-mode", choices=("pair_shuffle", "mantel"), default="pair_shuffle")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--stats-seed", type=int, default=0)
    parser.add_argument(
        "--stats-workers",
        type=int,
        default=0,
        help="Parallel workers for stats (0 = conservative memory-aware default). Region-level tasks run in parallel.",
    )
    parser.add_argument(
        "--rdm-workers",
        type=int,
        default=0,
        help="Parallel workers for CPU RDM construction (0 = one worker per layer, capped by CPUs).",
    )
    parser.add_argument("--max-stat-regions", type=int, default=None, help="Optional debugging limit for expensive stats.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-every-n-batches", type=int, default=25)
    parser.add_argument("--log-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stats-log-per-task", action="store_true", help="Log each layer×region stats task at INFO.")
    parser.add_argument("--features-only", action="store_true", help="Run GPU inference, save layer features, then stop.")
    parser.add_argument("--use-cached-features", action="store_true", help="Skip GPU inference and load saved layer features.")
    parser.add_argument("--rdms-only", action="store_true", help="Build/save model RDMs from cached features, then stop.")
    parser.add_argument("--use-cached-rdms", action="store_true", help="Skip model RDM construction and load saved RDMs.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def standardize_features(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return (matrix - mean) / std


def resolve_rdm_workers(requested: int, n_layers: int) -> int:
    if requested > 0:
        return max(1, requested)
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, n_layers))


def resolve_layerwise_stats_workers(requested: int) -> int:
    if requested > 0:
        return requested
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu))


def compute_and_save_layer_rdm(layer_name: str, feature_path: Path, rdm_path: Path) -> tuple[str, Path, tuple[int, int]]:
    features = np.load(feature_path, mmap_mode="r")
    model_rdm = compute_rdm(features).astype(np.float32, copy=False)
    rdm_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(rdm_path, model_rdm)
    return layer_name, rdm_path, model_rdm.shape


def layerwise_feature_metadata_path(feature_dir: Path) -> Path:
    return feature_dir / "metadata.json"


def rdm_vec(rdm: np.ndarray) -> np.ndarray:
    return squareform(rdm, checks=False)


def stable_seed(*parts: object, base: int = 0) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int((int(digest[:8], 16) + int(base)) % (2**32 - 1))


def bootstrap_ci(values: list[float], *, n_bootstrap: int, seed: int) -> dict[str, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if len(clean) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if len(clean) == 1 or n_bootstrap <= 0:
        value = float(clean.mean())
        return {"mean": value, "std": 0.0, "ci_low": value, "ci_high": value}
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(n_bootstrap, len(clean)), replace=True).mean(axis=1)
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def noise_ceiling(subject_rdms: dict[str, np.ndarray]) -> dict[str, Any]:
    subjects = sorted(subject_rdms)
    if len(subjects) < 2:
        return {"num_subjects": len(subjects), "lower": None, "upper": None, "subject_rows": []}
    rows = []
    all_rdms = [subject_rdms[subj] for subj in subjects]
    group_all = np.mean(all_rdms, axis=0)
    group_all_vec = rdm_vec(group_all)
    for idx, subject in enumerate(subjects):
        subj_vec = rdm_vec(subject_rdms[subject])
        others = [rdm for j, rdm in enumerate(all_rdms) if j != idx]
        lower_vec = rdm_vec(np.mean(others, axis=0))
        rows.append(
            {
                "subject": subject,
                "lower": fast_spearman_vec(subj_vec, lower_vec),
                "upper": fast_spearman_vec(subj_vec, group_all_vec),
            }
        )
    return {
        "num_subjects": len(subjects),
        "lower": float(np.mean([row["lower"] for row in rows])),
        "upper": float(np.mean([row["upper"] for row in rows])),
        "subject_rows": rows,
    }


def subject_uncertainty(model_rdm: np.ndarray, subject_rdms: dict[str, np.ndarray], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    model_vec = rdm_vec(model_rdm)
    rows = []
    scores = []
    for subject, neural_rdm in sorted(subject_rdms.items()):
        score = fast_spearman_vec(model_vec, rdm_vec(neural_rdm))
        scores.append(score)
        rows.append({"subject": subject, "rho": score})
    return {"subjects": rows, **bootstrap_ci(scores, n_bootstrap=n_bootstrap, seed=seed)}


def load_neural_rank_vectors(neural_rdms: dict[str, np.ndarray], neural_dir: Path) -> dict[str, np.ndarray]:
    ranks: dict[str, np.ndarray] = {}
    for region, neural_rdm in neural_rdms.items():
        vec = rdm_vec(neural_rdm)
        cache_path = neural_rank_cache_path(neural_dir, region, len(vec))

        def build() -> np.ndarray:
            return rank_vector(vec)

        if cache_path.exists():
            ranks[region] = np.load(cache_path)
        else:
            built = build()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, built)
            ranks[region] = built
    return ranks


def compute_region_statistics(
    layer_name: str,
    region: str,
    model_rdm: np.ndarray,
    neural_vec: np.ndarray,
    neural_ranks: np.ndarray,
    observed_rho: float,
    subject_rdms: dict[str, np.ndarray],
    noise_ceiling: dict[str, Any] | None,
    *,
    n_permutations: int,
    n_bootstrap: int,
    permutation_mode: str,
    stats_seed: int,
    perm_workers: int,
    nested_outer_parallel: bool = False,
    perm_thread_workers: int = 8,
) -> dict[str, Any]:
    return {
        "permutation": permutation_test_rdm(
            model_rdm,
            neural_vec,
            observed_rho,
            n_permutations=n_permutations,
            seed=stable_seed("perm", layer_name, region, base=stats_seed),
            mode=permutation_mode,
            n_workers=perm_workers,
            neural_ranks=neural_ranks,
            nested_outer_parallel=nested_outer_parallel,
            perm_thread_workers=perm_thread_workers,
        ),
        "subject_uncertainty": subject_uncertainty(
            model_rdm,
            subject_rdms,
            n_bootstrap=n_bootstrap,
            seed=stable_seed("bootstrap", layer_name, region, base=stats_seed),
        ),
        "noise_ceiling": noise_ceiling,
        "noise_ceiling_normalized_rho": (
            float(observed_rho) / float(noise_ceiling["upper"])
            if noise_ceiling is not None and noise_ceiling.get("upper") not in (None, 0.0)
            else None
        ),
    }


def build_dataset(config: dict[str, Any], split: str) -> EgoMuscleDataset:
    split_cfg = config["data"][split]
    return EgoMuscleDataset(
        clip_dir=split_cfg["clip_dir"],
        muscle_dir=split_cfg.get("muscle_dir"),
        metadata_path=split_cfg.get("metadata_path"),
        n_frames=int(config["data"].get("n_frames", 16)),
        image_size=int(config["data"].get("image_size", 224)),
        muscle_dim=config["data"].get("muscle_dim"),
        require_muscle=True,
        scramble_video=bool(split_cfg.get("scramble_video", False)),
        temporal_sample_mode="sparse_uniform",
        muscle_time_offset=int(split_cfg.get("muscle_time_offset", 0)),
        muscle_noise_std=0.0,
        frame_cache_dir=None,
        full_cache_dir=split_cfg.get("full_cache_dir"),
        write_frame_cache=False,
        is_train=False,
    )


def resolve_stimuli(stimuli_dir: Path, stimuli_list: Path | None) -> list[Path]:
    if stimuli_list is not None:
        lines = [line.strip() for line in stimuli_list.read_text().splitlines() if line.strip()]
        filename_index: dict[str, Path] = {}
        for path in stimuli_dir.rglob("*"):
            if path.is_file():
                filename_index.setdefault(path.name, path)
        paths: list[Path] = []
        for line in lines:
            line_path = Path(line)
            candidates = [
                stimuli_dir / line,
                stimuli_dir / "Scene_Stimuli" / "Presented_Stimuli" / line,
                stimuli_dir / "Scene_Stimuli" / "Original_Images" / line,
            ]
            if Path(line).suffix == "":
                for indexed in filename_index.values():
                    if indexed.stem == line:
                        candidates.append(indexed)
            else:
                indexed = filename_index.get(line_path.name)
                if indexed is not None:
                    candidates.append(indexed)
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    paths.append(candidate)
                    break
            else:
                raise FileNotFoundError(f"Stimulus '{line}' from {stimuli_list} was not found under {stimuli_dir}")
        return paths

    image_paths = sorted(list(stimuli_dir.rglob("*.jpg")) + list(stimuli_dir.rglob("*.jpeg")) + list(stimuli_dir.rglob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No stimuli images were found under {stimuli_dir}")
    return image_paths


def resolve_stimuli_cached(stimuli_dir: Path, stimuli_list: Path | None) -> list[Path]:
    if stimuli_list is None:
        return resolve_stimuli(stimuli_dir, None)
    cache_path = stimuli_list_cache_path(stimuli_dir, stimuli_list)

    def build() -> list[str]:
        return [str(path) for path in resolve_stimuli(stimuli_dir, stimuli_list)]

    return [Path(path) for path in load_or_build(cache_path, build)]


@dataclass
class LayerwiseSharedContext:
    neural_rdms: dict[str, np.ndarray]
    neural_vecs: dict[str, np.ndarray]
    neural_ranks: dict[str, np.ndarray]
    stimuli_paths: list[Path] | None = None


def build_shared_context(args: argparse.Namespace) -> LayerwiseSharedContext:
    neural_rdms = {path.stem: np.load(path) for path in sorted(args.neural_dir.glob("*.npy"))}
    neural_vecs = {region: rdm_vec(rdm) for region, rdm in neural_rdms.items()}
    neural_ranks = load_neural_rank_vectors(neural_rdms, args.neural_dir)
    stimuli_paths = None
    if args.stimuli_dir is not None:
        stimuli_paths = resolve_stimuli_cached(args.stimuli_dir, args.stimuli_list)
    return LayerwiseSharedContext(
        neural_rdms=neural_rdms,
        neural_vecs=neural_vecs,
        neural_ranks=neural_ranks,
        stimuli_paths=stimuli_paths,
    )


def run_layerwise_analysis(args: argparse.Namespace, shared: LayerwiseSharedContext | None = None) -> dict[str, Any]:
    configure_logging(args.log_level, log_memory=args.log_memory)
    log_mem = bool(args.log_memory)
    t0 = time.perf_counter()
    device = torch.device(args.device)
    log_phase(
        f"startup checkpoint={args.checkpoint} device={device} batch_size={args.batch_size} "
        f"num_workers={args.num_workers}",
        log_memory=log_mem,
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if shared is None:
        log_phase("loading shared neural context", log_memory=log_mem)
        shared = build_shared_context(args)
    neural_rdms = shared.neural_rdms
    neural_vecs = shared.neural_vecs
    neural_ranks = shared.neural_ranks
    if not neural_rdms:
        raise FileNotFoundError(
            "Layerwise hierarchy requires neural RDM .npy files under "
            f"{args.neural_dir}."
        )

    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)

    if args.stimuli_dir is not None:
        image_paths = shared.stimuli_paths or resolve_stimuli_cached(args.stimuli_dir, args.stimuli_list)
        dataset = StaticImageDataset(
            image_paths=image_paths,
            n_frames=int(config["data"].get("n_frames", 16)),
            image_size=int(config["data"].get("image_size", 224)),
        )
        collate_fn = collate_static_images
    else:
        dataset = build_dataset(config, args.split)
        collate_fn = collate_egomuscle
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True
    loader = DataLoader(**loader_kwargs)
    n_stimuli = len(dataset)
    feature_dir = layerwise_run_cache_dir(args.checkpoint, args.output, n_stimuli, "features")
    feature_paths: dict[str, Path] = {}
    feature_dims: dict[str, int] = {}

    if getattr(args, "use_cached_features", False):
        metadata_path = layerwise_feature_metadata_path(feature_dir)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing cached layerwise feature metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        clip_ids = list(metadata["clip_ids"])
        activities = list(metadata["activities"])
        for feature_path in sorted(feature_dir.glob("*.npy")):
            features = np.load(feature_path, mmap_mode="r")
            feature_paths[feature_path.stem] = feature_path
            feature_dims[feature_path.stem] = int(features.shape[1])
            log_phase(f"feature_bank cached layer={feature_path.stem} path={feature_path} shape={features.shape}", log_memory=log_mem)
    else:
        log_phase("loading checkpoint", log_memory=log_mem)
        module = load_lightning_module(args.checkpoint, args.config, args.override, device)
        log_phase(f"inference start n_stimuli={n_stimuli} n_batches={len(loader)}", log_memory=log_mem)

        layerwise: dict[str, list[np.ndarray]] = {}
        clip_ids: list[str] = []
        activities: list[str | None] = []

        module.eval()
        use_autocast = device.type == "cuda"
        infer_t0 = time.perf_counter()
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(loader, desc="  processing batches", leave=False)):
                frames = batch["frames"].to(device, non_blocking=use_autocast)
                muscle = None if batch.get("muscle") is None else batch["muscle"].to(device, non_blocking=use_autocast)
                activity_ids = batch.get("activity_id")
                if activity_ids is not None:
                    activity_ids = activity_ids.to(device, non_blocking=use_autocast)
                with torch.autocast(device_type=device.type, enabled=use_autocast):
                    outputs = module.model(
                        frames=frames,
                        muscle=muscle,
                        activity_ids=activity_ids,
                        mask_ratio=0.0,
                        return_layerwise_video=True,
                    )
                if outputs.layerwise_video_repr is None:
                    raise ValueError("Model did not return layerwise video representations.")
                for name, tensor in outputs.layerwise_video_repr.items():
                    layerwise.setdefault(name, []).append(tensor.mean(dim=1).detach().cpu().numpy())
                clip_ids.extend(batch["clip_id"])
                activities.extend(batch["activity"])
                if (batch_idx + 1) % max(int(args.log_every_n_batches), 1) == 0:
                    elapsed = max(time.perf_counter() - infer_t0, 1e-6)
                    rate = (batch_idx + 1) * args.batch_size / elapsed
                    log_phase(
                        f"inference batch={batch_idx + 1}/{len(loader)} clips_per_sec≈{rate:.1f}",
                        log_memory=log_mem,
                    )

        infer_elapsed = time.perf_counter() - infer_t0
        log_phase(f"inference done elapsed_s={infer_elapsed:.1f} layers={len(layerwise)}", log_memory=log_mem)

        for name, parts in list(layerwise.items()):
            features = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            feature_path = feature_dir / f"{name}.npy"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, features)
            feature_paths[name] = feature_path
            feature_dims[name] = int(features.shape[1])
            nbytes = features.nbytes
            log_phase(
                f"feature_bank layer={name} path={feature_path} shape={features.shape} mb={nbytes / 1e6:.1f}",
                log_memory=log_mem,
            )
            del features
            del layerwise[name]
        layerwise.clear()
        layerwise_feature_metadata_path(feature_dir).write_text(
            json.dumps({"clip_ids": clip_ids, "activities": activities}, indent=2)
        )
        if device.type == "cuda":
            del module
            torch.cuda.empty_cache()
        gc.collect()

    if getattr(args, "features_only", False):
        log_phase(f"features_only done feature_cache_dir={feature_dir}", log_memory=log_mem)
        return {
            "checkpoint": str(args.checkpoint),
            "num_clips": len(clip_ids),
            "feature_cache_dir": str(feature_dir),
            "layers": sorted(feature_paths),
        }

    stats_workers = resolve_layerwise_stats_workers(args.stats_workers)
    layer_results: dict[str, Any] = {}
    best_by_region: dict[str, dict[str, float | str]] = {}
    shape_mismatches: dict[str, dict[str, list[int]]] = {}

    from joblib import Parallel, delayed

    rdm_dir = layerwise_run_cache_dir(args.checkpoint, args.output, n_stimuli, "rdms")
    if getattr(args, "use_cached_rdms", False):
        rdm_paths = {path.stem: path for path in sorted(rdm_dir.glob("*.npy"))}
        missing = sorted(set(feature_paths) - set(rdm_paths))
        if missing:
            raise FileNotFoundError(f"Missing cached RDMs for {len(missing)} layer(s) under {rdm_dir}: {missing[:5]}")
        rdm_workers = 0
        log_phase(f"rdm cached layers={len(rdm_paths)} cache_dir={rdm_dir}", log_memory=log_mem)
    else:
        rdm_workers = resolve_rdm_workers(args.rdm_workers, len(feature_paths))
        log_phase(f"rdm start layers={len(feature_paths)} rdm_workers={rdm_workers} cache_dir={rdm_dir}", log_memory=log_mem)
        rdm_results = Parallel(n_jobs=rdm_workers, backend="loky")(
            delayed(compute_and_save_layer_rdm)(layer_name, feature_path, rdm_dir / f"{layer_name}.npy")
            for layer_name, feature_path in feature_paths.items()
        )
        rdm_paths = {layer_name: rdm_path for layer_name, rdm_path, _shape in rdm_results}
        for layer_name, _rdm_path, shape in rdm_results:
            log_phase(f"rdm layer={layer_name} shape={shape} saved={_rdm_path}", log_memory=log_mem)

    if getattr(args, "rdms_only", False):
        log_phase(f"rdms_only done rdm_cache_dir={rdm_dir}", log_memory=log_mem)
        return {
            "checkpoint": str(args.checkpoint),
            "num_clips": len(clip_ids),
            "feature_cache_dir": str(feature_dir),
            "rdm_cache_dir": str(rdm_dir),
            "layers": sorted(rdm_paths),
        }

    score_tasks: list[tuple[str, str, Path, float]] = []
    for layer_name, rdm_path in rdm_paths.items():
        model_rdm = np.load(rdm_path, mmap_mode="r")
        model_vec = rdm_vec(model_rdm)
        scores: dict[str, float] = {}
        for region, neural_rdm in neural_rdms.items():
            if neural_rdm.shape != model_rdm.shape:
                shape_mismatches.setdefault(layer_name, {})[region] = {
                    "model_shape": list(model_rdm.shape),
                    "neural_shape": list(neural_rdm.shape),
                }
                continue
            rho = fast_spearman_vec(model_vec, neural_vecs[region])
            scores[region] = rho
            score_tasks.append((layer_name, region, rdm_path, float(rho)))
            current_best = best_by_region.get(region)
            if current_best is None or rho > float(current_best["rho"]):
                best_by_region[region] = {"layer": layer_name, "rho": float(rho)}
        layer_results[layer_name] = {
            "feature_dim": feature_dims[layer_name],
            "rsa": scores,
            "statistics": {},
        }
        del model_rdm, model_vec
        gc.collect()

    stat_regions = {task[1] for task in score_tasks}
    if args.max_stat_regions is not None:
        stat_regions = set(sorted(stat_regions)[: args.max_stat_regions])
    stat_tasks = [task for task in score_tasks if task[1] in stat_regions]

    if stat_tasks:
        parallel_tasks = stats_workers > 1 and len(stat_tasks) >= stats_workers
        perm_workers = 1 if parallel_tasks else stats_workers
        log_phase(
            f"stats start n_tasks={len(stat_tasks)} stats_workers={stats_workers} "
            f"perm_workers={perm_workers} perm_threads={args.perm_thread_workers if parallel_tasks else 'n/a'} "
            f"permutation_mode={args.permutation_mode} n_permutations={args.n_permutations}",
            log_memory=log_mem,
        )

        def run_stat_task(task: tuple[str, str, Path, float]) -> tuple[str, str, dict[str, Any]]:
            layer_name, region, rdm_path, rho = task
            if args.stats_log_per_task:
                logger.info("stats task start layer=%s region=%s rho=%.4f", layer_name, region, rho)
            model_rdm = np.load(rdm_path, mmap_mode="r")
            return (
                layer_name,
                region,
                compute_region_statistics(
                    layer_name,
                    region,
                    model_rdm,
                    neural_vecs[region],
                    neural_ranks[region],
                    rho,
                    {},
                    None,
                    n_permutations=args.n_permutations,
                    n_bootstrap=args.n_bootstrap,
                    permutation_mode=args.permutation_mode,
                    stats_seed=args.stats_seed,
                    perm_workers=perm_workers,
                    nested_outer_parallel=parallel_tasks,
                    perm_thread_workers=args.perm_thread_workers,
                ),
            )

        if parallel_tasks:
            stat_results = Parallel(n_jobs=stats_workers, backend="loky")(
                delayed(run_stat_task)(task) for task in stat_tasks
            )
        else:
            stat_results = [run_stat_task(task) for task in stat_tasks]
        for layer_name, region, region_stats in stat_results:
            layer_results[layer_name]["statistics"][region] = region_stats
            if args.stats_log_per_task:
                p = region_stats.get("permutation", {}).get("p_greater")
                logger.info("stats task done layer=%s region=%s p_greater=%s", layer_name, region, p)

    log_phase(f"finish total_elapsed_s={time.perf_counter() - t0:.1f} output={args.output}", log_memory=log_mem)
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_clips": len(clip_ids),
        "layers": layer_results,
        "best_by_region": best_by_region,
        "shape_mismatches": shape_mismatches,
        "using_stimuli_benchmark": args.stimuli_dir is not None,
        "stimuli_dir": str(args.stimuli_dir) if args.stimuli_dir is not None else None,
        "stimuli_list": str(args.stimuli_list) if args.stimuli_list is not None else None,
        "clip_ids": clip_ids,
        "activities": activities,
        "statistics": {
            "n_permutations": int(args.n_permutations),
            "permutation_mode": args.permutation_mode,
            "n_bootstrap": int(args.n_bootstrap),
            "stats_workers": stats_workers,
            "rdm_workers": rdm_workers,
            "feature_cache_dir": str(feature_dir),
            "rdm_cache_dir": str(rdm_dir),
        },
    }
    if not best_by_region:
        n_model = len(clip_ids)
        example_region, example_rdm = next(iter(neural_rdms.items()))
        n_neural = int(example_rdm.shape[0])
        raise ValueError(
            "Layerwise RSA produced no region scores: every neural RDM shape mismatched the model RDM. "
            f"Model used n={n_model} stimuli (RDM {n_model}x{n_model}); neural '{example_region}' is "
            f"{n_neural}x{n_neural}. "
            "Use --stimuli-list to ensure model and neural shapes match."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", args.output)
    print(json.dumps(best_by_region, indent=2))
    return payload


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, log_memory=args.log_memory)
    run_layerwise_analysis(args)


if __name__ == "__main__":
    main()

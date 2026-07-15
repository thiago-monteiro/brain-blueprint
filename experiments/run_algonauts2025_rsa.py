from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

H5_KEY_PATTERN = re.compile(r"ses-\d+_task-(?P<movie>[a-z]+)(?P<part>\d+)")
TR_DURATION = 1.49
N_FRAMES_PER_TR = 16
IMAGE_SIZE = 224

SCHAEFFER_7NETWORKS = [
    "Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
    "Limbic", "Frontoparietal", "DefaultMode",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("algonauts_rsa")


import decord
import torch.nn.functional as F
from egomuscle.data.dataset import IMAGE_MEAN, IMAGE_STD

_PRELOAD_BATCH = 16


def _preload_mkv_frames(mkv_path: str, needed_indices: list[int], image_size: int) -> torch.Tensor:
    """Pre-load specific frame indices from an MKV, resize, return uint8 tensor (N, 3, H, W)."""
    vr = decord.VideoReader(mkv_path)
    all_idx = list(needed_indices)
    n_total = len(all_idx)
    result = torch.empty(n_total, 3, image_size, image_size, dtype=torch.uint8)
    try:
        for start in range(0, n_total, _PRELOAD_BATCH):
            end = min(start + _PRELOAD_BATCH, n_total)
            chunk = all_idx[start:end]
            raw = vr.get_batch(chunk).asnumpy()
            t = torch.from_numpy(raw).permute(0, 3, 1, 2)
            if t.shape[-1] != image_size or t.shape[-2] != image_size:
                t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
            result[start:end] = t.to(torch.uint8)
        return result
    finally:
        del vr


class AlgonautsVideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        stimuli_root: Path,
        image_size: int = IMAGE_SIZE,
        n_frames: int = N_FRAMES_PER_TR,
    ) -> None:
        self.image_size = image_size
        self.n_frames = n_frames

        entries_raw: list[dict[str, Any]] = []
        with manifest_path.open("r") as handle:
            for line in handle:
                if line.strip():
                    entries_raw.append(json.loads(line))

        if not entries_raw:
            raise ValueError("Empty manifest")

        mkv_path = entries_raw[0]["mkv_path"]
        all_needed = sorted(set(i for e in entries_raw for i in e["frame_indices"]))

        logger.info("Pre-loading %d frames from %s ...", len(all_needed), Path(mkv_path).name)
        self._frames = _preload_mkv_frames(mkv_path, all_needed, image_size)
        idx_to_local = {g: l for l, g in enumerate(all_needed)}

        self.entries: list[dict[str, Any]] = []
        for e in entries_raw:
            local_idx = [idx_to_local[fi] for fi in e["frame_indices"]]
            self.entries.append({
                "h5_key": e["h5_key"],
                "task_name": e.get("task_name", e["h5_key"]),
                "tr_index": e["tr_index"],
                "movie": e["movie"],
                "part": e["part"],
                "local_indices": local_idx,
            })

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        frames = self._frames[entry["local_indices"]].float() / 255.0
        frames = (frames - IMAGE_MEAN) / IMAGE_STD
        return {
            "frames": frames.contiguous(),
            "h5_key": entry["h5_key"],
            "task_name": entry["task_name"],
            "tr_index": entry["tr_index"],
            "movie": entry["movie"],
            "part": entry["part"],
        }


class _PreloadedDataset(Dataset):
    def __init__(self, frames: torch.Tensor, entries: list[dict[str, Any]]) -> None:
        self._frames = frames
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        e = self._entries[idx]
        f = self._frames[e["local_indices"]].float() / 255.0
        f = (f - IMAGE_MEAN) / IMAGE_STD
        return {"frames": f.contiguous(), "task_name": e["task_name"]}


def compute_checkpoint_hash(checkpoint_path: Path) -> str:
    return hashlib.sha256(str(checkpoint_path.resolve()).encode()).hexdigest()[:12]


def feature_cache_path(output_dir: Path, movie: str, checkpoint_hash: str) -> Path:
    return output_dir / "features" / f"{movie}_{checkpoint_hash}.npy"


def rdm_cache_path(output_dir: Path, movie: str, checkpoint_hash: str) -> Path:
    return output_dir / "rdms" / f"{movie}_{checkpoint_hash}.npy"


def extract_features(
    checkpoint_path: Path,
    config_path: Path,
    dataset: AlgonautsVideoDataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    use_fp16: bool = False,
    use_compile: bool = False,
) -> dict[str, np.ndarray]:
    from egomuscle.training.train import EgoMuscleLightningModule, apply_override, load_config

    payload = torch.load(checkpoint_path, map_location="cpu")
    hyper_params = payload.get("hyper_parameters")
    if isinstance(hyper_params, dict):
        config = hyper_params
        for override in ["model.use_video=true", "model.use_muscle=false", "training.compile=false"]:
            apply_override(config, override)
    else:
        config = load_config(config_path)
        for override in ["model.use_video=true", "model.use_muscle=false", "training.compile=false"]:
            apply_override(config, override)

    module = EgoMuscleLightningModule(config)
    state_dict = payload.get("state_dict", payload)
    stripped = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
    try:
        module.load_state_dict(stripped, strict=False)
    except RuntimeError:
        pass
    module.to(device)
    module.eval()

    model = module.model
    if use_compile:
        logger.info("Compiling model with torch.compile (may take a minute)...")
        model = torch.compile(model, mode="default")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_fp16 else torch.no_grad()

    movie_features: dict[str, list[np.ndarray]] = {}
    with torch.no_grad(), autocast_ctx:
        for batch in tqdm(loader, desc="Feature extraction"):
            frames = batch["frames"].to(device, non_blocking=use_fp16)
            if use_fp16 and frames.dtype != torch.float16:
                frames = frames.half()
            batch_size_actual = frames.shape[0]
            dummy_ids = torch.zeros(batch_size_actual, dtype=torch.long, device=device)
            outputs = model(frames=frames, muscle=None, activity_ids=dummy_ids, mask_ratio=0.0)
            pooled = outputs.pooled.detach().float().cpu().numpy()
            for i in range(batch_size_actual):
                movie_key = batch["task_name"][i]
                feat = pooled[i:i+1]
                movie_features.setdefault(movie_key, []).append(feat)

    return {k: np.concatenate(v, axis=0) for k, v in movie_features.items()}


def compute_rdm(features: np.ndarray) -> np.ndarray:
    return squareform(pdist(features, metric="cosine")).astype(np.float32)


def _perm_null(seed: int, n: int, model_vec: np.ndarray, neural_vec: np.ndarray) -> float:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return float(spearmanr(model_vec, neural_vec[perm]).statistic)


def _bootstrap_single(seed: int, n: int, model_vec: np.ndarray, neural_vec: np.ndarray) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n)
    return float(spearmanr(model_vec[idx], neural_vec[idx]).statistic)


def score_rdm(
    model_rdm: np.ndarray,
    neural_rdm: np.ndarray,
    n_permutations: int = 1000,
    n_bootstrap: int = 500,
    seed: int = 0,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, Any]:
    model_vec = squareform(model_rdm, checks=False)
    neural_vec = squareform(neural_rdm, checks=False)
    rho, p_value = spearmanr(model_vec, neural_vec)
    rho = float(0.0 if np.isnan(rho) else rho)
    p_value = float(0.0 if np.isnan(p_value) else p_value)

    n = len(model_vec)
    seeds = list(range(seed, seed + n_permutations))
    null = np.array(Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_perm_null)(s, n, model_vec, neural_vec)
        for s in tqdm(seeds, desc="Permutation test", leave=False, disable=not verbose)
    ), dtype=np.float32)
    p_perm = (float(np.sum(null >= rho)) + 1.0) / (float(n_permutations) + 1.0)

    boot_seeds = list(range(seed + n_permutations, seed + n_permutations + n_bootstrap))
    bootstrap_scores = np.array(Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_bootstrap_single)(s, n, model_vec, neural_vec)
        for s in tqdm(boot_seeds, desc="Bootstrap", leave=False, disable=not verbose)
    ), dtype=np.float32)
    ci_low = float(np.quantile(bootstrap_scores, 0.025))
    ci_high = float(np.quantile(bootstrap_scores, 0.975))

    return {
        "rho": rho,
        "p_spearman": p_value,
        "p_permutation": p_perm,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
    }


def compute_noise_ceiling(
    neural_rdms: list[np.ndarray],
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    if len(neural_rdms) < 2:
        return {"nc_lower": float("nan"), "nc_upper": float("nan"), "n_subjects": len(neural_rdms)}

    n = neural_rdms[0].shape[0]
    vec_size = n * (n - 1) // 2
    n_subj = len(neural_rdms)
    subject_vecs = np.zeros((n_subj, vec_size), dtype=np.float32)
    for i, rdm in enumerate(neural_rdms):
        subject_vecs[i] = squareform(rdm, checks=False)

    loo_corrs = []
    for i in range(n_subj):
        loo_mean = np.mean(np.delete(subject_vecs, i, axis=0), axis=0)
        r = float(spearmanr(subject_vecs[i], loo_mean).statistic)
        if not np.isnan(r):
            loo_corrs.append(r)
    nc_lower = float(np.mean(loo_corrs)) if loo_corrs else float("nan")

    rng = np.random.default_rng(seed)
    split_corrs = []
    for _ in range(n_bootstrap):
        perm = rng.permutation(n_subj)
        half = n_subj // 2
        half1 = np.mean(subject_vecs[perm[:half]], axis=0)
        half2 = np.mean(subject_vecs[perm[half:]], axis=0)
        r = float(spearmanr(half1, half2).statistic)
        if not np.isnan(r):
            split_corrs.append(r)

    mean_split = float(np.mean(split_corrs)) if split_corrs else 0.0
    nc_upper = float((2 * mean_split) / (1 + mean_split)) if mean_split > 0 else float("nan")

    return {"nc_lower": nc_lower, "nc_upper": nc_upper, "n_subjects": n_subj}


def neural_rdm_path(neural_dir: Path, movie: str, network: str, subject: str) -> Path:
    return neural_dir / f"{movie}_{network}_{subject}.npy"


def neural_group_rdm_path(neural_dir: Path, movie: str, network: str) -> Path:
    return neural_dir / f"{movie}_{network}_group.npy"


def load_neural_rdms(
    neural_dir: Path,
    movies: list[str],
    networks: list[str],
    subjects: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
    result: dict[str, dict[str, Any]] = {}
    for movie in movies:
        result[movie] = {}
        for network in networks:
            if subjects:
                subject_rdms = []
                for subj in subjects:
                    p = neural_rdm_path(neural_dir, movie, network, subj)
                    if p.exists():
                        subject_rdms.append(np.load(p))
                if subject_rdms:
                    result[movie][network] = {
                        "subject_rdms": subject_rdms,
                        "group": None,
                    }
            gpath = neural_group_rdm_path(neural_dir, movie, network)
            if gpath.exists():
                if movie not in result:
                    result[movie] = {}
                if network not in result[movie]:
                    result[movie][network] = {"subject_rdms": [], "group": None}
                result[movie][network]["group"] = np.load(gpath)
                if not result[movie][network]["subject_rdms"]:
                    result[movie][network].pop("subject_rdms")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Algonauts 2025 RSA benchmark.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--h5-path", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/results/algonauts2025_clip_manifest.jsonl"))
    parser.add_argument("--stimuli-root", type=Path, default=Path("data/raw/algonauts2025/stimuli/movies/movie10"))
    parser.add_argument("--neural-dir", type=Path, default=Path("egomuscle/eval/algonauts2025_rdms"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/algonauts2025_rsa"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-workers", type=int, default=0,
                        help="Pre-load threads (0 = auto: max(2, cpu/4))")
    parser.add_argument("--no-frame-cache", action="store_true", help="Skip persistent frame cache on disk")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--features-only", action="store_true", help="Only run GPU feature extraction")
    parser.add_argument("--use-cached-features", action="store_true", help="Skip GPU, use cached features")
    parser.add_argument("--rdms-only", action="store_true", help="Only compute RDMs from cached features")
    parser.add_argument("--use-cached-rdms", action="store_true", help="Use cached RDMs, skip to scoring")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU entirely")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 autocast for feature extraction")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on the model")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_model_for_features(
    checkpoint_path: Path, config_path: Path, device: torch.device,
    use_compile: bool = False,
) -> torch.nn.Module:
    from egomuscle.training.train import EgoMuscleLightningModule, apply_override, load_config

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("hyper_parameters")
    if isinstance(config, dict):
        for o in ["model.use_video=true", "model.use_muscle=false", "training.compile=false"]:
            apply_override(config, o)
    else:
        config = load_config(config_path)
        for o in ["model.use_video=true", "model.use_muscle=false", "training.compile=false"]:
            apply_override(config, o)

    module = EgoMuscleLightningModule(config)
    state_dict = payload.get("state_dict", payload)
    stripped = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
    try:
        module.load_state_dict(stripped, strict=False)
    except RuntimeError:
        pass
    module.to(device).eval()

    model = module.model
    if use_compile:
        logger.info("Compiling model with torch.compile (may take a minute)...")
        model = torch.compile(model, mode="default")
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "features").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rdms").mkdir(parents=True, exist_ok=True)

    chash = compute_checkpoint_hash(args.checkpoint)
    logger.info("Checkpoint hash: %s", chash)

    manifest_exists = args.manifest.exists()
    if not manifest_exists and not args.use_cached_features and not args.cpu_only:
        logger.info("Building clip manifest from h5...")
        if args.h5_path is None:
            h5_files = sorted(
                Path("data/raw/algonauts2025/fmri").rglob("*_task-movie10_*_bold.h5")
            )
            if not h5_files:
                raise FileNotFoundError("No movie10 h5 files found. Set --h5-path")
            args.h5_path = h5_files[0]
        from experiments.build_algonauts2025_clips import build_movie_mapping, build_manifest_for_movie
        mapping = build_movie_mapping(args.h5_path)
        manifest = build_manifest_for_movie(args.stimuli_root, mapping)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w") as handle:
            for entry in manifest:
                handle.write(json.dumps(entry) + "\n")
        logger.info("Built manifest with %d entries", len(manifest))

    if not args.cpu_only and not args.use_cached_features:
        logger.info("=== Phase 1: Feature extraction (GPU) ===")
        logger.info("Options: batch_size=%d, num_workers=%d, fp16=%s, compile=%s",
                     args.batch_size, args.num_workers, args.fp16, args.compile)
        if args.compile and not args.fp16:
            logger.warning("--compile enabled without --fp16; compile is most effective with fp16")

        segments: dict[str, list[dict]] = defaultdict(list)
        with open(args.manifest) as f:
            for line in f:
                e = json.loads(line)
                segments[e["task_name"]].append(e)

        segment_names = sorted(segments.keys())
        logger.info("Processing %d segments one at a time", len(segment_names))

        model = _load_model_for_features(args.checkpoint, args.config, device, args.compile)
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if args.fp16 else torch.no_grad()

        import queue as _queue, threading as _threading

        _prefetch_queue: _queue.Queue = _queue.Queue(maxsize=2)
        _idx_lock = _threading.Lock()
        _next_seg = 0
        _ALL_DONE = object()

        _frame_cache_dir = args.output_dir / "frame_cache"
        _frame_cache_dir.mkdir(parents=True, exist_ok=True)

        def _preloader() -> None:
            nonlocal _next_seg
            while True:
                with _idx_lock:
                    if _next_seg >= len(segment_names):
                        break
                    idx = _next_seg
                    _next_seg += 1

                task_name = segment_names[idx]
                cpath = feature_cache_path(args.output_dir, task_name, chash)
                if cpath.exists():
                    _prefetch_queue.put(task_name)
                    continue

                entries = segments[task_name]
                mkv_path = entries[0]["mkv_path"]
                all_needed = sorted(set(i for e in entries for i in e["frame_indices"]))
                fcache_path = _frame_cache_dir / f"{task_name}.npy"

                if not args.no_frame_cache and fcache_path.exists():
                    logger.info("Frame cache exists for %s, skipping decode", task_name)
                else:
                    frames = _preload_mkv_frames(mkv_path, all_needed, IMAGE_SIZE)
                    if not args.no_frame_cache:
                        np.save(fcache_path, frames.numpy())
                        mb = frames.numel() * frames.element_size() // 1048576
                        logger.info("Saved frame cache %s (%d MB)", task_name, mb)
                    del frames
                    gc.collect()

                _prefetch_queue.put(task_name)

            _prefetch_queue.put(_ALL_DONE)

        n_preload = args.prefetch_workers if args.prefetch_workers > 0 else 2
        logger.info("Pre-loading with %d worker threads", n_preload)
        threads = [_threading.Thread(target=_preloader, daemon=True) for _ in range(n_preload)]
        for t in threads:
            t.start()

        _done_count = 0
        pbar = tqdm(total=len(segment_names), desc="Segments")
        while _done_count < len(segment_names):
            task_name = _prefetch_queue.get()
            if task_name is _ALL_DONE:
                continue

            cpath = feature_cache_path(args.output_dir, task_name, chash)
            if cpath.exists():
                _done_count += 1
                pbar.update(1)
                continue

            pbar.set_description(f"GPU: {task_name}")

            fcache_path = _frame_cache_dir / f"{task_name}.npy"
            if fcache_path.exists():
                frames_tensor = torch.from_numpy(np.load(fcache_path))
                try:
                    _fd = os.open(fcache_path, os.O_RDONLY)
                    os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(_fd)
                except Exception:
                    pass
            else:
                entries = segments[task_name]
                mkv_path = entries[0]["mkv_path"]
                all_needed = sorted(set(i for e in entries for i in e["frame_indices"]))
                frames_tensor = _preload_mkv_frames(mkv_path, all_needed, IMAGE_SIZE)

            entries = segments[task_name]
            all_needed = sorted(set(i for e in entries for i in e["frame_indices"]))
            idx_map = {g: l for l, g in enumerate(all_needed)}
            local_entries = [{
                "task_name": e.get("task_name", e["h5_key"]),
                "local_indices": [idx_map[fi] for fi in e["frame_indices"]],
            } for e in entries]

            loader = DataLoader(
                _PreloadedDataset(frames_tensor, local_entries),
                batch_size=args.batch_size, shuffle=False, num_workers=0,
            )

            feats_list: list[torch.Tensor] = []
            with torch.no_grad(), autocast_ctx:
                for batch in loader:
                    f = batch["frames"].to(device)
                    if args.fp16:
                        f = f.half()
                    bs = f.shape[0]
                    out = model(frames=f, muscle=None, activity_ids=torch.zeros(bs, dtype=torch.long, device=device), mask_ratio=0.0)
                    feats_list.append(out.pooled.detach().float().cpu())

            feats = torch.cat(feats_list, dim=0).numpy()
            np.save(cpath, feats)
            _done_count += 1
            logger.info("Saved %s (%d/%d)", task_name, _done_count, len(segment_names))
            pbar.update(1)

            del frames_tensor, loader, feats_list, feats, local_entries
            gc.collect()
        pbar.close()
        for t in threads:
            t.join(timeout=5)
        if args.features_only:
            logger.info("Features-only mode. Exiting.")
            return

    movie_features: dict[str, np.ndarray] = {}
    pattern = str(args.output_dir / "features" / f"*_{chash}.npy")
    import glob as glob_mod
    for cpath_str in sorted(glob_mod.glob(pattern)):
        cpath = Path(cpath_str)
        movie = cpath.stem.replace(f"_{chash}", "")
        movie_features[movie] = np.load(cpath)
    if not movie_features:
        raise FileNotFoundError(f"No cached features found matching {pattern}")
    logger.info("Loaded %d feature sets", len(movie_features))

    if args.use_cached_rdms:
        logger.info("Skipping RDM computation (--use-cached-rdms)")
    else:
        need_rdm_compute = any(
            not rdm_cache_path(args.output_dir, movie, chash).exists()
            for movie in movie_features
        )
        if need_rdm_compute:
            logger.info("=== Phase 2: RDM computation (CPU) ===")

            def compute_and_cache(movie: str, feats: np.ndarray) -> str:
                rdm = compute_rdm(feats)
                rpath = rdm_cache_path(args.output_dir, movie, chash)
                np.save(rpath, rdm)
                return movie

            Parallel(n_jobs=args.n_jobs, backend="threading")(
                delayed(compute_and_cache)(movie, feats)
                for movie, feats in tqdm(movie_features.items(), desc="RDMs")
            )
            logger.info("RDMs cached for %d movies", len(movie_features))
        else:
            logger.info("All RDMs already cached")

    logger.info("=== Phase 3: Scoring (CPU) ===")
    rdm_pattern = str(args.output_dir / "rdms" / f"*_{chash}.npy")
    import glob as glob_mod2
    movie_rdms: dict[str, np.ndarray] = {}
    for rpath_str in sorted(glob_mod2.glob(rdm_pattern)):
        rpath = Path(rpath_str)
        movie = rpath.stem.replace(f"_{chash}", "")
        movie_rdms[movie] = np.load(rpath)

    fmri_root = Path("data/raw/algonauts2025/fmri")
    subjects = sorted({
        p.parent.parent.name
        for p in fmri_root.glob("sub-*/func/*_task-movie10_*_bold.h5")
    }) if fmri_root.exists() else []

    neural_data = load_neural_rdms(args.neural_dir, list(movie_rdms.keys()), SCHAEFFER_7NETWORKS, subjects)

    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_hash": chash,
        "config": str(args.config),
        "movies": {},
        "best_by_network": {},
        "cross_movie_average": {},
    }

    scoring_tasks: list[tuple[str, str, np.ndarray, dict]] = []
    for movie in sorted(movie_rdms.keys()):
        model_rdm = movie_rdms[movie]
        for network in SCHAEFFER_7NETWORKS:
            neural_info = neural_data.get(movie, {}).get(network)
            if neural_info is None:
                continue
            neural_group = neural_info.get("group") if isinstance(neural_info, dict) else neural_info
            if neural_group is None:
                continue
            scoring_tasks.append((movie, network, model_rdm, neural_info))

    _n_perm = args.n_permutations
    _n_boot = args.n_bootstrap
    _seed = args.seed

    def _score_one(
        movie: str, network: str,
        model_rdm: np.ndarray,
        neural_info: dict,
    ) -> tuple[str, str, dict[str, Any] | None]:
        neural_group = neural_info.get("group") if isinstance(neural_info, dict) else neural_info
        if neural_group is None:
            return (movie, network, None)
        score = score_rdm(
            model_rdm, neural_group,
            n_permutations=_n_perm, n_bootstrap=_n_boot,
            seed=_seed, n_jobs=1, verbose=False,
        )
        subject_rdms = neural_info.get("subject_rdms", []) if isinstance(neural_info, dict) else []
        if subject_rdms:
            nc = compute_noise_ceiling(subject_rdms, n_bootstrap=_n_boot, seed=_seed)
            nc_lower, nc_upper = nc["nc_lower"], nc["nc_upper"]
            nc_norm = float(min(score["rho"] / nc_upper, 1.0)) if (nc_upper and nc_upper > 0 and score["rho"] is not None) else float("nan")
        else:
            nc_lower = nc_upper = nc_norm = float("nan")
        score.update({"nc_lower": nc_lower, "nc_upper": nc_upper, "nc_normalized": nc_norm})
        return (movie, network, score)

    logger.info("Scoring %d movie-network pairs with loky (n_jobs=%s)", len(scoring_tasks), args.n_jobs)
    scored = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_score_one)(movie, network, model_rdm, neural_info)
        for movie, network, model_rdm, neural_info in tqdm(scoring_tasks, desc="Scoring")
    )
    for movie, network, score in scored:
        if score is not None:
            results["movies"].setdefault(movie, {})[network] = score

    for network in SCHAEFFER_7NETWORKS:
        network_scores = [(m, r["rho"]) for m, mr in results["movies"].items() if network in mr for r in [mr[network]]]
        if network_scores:
            best = max(network_scores, key=lambda x: x[1])
            results["best_by_network"][network] = {"movie": best[0], "rho": best[1]}
            scores = [s[1] for s in network_scores]
            results["cross_movie_average"][network] = {
                "mean_rho": float(np.mean(scores)),
                "std_rho": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "n_movies": len(scores),
            }

    output_path = args.output_dir / f"checkpoint_{chash}.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()

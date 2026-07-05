from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from decord import VideoReader, cpu


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MUSCLE_EXTENSIONS = {".npy", ".npz"}

THREAT_KEYWORDS = {
    "avoid",
    "collapse",
    "dodge",
    "duck",
    "fall",
    "fight",
    "hit",
    "jump",
    "kick",
    "land",
    "landing",
    "leap",
    "punch",
    "protect",
    "slip",
    "startle",
    "stumble",
    "threat",
    "trip",
    "withdraw",
}


@dataclass
class EgoMuscleSample:
    video_path: Path
    muscle_path: Path | None = None
    activity: str | None = None
    activity_id: int | None = None
    metadata: dict[str, Any] | None = None
    frame_count: int | None = None


def _load_metadata_table(metadata_path: Path | None) -> dict[str, dict[str, Any]]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    if metadata_path.suffix.lower() == ".json":
        payload = json.loads(metadata_path.read_text())
        if isinstance(payload, list):
            return {str(item["clip_id"]): item for item in payload}
        return {str(k): v for k, v in payload.items()}
    if metadata_path.suffix.lower() == ".csv":
        with metadata_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            return {str(row["clip_id"]): row for row in reader}
    raise ValueError(f"Unsupported metadata file: {metadata_path}")


def _infer_activity(stem: str, metadata: dict[str, Any] | None) -> str | None:
    if metadata and metadata.get("activity"):
        return str(metadata["activity"])
    if "__" in stem:
        return stem.split("__", maxsplit=1)[0]
    return None


def _flatten_metadata_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(_flatten_metadata_strings(child))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for child in value:
            out.extend(_flatten_metadata_strings(child))
        return out
    return []


def metadata_has_threat(metadata: dict[str, Any] | None, activity: str | None = None) -> bool:
    labels = _flatten_metadata_strings(metadata or {})
    if activity:
        labels.append(activity)
    text = " ".join(labels).lower().replace("_", " ")
    tokens = set(re.findall(r"[a-z]+", text))
    return any(keyword in tokens or keyword in text for keyword in THREAT_KEYWORDS)


def add_threat_signature(
    muscle: torch.Tensor,
    *,
    strength: float,
    startle_width: float,
    withdrawal_bias: float,
    autonomic_ramp: float,
) -> torch.Tensor:
    if strength <= 0.0 or muscle.numel() == 0:
        return muscle
    seq_len, dim = muscle.shape
    device = muscle.device
    dtype = muscle.dtype
    time = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype).unsqueeze(-1)
    width = max(float(startle_width), 1.0 / max(seq_len, 1))
    startle = torch.exp(-0.5 * torch.square((time - 0.18) / width))
    withdrawal = torch.clamp((time - 0.35) / 0.65, min=0.0, max=1.0)
    autonomic = torch.sin(math.pi * torch.clamp(time, 0.0, 1.0)).square()

    channels = torch.arange(dim, device=device)
    flexor_mask = ((channels % 4) < 2).to(dtype).unsqueeze(0)
    extensor_mask = 1.0 - flexor_mask
    signature = startle
    signature = signature + float(withdrawal_bias) * withdrawal * flexor_mask
    signature = signature - 0.5 * float(withdrawal_bias) * withdrawal * extensor_mask
    signature = signature + float(autonomic_ramp) * autonomic

    scale = muscle.detach().std().clamp_min(torch.tensor(1.0e-4, dtype=dtype, device=device))
    return muscle + float(strength) * scale * signature


def discover_records(
    video_dir: str | Path,
    muscle_dir: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> list[EgoMuscleSample]:
    video_root = Path(video_dir)
    muscle_root = Path(muscle_dir) if muscle_dir is not None else None
    metadata_table = _load_metadata_table(Path(metadata_path) if metadata_path else None)

    muscle_files: dict[str, Path] = {}
    if muscle_root is not None and muscle_root.exists():
        for path in muscle_root.rglob("*"):
            if path.suffix.lower() in MUSCLE_EXTENSIONS:
                muscle_files[path.stem] = path

    records: list[EgoMuscleSample] = []
    for video_path in sorted(video_root.rglob("*")):
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        metadata = metadata_table.get(video_path.stem, {})
        activity = _infer_activity(video_path.stem, metadata)
        activity_id = metadata.get("activity_id")
        if activity_id is not None:
            activity_id = int(activity_id)

        frame_count = None
        quality_path = Path(metadata.get("quality_path")) if metadata.get("quality_path") else None
        if quality_path and quality_path.exists():
            with quality_path.open() as f:
                q = json.load(f)
                frame_count = q.get("frame_count")

        records.append(
            EgoMuscleSample(
                video_path=video_path,
                muscle_path=muscle_files.get(video_path.stem),
                activity=activity,
                activity_id=activity_id,
                metadata=metadata,
                frame_count=frame_count,
            )
        )
    return records


def _sample_indices(num_frames: int, n_frames: int, mode: str = "sparse_uniform") -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("Video contains no frames.")

    if mode == "sparse_uniform":
        if num_frames >= n_frames:
            return np.linspace(0, num_frames - 1, num=n_frames, dtype=np.int64)
        return np.linspace(0, num_frames - 1, num=n_frames, dtype=np.float32).round().astype(np.int64)

    if mode == "random_window":
        if num_frames <= n_frames:
            return _sample_indices(num_frames, n_frames, mode="sparse_uniform")
        start = np.random.randint(0, num_frames - n_frames + 1)
        return np.arange(start, start + n_frames)

    if mode == "random_stride":
        if num_frames <= n_frames:
            return _sample_indices(num_frames, n_frames, mode="sparse_uniform")
        max_stride = max(1, num_frames // n_frames)
        stride = np.random.randint(1, max_stride + 1)
        max_start = num_frames - (n_frames - 1) * stride - 1
        start = np.random.randint(0, max_start + 1)
        return np.arange(start, start + n_frames * stride, stride)[:n_frames]

    raise ValueError(f"Unknown temporal_sample_mode: {mode}")


def _resize_and_normalize_frames(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frame tensor with shape (T, C, H, W), got {tuple(frames.shape)}")
    frames = frames.float() / 255.0
    frames = F.interpolate(frames, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return (frames - IMAGE_MEAN) / IMAGE_STD


def _resize_frames_uint8(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frame tensor with shape (T, C, H, W), got {tuple(frames.shape)}")
    frames = F.interpolate(frames.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
    return frames.round().clamp_(0, 255).to(torch.uint8)


def _read_video_decord(reader: Any, indices: np.ndarray) -> torch.Tensor:
    batch = reader.get_batch(indices).asnumpy()
    return torch.from_numpy(batch).permute(0, 3, 1, 2)


def _open_video_reader(path: Path) -> Any:
    try:
        return VideoReader(str(path), ctx=cpu(0))
    except Exception as exc:
        raise RuntimeError(f"Failed to open video with decord: {path}") from exc


def load_video_frames(
    video_path: str | Path,
    n_frames: int = 16,
    image_size: int = 224,
    indices: np.ndarray | None = None,
) -> torch.Tensor:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    reader = _open_video_reader(path)
    if indices is None:
        indices = _sample_indices(len(reader), n_frames)
    frames = _read_video_decord(reader, indices)
    return _resize_and_normalize_frames(frames, image_size=image_size)


def load_video_frames_uint8(
    video_path: str | Path,
    n_frames: int = 16,
    image_size: int = 224,
    indices: np.ndarray | None = None,
) -> torch.Tensor:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    reader = _open_video_reader(path)
    if indices is None:
        indices = _sample_indices(len(reader), n_frames)
    frames = _read_video_decord(reader, indices)
    return _resize_frames_uint8(frames, image_size=image_size)


def get_video_frame_count(video_path: str | Path) -> int:
    path = Path(video_path)
    return len(_open_video_reader(path))


@contextmanager
def _frame_cache_lock(cache_path: Path):
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def _load_frame_cache(cache_path: Path) -> np.ndarray:
    try:
        return np.load(cache_path)
    except ValueError as exc:
        message = str(exc).lower()
        if "failed to read" in message or "not fully written" in message:
            cache_path.unlink(missing_ok=True)
            raise
        raise


def _save_frame_cache_atomic(cache_path: Path, array: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_muscle_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    payload = np.load(path)
    if "muscle" in payload:
        return payload["muscle"]
    first_key = next(iter(payload.files))
    return payload[first_key]


def _resize_sequence(sequence: torch.Tensor, target_len: int) -> torch.Tensor:
    if sequence.shape[0] == target_len:
        return sequence
    transposed = sequence.transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(transposed, size=target_len, mode="linear", align_corners=False)
    return resized.squeeze(0).transpose(0, 1)


class EgoMuscleDataset(Dataset):
    def __init__(
        self,
        clip_dir: str | Path,
        muscle_dir: str | Path | None = None,
        metadata_path: str | Path | None = None,
        n_frames: int = 16,
        image_size: int = 224,
        muscle_dim: int | None = None,
        require_muscle: bool = True,
        scramble_video: bool = False,
        temporal_sample_mode: str = "sparse_uniform",
        muscle_time_offset: int = 0,
        muscle_noise_std: float = 0.0,
        frame_cache_dir: str | Path | None = None,
        write_frame_cache: bool = True,
        full_cache_dir: str | Path | None = None,
        video_feature_cache_dir: str | Path | None = None,
        is_train: bool = False,
        replacement_sampling: bool = False,
        virtual_size: int | None = None,
        threat_correlation_fraction: float = 0.0,
        threat_signature_strength: float = 0.0,
        threat_startle_width: float = 0.08,
        threat_withdrawal_bias: float = 0.75,
        threat_autonomic_ramp: float = 0.25,
        threat_seed: int = 0,
    ) -> None:
        self.records = discover_records(clip_dir, muscle_dir, metadata_path)
        self.natural_size = len(self.records)
        self.n_frames = n_frames
        self.image_size = image_size
        self.muscle_dim = muscle_dim
        self.require_muscle = require_muscle
        self.scramble_video = scramble_video
        self.temporal_sample_mode = temporal_sample_mode
        self.muscle_time_offset = int(muscle_time_offset)
        self.muscle_noise_std = float(muscle_noise_std)
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir is not None else None
        self.full_cache_dir = Path(full_cache_dir) if full_cache_dir is not None else None
        self.video_feature_cache_dir = Path(video_feature_cache_dir) if video_feature_cache_dir is not None else None
        self.write_frame_cache = write_frame_cache
        self.is_train = is_train
        self.replacement_sampling = bool(replacement_sampling)
        self.virtual_size = int(virtual_size) if virtual_size is not None else None
        self.threat_correlation_fraction = float(threat_correlation_fraction)
        self.threat_signature_strength = float(threat_signature_strength)
        self.threat_startle_width = float(threat_startle_width)
        self.threat_withdrawal_bias = float(threat_withdrawal_bias)
        self.threat_autonomic_ramp = float(threat_autonomic_ramp)
        self.threat_seed = int(threat_seed)
        if not self.records:
            raise ValueError(f"No video records found under {clip_dir}")
        if self.virtual_size is not None and self.virtual_size <= 0:
            raise ValueError("virtual_size must be positive when provided.")
        if not 0.0 <= self.threat_correlation_fraction <= 1.0:
            raise ValueError("threat_correlation_fraction must be in [0, 1].")
        if self.video_feature_cache_dir is not None:
            print(f"Using video feature cache: {self.video_feature_cache_dir}")
        else:
            print(f"Dataset initialized with {len(self.records)} samples. Probing metadata on 12 threads...")
        frame_count_cache_path = self.frame_cache_dir / "frame_counts.json" if self.frame_cache_dir is not None else None
        frame_count_cache: dict[str, int] = {}
        if frame_count_cache_path is not None and frame_count_cache_path.exists():
            try:
                frame_count_cache = {str(k): int(v) for k, v in json.loads(frame_count_cache_path.read_text()).items()}
            except (json.JSONDecodeError, ValueError, TypeError):
                frame_count_cache = {}
        for record in self.records:
            cached_count = frame_count_cache.get(record.video_path.stem)
            if cached_count is not None:
                record.frame_count = cached_count

        def probe_one(r: EgoMuscleSample) -> None:
            if r.frame_count is None:
                r.frame_count = get_video_frame_count(r.video_path)
        
        with ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(probe_one, self.records)
        if frame_count_cache_path is not None:
            frame_count_cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame_count_cache_path.write_text(
                json.dumps({record.video_path.stem: int(record.frame_count or 0) for record in self.records}, indent=2)
            )

    def _load_frames(self, record: EgoMuscleSample, indices: np.ndarray | None = None) -> torch.Tensor:
        if self.full_cache_dir is not None:
            full_path = self.full_cache_dir / f"{record.video_path.stem}.npy"
            if not full_path.exists():
                raise FileNotFoundError(f"Missing full cache for {record.video_path.stem}: {full_path}")
            frames_all = np.load(full_path, mmap_mode="r")
            frames = frames_all if indices is None else frames_all[indices]
            tensor = torch.from_numpy(np.array(frames))
            if tensor.ndim == 4 and tensor.shape[1] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            return tensor

        if self.frame_cache_dir is None:
            return load_video_frames(record.video_path, n_frames=self.n_frames, image_size=self.image_size, indices=indices)

        cache_path = self.frame_cache_dir / f"{record.video_path.stem}.npy"
        if cache_path.exists():
            try:
                frames = torch.from_numpy(_load_frame_cache(cache_path))
            except ValueError:
                frames = None
            if frames is not None:
                if indices is not None:
                    if int(indices.max()) >= frames.shape[0]:
                        return load_video_frames_uint8(
                            record.video_path,
                            n_frames=self.n_frames,
                            image_size=self.image_size,
                            indices=indices,
                        )
                    return frames[indices]
                return frames

        if self.write_frame_cache:
            with _frame_cache_lock(cache_path):
                if cache_path.exists():
                    try:
                        frames = torch.from_numpy(_load_frame_cache(cache_path))
                    except ValueError:
                        frames = None
                    if frames is not None:
                        if indices is not None:
                            if int(indices.max()) >= frames.shape[0]:
                                return load_video_frames_uint8(
                                    record.video_path,
                                    n_frames=self.n_frames,
                                    image_size=self.image_size,
                                    indices=indices,
                                )
                            return frames[indices]
                        return frames
                frames = load_video_frames_uint8(
                    record.video_path,
                    n_frames=self.n_frames,
                    image_size=self.image_size,
                    indices=indices,
                )
                _save_frame_cache_atomic(cache_path, frames.numpy())
                return frames

        return load_video_frames_uint8(
            record.video_path,
            n_frames=self.n_frames,
            image_size=self.image_size,
            indices=indices,
        )

    def __len__(self) -> int:
        if self.is_train and self.virtual_size is not None:
            return self.virtual_size
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.is_train and self.replacement_sampling:
            idx = int(torch.randint(0, self.natural_size, (1,)).item())
        elif self.is_train and self.virtual_size is not None:
            idx = idx % self.natural_size
        record = self.records[idx]
        total_frames = record.frame_count or self.n_frames
        indices = _sample_indices(total_frames, self.n_frames, mode=self.temporal_sample_mode)

        video_features: torch.Tensor | None = None
        if self.video_feature_cache_dir is not None:
            feature_path = self.video_feature_cache_dir / f"{record.video_path.stem}.npy"
            if feature_path.exists():
                video_features = torch.from_numpy(np.load(feature_path)).float()

        if video_features is not None:
            frames = None
        else:
            frames = self._load_frames(record, indices=indices)
            if self.scramble_video:
                permutation = torch.randperm(frames.shape[0])
                frames = frames[permutation]

        muscle_tensor: torch.Tensor | None = None
        if record.muscle_path is not None:
            muscle_raw = torch.from_numpy(_load_muscle_array(record.muscle_path)).float()
            if muscle_raw.ndim == 1:
                muscle_raw = muscle_raw.unsqueeze(-1)
            m_frames = muscle_raw.shape[0]
            if m_frames != total_frames:
                m_indices = (indices * (m_frames - 1) / (total_frames - 1)).astype(np.int64)
            else:
                m_indices = indices
            if self.muscle_time_offset != 0:
                m_indices = m_indices + self.muscle_time_offset
            m_indices = np.clip(m_indices, 0, m_frames - 1)
            muscle = muscle_raw[m_indices]
            if self.muscle_dim is not None and muscle.shape[-1] != self.muscle_dim:
                raise ValueError(
                    f"Expected muscle dim {self.muscle_dim}, received {muscle.shape[-1]} from {record.muscle_path}"
                )
            if self.is_train and self.muscle_noise_std > 0:
                muscle = muscle + torch.randn_like(muscle) * self.muscle_noise_std
            if (
                self.is_train
                and self.threat_signature_strength > 0.0
                and metadata_has_threat(record.metadata, record.activity)
            ):
                rng = np.random.default_rng(self.threat_seed + idx)
                if float(rng.random()) < self.threat_correlation_fraction:
                    muscle = add_threat_signature(
                        muscle,
                        strength=self.threat_signature_strength,
                        startle_width=self.threat_startle_width,
                        withdrawal_bias=self.threat_withdrawal_bias,
                        autonomic_ramp=self.threat_autonomic_ramp,
                    )
            muscle_tensor = muscle
        elif self.require_muscle:
            raise FileNotFoundError(f"Missing muscle file for clip {record.video_path.stem}")

        return {
            "frames": frames,
            "muscle": muscle_tensor,
            "video_features": video_features,
            "activity": record.activity,
            "activity_id": -1 if record.activity_id is None else record.activity_id,
            "clip_id": record.video_path.stem,
            "metadata": record.metadata or {},
        }


def collate_egomuscle(batch: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(batch)
    frames_list = [item["frames"] for item in items]
    frames = torch.stack(frames_list, dim=0) if frames_list[0] is not None else None
    muscles = [item["muscle"] for item in items]
    muscle_tensor = None if any(m is None for m in muscles) else torch.stack(muscles, dim=0)
    video_features_list = [item["video_features"] for item in items]
    video_features = torch.stack(video_features_list, dim=0) if video_features_list[0] is not None else None
    return {
        "frames": frames,
        "muscle": muscle_tensor,
        "video_features": video_features,
        "activity": [item["activity"] for item in items],
        "activity_id": torch.tensor([item["activity_id"] for item in items], dtype=torch.long),
        "clip_id": [item["clip_id"] for item in items],
        "metadata": [item["metadata"] for item in items],
    }

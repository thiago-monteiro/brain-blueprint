from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable

CACHE_ROOT = Path("experiments/results/.cache/layerwise")


def _cache_key(*parts: object) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:20]


def load_or_build(path: Path, builder: Callable[[], Any]) -> Any:
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    value = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return value


def subject_rdms_cache_path(mat_root: Path, target_n: int) -> Path:
    return CACHE_ROOT / f"subject_rdms_{_cache_key(mat_root.resolve(), target_n)}_n{target_n}.pkl"


def neural_rank_cache_path(neural_dir: Path, region: str, n_stimuli: int) -> Path:
    return CACHE_ROOT / f"neural_ranks_{_cache_key(neural_dir.resolve(), region)}_n{n_stimuli}.npy"


def stimuli_list_cache_path(stimuli_dir: Path, stimuli_list: Path) -> Path:
    return CACHE_ROOT / f"stimuli_{_cache_key(stimuli_dir.resolve(), stimuli_list.resolve())}.pkl"


def layerwise_run_cache_dir(checkpoint: Path, output: Path, n_stimuli: int, suffix: str) -> Path:
    key = _cache_key(checkpoint.resolve(), output.resolve(), n_stimuli, suffix)
    return CACHE_ROOT / f"run_{key}_n{n_stimuli}" / suffix

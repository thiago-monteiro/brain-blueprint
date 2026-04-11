from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA


def build_contiguous_groups(input_dim: int, n_groups: int = 20) -> list[list[int]]:
    indices = np.array_split(np.arange(input_dim), n_groups)
    return [chunk.astype(int).tolist() for chunk in indices]


def load_group_map(path: str | Path) -> list[list[int]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        return [list(map(int, members)) for _, members in sorted(payload.items())]
    return [list(map(int, members)) for members in payload]


def reduce_by_groups(sequence: np.ndarray, groups: Iterable[Iterable[int]]) -> np.ndarray:
    reduced = []
    for group in groups:
        indices = list(group)
        reduced.append(sequence[:, indices].mean(axis=1, keepdims=True))
    return np.concatenate(reduced, axis=1)


def fit_pca(sequences: list[np.ndarray], n_components: int = 32) -> PCA:
    stacked = np.concatenate(sequences, axis=0)
    pca = PCA(n_components=n_components, whiten=False, random_state=0)
    pca.fit(stacked)
    return pca


def reduce_sequence(
    sequence: np.ndarray,
    method: str = "groups",
    groups: list[list[int]] | None = None,
    pca: PCA | None = None,
    n_components: int = 32,
) -> np.ndarray:
    if method == "groups":
        return reduce_by_groups(sequence, groups or build_contiguous_groups(sequence.shape[1]))
    if method == "pca":
        if pca is None:
            pca = PCA(n_components=n_components, whiten=False, random_state=0).fit(sequence)
        return pca.transform(sequence)
    raise ValueError(f"Unknown reduction method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce MinT muscle activations to a compact representation.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["groups", "pca"], default="groups")
    parser.add_argument("--group-map", type=Path)
    parser.add_argument("--components", type=int, default=32)
    args = parser.parse_args()

    input_paths = sorted(args.input_dir.glob("*.npy"))
    if not input_paths:
        raise ValueError(f"No .npy files found in {args.input_dir}")

    sequences = [np.load(path) for path in input_paths]
    if any(seq.ndim != 2 for seq in sequences):
        raise ValueError("Expected all muscle arrays to have shape (T, D)")

    pca = fit_pca(sequences, n_components=args.components) if args.method == "pca" else None
    groups = load_group_map(args.group_map) if args.group_map else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, sequence in zip(input_paths, sequences):
        reduced = reduce_sequence(sequence, method=args.method, groups=groups, pca=pca, n_components=args.components)
        np.save(args.output_dir / path.name, reduced.astype(np.float32))

    if pca is not None:
        np.save(args.output_dir / "pca_components.npy", pca.components_.astype(np.float32))
        np.save(args.output_dir / "pca_mean.npy", pca.mean_.astype(np.float32))


if __name__ == "__main__":
    main()

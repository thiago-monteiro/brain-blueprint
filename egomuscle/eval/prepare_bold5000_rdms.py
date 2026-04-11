from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform


MAT_FILE_PATTERN = re.compile(
    r"CSI(?P<subject>\d+)_GLMbetas-[^_]+_allses_(?P<hemisphere>LH|RH)(?P<roi>[A-Za-z]+)\.mat$"
)

ROI_NAME_MAP = {
    "EarlyVis": "primary_visual",
    "LOC": "inferotemporal",
    "LO": "lateral_occipital",
    "OPA": "occipital_place_area",
    "PPA": "parahippocampal_place_area",
    "RSC": "retrosplenial_cortex",
    "RRSC": "retrosplenial_cortex",
}


def standardize_features(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    feature_mean = matrix.mean(axis=0, keepdims=True)
    feature_std = matrix.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    return (matrix - feature_mean) / feature_std


def compute_rdm(matrix: np.ndarray) -> np.ndarray:
    return squareform(pdist(matrix, metric="cosine")).astype(np.float32)


def load_roi_betas(mat_path: Path) -> np.ndarray:
    payload = loadmat(mat_path)
    keys = [key for key in payload.keys() if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected a single ROI variable in {mat_path}, found {keys}")
    return payload[keys[0]]


def aggregate_roi_group(mat_files: list[Path]) -> np.ndarray:
    loaded = [(path, standardize_features(load_roi_betas(path))) for path in sorted(mat_files)]
    count_histogram: dict[int, int] = {}
    for _, matrix in loaded:
        count_histogram[matrix.shape[0]] = count_histogram.get(matrix.shape[0], 0) + 1

    target_rows = max(count_histogram.items(), key=lambda item: (item[1], item[0]))[0]
    filtered = [matrix for _, matrix in loaded if matrix.shape[0] == target_rows]
    if not filtered:
        raise ValueError(f"No matrices matched target stimulus count {target_rows}")
    return np.concatenate(filtered, axis=1)


def discover_roi_groups(mat_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(mat_root.glob("*.mat")):
        match = MAT_FILE_PATTERN.match(path.name)
        if not match:
            continue
        roi_name = match.group("roi")
        canonical = ROI_NAME_MAP.get(roi_name)
        if canonical is None:
            continue
        groups.setdefault(canonical, []).append(path)
    return groups


def write_manifest(manifest: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BOLD5000 ROI beta matrices into neural RDM .npy files.")
    parser.add_argument(
        "--mat-root",
        type=Path,
        default=Path("data/raw/bold5000/extracted/BOLD5000_GLMsingle_ROI_betas/mat"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("egomuscle/eval/neural_rdms"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/metadata/bold5000_rdm_manifest.json"))
    args = parser.parse_args()

    roi_groups = discover_roi_groups(args.mat_root)
    if not roi_groups:
        raise ValueError(f"No ROI .mat files found in {args.mat_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"mat_root": str(args.mat_root), "regions": {}}

    for canonical_name, mat_files in sorted(roi_groups.items()):
        loaded = [(path, load_roi_betas(path)) for path in sorted(mat_files)]
        row_histogram: dict[int, int] = {}
        for _, matrix in loaded:
            row_histogram[matrix.shape[0]] = row_histogram.get(matrix.shape[0], 0) + 1
        target_rows = max(row_histogram.items(), key=lambda item: (item[1], item[0]))[0]
        kept_files = [path for path, matrix in loaded if matrix.shape[0] == target_rows]
        skipped_files = [path for path, matrix in loaded if matrix.shape[0] != target_rows]

        combined = np.concatenate([standardize_features(matrix) for path, matrix in loaded if matrix.shape[0] == target_rows], axis=1)
        rdm = compute_rdm(combined)
        output_path = args.output_dir / f"{canonical_name}.npy"
        np.save(output_path, rdm)
        manifest["regions"][canonical_name] = {
            "output": str(output_path),
            "source_files": [str(path) for path in kept_files],
            "skipped_files": [str(path) for path in skipped_files],
            "stimulus_count": int(combined.shape[0]),
            "feature_count": int(combined.shape[1]),
        }
        print(f"Wrote {output_path} with shape {rdm.shape}")

    write_manifest(manifest, args.manifest)
    print(f"Wrote manifest {args.manifest}")


if __name__ == "__main__":
    main()

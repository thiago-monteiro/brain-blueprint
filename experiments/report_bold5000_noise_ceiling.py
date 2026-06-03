from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


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


def load_roi_betas(mat_path: Path) -> np.ndarray:
    payload = loadmat(mat_path)
    keys = [key for key in payload.keys() if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected a single ROI variable in {mat_path}, found {keys}")
    return payload[keys[0]]


def compute_rdm(matrix: np.ndarray) -> np.ndarray:
    return squareform(pdist(matrix, metric="cosine")).astype(np.float32)


def rdm_upper_triangle(rdm: np.ndarray) -> np.ndarray:
    return squareform(rdm, checks=False)


def discover_subject_roi_files(
    mat_root: Path,
) -> dict[str, dict[str, list[Path]]]:
    result: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(mat_root.glob("*.mat")):
        match = MAT_FILE_PATTERN.match(path.name)
        if not match:
            continue
        roi_name = match.group("roi")
        canonical = ROI_NAME_MAP.get(roi_name)
        if canonical is None:
            continue
        subject = f"CSI{match.group('subject')}"
        result.setdefault(canonical, {}).setdefault(subject, []).append(path)
    return result


def build_subject_rdm(mat_files: list[Path], target_rows: int) -> np.ndarray | None:
    matrices = []
    for path in sorted(mat_files):
        betas = load_roi_betas(path)
        if betas.shape[0] != target_rows:
            return None
        matrices.append(standardize_features(betas))
    combined = np.concatenate(matrices, axis=1)
    return compute_rdm(combined)


def noise_ceiling(
    subject_rdms: dict[str, np.ndarray],
) -> dict:
    subjects = sorted(subject_rdms)
    n = len(subjects)
    if n < 2:
        return {
            "num_subjects": n,
            "lower": None,
            "upper": None,
            "subject_rows": [],
        }

    vectors = {s: rdm_upper_triangle(subject_rdms[s]) for s in subjects}

    group_mean = np.mean([vectors[s] for s in subjects], axis=0)

    subject_rows = []
    for s in subjects:
        upper_rho = float(spearmanr(vectors[s], group_mean).statistic)

        loo_mean = np.mean([vectors[o] for o in subjects if o != s], axis=0)
        lower_rho = float(spearmanr(vectors[s], loo_mean).statistic)

        subject_rows.append({"subject": s, "lower": lower_rho, "upper": upper_rho})

    lower_bound = float(np.mean([r["lower"] for r in subject_rows]))
    upper_bound = float(np.mean([r["upper"] for r in subject_rows]))

    return {
        "num_subjects": n,
        "lower": lower_bound,
        "upper": upper_bound,
        "subject_rows": subject_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report noise ceilings for BOLD5000 ROI RDMs."
    )
    parser.add_argument(
        "--mat-root",
        type=Path,
        default=Path(
            "data/raw/bold5000/extracted/BOLD5000_GLMsingle_ROI_betas/mat"
        ),
    )
    parser.add_argument(
        "--stimuli-list",
        type=Path,
        default=Path("experiments/results/bold5000_stimuli_list_neural_order.txt"),
        help="Stimuli list (used only for its line count to verify stimulus alignment).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/bold5000_noise_ceiling"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stimuli = [
        line.strip()
        for line in args.stimuli_list.read_text().splitlines()
        if line.strip()
    ]
    print(f"Loaded stimuli list with {len(stimuli)} stimuli.")
    target_rows = len(stimuli)

    subject_roi_files = discover_subject_roi_files(args.mat_root)
    if not subject_roi_files:
        raise FileNotFoundError(f"No ROI .mat files found in {args.mat_root}")

    results: dict[str, dict] = {}
    print("Loading subject RDMs and computing noise ceilings...")

    for roi in sorted(subject_roi_files):
        per_subject = subject_roi_files[roi]

        subject_rdms: dict[str, np.ndarray] = {}
        for subject, mat_files in sorted(per_subject.items()):
            rdm = build_subject_rdm(mat_files, target_rows)
            if rdm is not None:
                subject_rdms[subject] = rdm

        result = noise_ceiling(subject_rdms)
        results[roi] = result

        lb = f"{result['lower']:.4f}" if result["lower"] is not None else "None"
        ub = f"{result['upper']:.4f}" if result["upper"] is not None else "None"
        print(f"Region: {roi}")
        print(f"  Num subjects: {result['num_subjects']}")
        print(f"  Lower bound:  {lb}")
        print(f"  Upper bound:  {ub}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "noise_ceiling.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote noise ceilings to {output_path}")


if __name__ == "__main__":
    main()
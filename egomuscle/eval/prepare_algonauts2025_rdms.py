from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

SCHAEFFER_7NETWORKS = {
    "Visual": slice(0, 130),
    "Somatomotor": slice(130, 260),
    "DorsalAttention": slice(260, 363),
    "VentralAttention": slice(363, 452),
    "Limbic": slice(452, 509),
    "Frontoparietal": slice(509, 680),
    "DefaultMode": slice(680, 1000),
}

SCHAEFFER_NETWORK_NAMES = list(SCHAEFFER_7NETWORKS.keys())


def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))


def inverse_fisher_z(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def discover_h5_files(fmri_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for subj_dir in sorted(fmri_root.glob("sub-*")):
        subject = subj_dir.name
        func_dir = subj_dir / "func"
        if not func_dir.is_dir():
            continue
        for h5_path in sorted(func_dir.glob("*.h5")):
            records.append({"subject": subject, "path": h5_path})
    return records


H5_TASK_PATTERN = re.compile(r"ses-\d+_task-(?P<task>[a-z]+\d+)(?:_run-(\d+))?")


def extract_task_name(h5_key: str) -> str:
    match = H5_TASK_PATTERN.match(h5_key)
    if match:
        task = match.group("task")
        run = match.group(2)
        return f"{task}_run{run}" if run else task
    return h5_key


def discover_movies_from_h5(h5_paths: list[Path]) -> list[str]:
    movies: set[str] = set()
    for path in h5_paths:
        try:
            with h5py.File(path, "r") as handle:
                for key in handle.keys():
                    movies.add(extract_task_name(str(key)))
        except Exception:
            pass
    return sorted(movies)


def _build_task_to_h5key_map(h5_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        with h5py.File(h5_path, "r") as handle:
            for key in handle.keys():
                name = extract_task_name(key)
                mapping[name] = key
    except Exception:
        pass
    return mapping


def compute_network_rdm(
    h5_path: Path,
    movie_name: str,
    network_name: str,
    parcel_slice: slice,
    subject: str,
    output_dir: Path,
    task_to_h5key: dict[str, str] | None = None,
) -> dict[str, object]:
    try:
        if task_to_h5key is not None and movie_name in task_to_h5key:
            h5_key = task_to_h5key[movie_name]
        else:
            with h5py.File(h5_path, "r") as handle:
                matching_keys = [k for k in handle.keys() if extract_task_name(k) == movie_name]
                if not matching_keys:
                    return {"status": "skipped", "reason": f"task-{movie_name} not found in {h5_path.name}"}
                h5_key = matching_keys[0]

        with h5py.File(h5_path, "r") as handle:
            data = handle[h5_key][()]

        if data.ndim != 2 or data.shape[1] != 1000:
            return {"status": "skipped", "reason": f"unexpected shape {data.shape}"}

        network_data = data[:, parcel_slice]
        network_repr = network_data.mean(axis=1, keepdims=False)

        rdm = squareform(pdist(network_repr.reshape(-1, 1), metric="cosine")).astype(np.float32)

        out_name = f"{movie_name}_{network_name}_{subject}.npy"
        out_path = output_dir / out_name
        np.save(out_path, rdm)

        return {
            "status": "ok",
            "movie": movie_name,
            "network": network_name,
            "subject": subject,
            "output": str(out_path),
            "n_trs": int(data.shape[0]),
        }
    except Exception as exc:
        return {"status": "failed", "movie": movie_name, "network": network_name, "subject": subject, "error": str(exc)}


def compute_group_average_rdms(
    output_dir: Path,
    movies: list[str],
    subjects: list[str],
    manifest: list[dict[str, object]],
) -> list[dict[str, object]]:
    group_records: list[dict[str, object]] = []
    for movie in movies:
        for network_name in SCHAEFFER_NETWORK_NAMES:
            subject_rdms: list[np.ndarray] = []
            subject_ids: list[str] = []
            for subject in subjects:
                pattern = f"{movie}_{network_name}_{subject}.npy"
                path = output_dir / pattern
                if path.exists():
                    rdm = np.load(path)
                    subject_rdms.append(rdm)
                    subject_ids.append(subject)
            if len(subject_rdms) < 2:
                group_records.append({
                    "movie": movie,
                    "network": network_name,
                    "status": "skipped",
                    "reason": f"only {len(subject_rdms)} subject(s) available, need >=2",
                })
                continue

            shapes = {rdm.shape for rdm in subject_rdms}
            if len(shapes) > 1:
                group_records.append({
                    "movie": movie,
                    "network": network_name,
                    "status": "skipped",
                    "reason": f"inconsistent shapes {shapes} across subjects {subject_ids}",
                })
                continue

            z_values = np.stack([fisher_z(rdm) for rdm in subject_rdms], axis=0)
            mean_z = z_values.mean(axis=0)
            group_rdm = inverse_fisher_z(mean_z).astype(np.float32)

            out_name = f"{movie}_{network_name}_group.npy"
            out_path = output_dir / out_name
            np.save(out_path, group_rdm)

            group_records.append({
                "status": "ok",
                "movie": movie,
                "network": network_name,
                "output": str(out_path),
                "n_subjects": int(len(subject_rdms)),
            })
    return group_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Algonauts 2025 neural RDMs from fMRI .h5 files.")
    parser.add_argument("--fmri-root", type=Path, default=Path("data/raw/algonauts2025/fmri"))
    parser.add_argument("--output-dir", type=Path, default=Path("egomuscle/eval/algonauts2025_rdms"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/metadata/algonauts2025_rdm_manifest.json"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_h5_files(args.fmri_root)
    if not records:
        raise FileNotFoundError(f"No .h5 files found under {args.fmri_root}")

    subjects = sorted({r["subject"] for r in records})
    h5_paths = [r["path"] for r in records]

    print(f"Found {len(records)} .h5 files across {len(subjects)} subjects: {subjects}")

    movies = discover_movies_from_h5(h5_paths)
    print(f"Discovered {len(movies)} movies: {movies[:5]}...")

    task_to_h5key_map: dict[str, dict[str, str]] = {}
    for r in records:
        task_to_h5key_map[str(r["path"])] = _build_task_to_h5key_map(r["path"])

    tasks = [
        (r["path"], movie, network_name, parcel_slice, r["subject"])
        for r in records
        for movie in movies
        for network_name, parcel_slice in SCHAEFFER_7NETWORKS.items()
    ]

    print(f"Computing {len(tasks)} network RDMs (movie × subject × network)...")

    def _compute_wrapper(
        h5_path: Path, movie: str, network: str, pslice: slice, subj: str, out_dir: Path,
    ) -> dict[str, object]:
        tmap = task_to_h5key_map.get(str(h5_path))
        return compute_network_rdm(h5_path, movie, network, pslice, subj, out_dir, task_to_h5key=tmap)

    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_compute_wrapper)(h5_path, movie, network_name, parcel_slice, subject, args.output_dir)
        for h5_path, movie, network_name, parcel_slice, subject in tqdm(tasks, desc="Neural RDMs")
    )

    ok_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = sum(1 for r in results if r["status"] == "failed")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    print(f"Individual RDMs: {ok_count} ok, {fail_count} failed, {skip_count} skipped")

    group_records = compute_group_average_rdms(args.output_dir, movies, subjects, results)
    group_ok = sum(1 for r in group_records if r["status"] == "ok")
    print(f"Group-average RDMs: {group_ok} ok")

    full_manifest = {
        "fmri_root": str(args.fmri_root),
        "output_dir": str(args.output_dir),
        "subjects": subjects,
        "movies": movies,
        "networks": SCHAEFFER_NETWORK_NAMES,
        "individual_rdms": [r for r in results if r["status"] in ("ok", "failed")],
        "group_average_rdms": group_records,
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(full_manifest, indent=2))
    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()

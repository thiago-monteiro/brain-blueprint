from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for the paper-level scaling and threat experiments.")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/paper_experiment_preflight.json"))
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument("--threat-rdm", type=Path, default=Path("experiments/results/bold5000_threat_rdm/human_threat_rdm.npy"))
    parser.add_argument("--threat-manifest", type=Path, default=Path("experiments/results/bold5000_threat_rdm/stimuli_manifest.csv"))
    return parser.parse_args()


def count_files(root: Path, suffixes: tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def check_path(path: Path, *, kind: str = "path") -> dict[str, object]:
    return {"path": str(path), "kind": kind, "exists": path.exists()}


def processed_split_status(split: str) -> dict[str, object]:
    root = Path("data/processed") / split
    clips = root / "clips"
    muscles = root / "muscles"
    metadata = root / "metadata.json"
    clip_count = count_files(clips, (".mp4", ".mov", ".avi", ".mkv"))
    muscle_count = count_files(muscles, (".npy", ".npz"))
    ok = clips.is_dir() and muscles.is_dir() and metadata.is_file() and clip_count > 0 and muscle_count > 0
    return {
        "split": split,
        "ok": ok,
        "clips_dir": str(clips),
        "muscles_dir": str(muscles),
        "metadata": str(metadata),
        "clip_count": clip_count,
        "muscle_count": muscle_count,
    }


def checkpoint_status(pattern: str) -> dict[str, object]:
    paths = sorted(Path(".").glob(pattern))
    return {"pattern": pattern, "count": len(paths), "examples": [str(path) for path in paths[:5]]}


def main() -> None:
    args = parse_args()
    checks: dict[str, object] = {}
    checks["raw_data"] = {
        "amass_npz": count_files(Path("data/raw/amass"), (".npz",)),
        "mint_muscle_activations": len(list(Path("data/raw/mint_extracted").rglob("muscle_activations.pkl"))) if Path("data/raw/mint_extracted").exists() else 0,
        "babel_train": check_path(Path("data/raw/babel/babel_v1.0_release/train.json"), kind="file"),
        "smplh_female": check_path(Path("data/raw/models/smplh/SMPLH_FEMALE.pkl"), kind="file"),
        "smplh_male": check_path(Path("data/raw/models/smplh/SMPLH_MALE.pkl"), kind="file"),
    }
    checks["processed_splits"] = {split: processed_split_status(split) for split in ("train", "val", "test")}
    checks["twente"] = {
        "clips": count_files(Path("data/processed_real/twente/clips"), (".mp4", ".mov", ".avi", ".mkv")),
        "muscles": count_files(Path("data/processed_real/twente/muscles"), (".npy", ".npz")),
        "metadata": check_path(Path("data/processed_real/twente/metadata.json"), kind="file"),
    }
    checks["bold5000"] = {
        "inferotemporal_rdm": check_path(Path("egomuscle/eval/neural_rdms/inferotemporal.npy"), kind="file"),
        "stimuli_list": check_path(Path("experiments/results/bold5000_stimuli_list_neural_order.txt"), kind="file"),
        "threat_rdm": check_path(args.threat_rdm, kind="file"),
        "threat_manifest": check_path(args.threat_manifest, kind="file"),
    }
    if args.threat_rdm.exists():
        rdm = np.load(args.threat_rdm)
        checks["bold5000"]["threat_rdm_shape"] = list(rdm.shape)
        checks["bold5000"]["threat_rdm_square"] = bool(rdm.ndim == 2 and rdm.shape[0] == rdm.shape[1])
    checks["checkpoints"] = {
        "scaling": checkpoint_status("checkpoints/S_*/*.ckpt"),
        "threat": checkpoint_status("checkpoints/threat_f*_s*/*.ckpt"),
    }

    failures: list[str] = []
    raw = checks["raw_data"]
    if int(raw["amass_npz"]) == 0:
        failures.append("No AMASS .npz files under data/raw/amass.")
    if int(raw["mint_muscle_activations"]) == 0:
        failures.append("No MinT muscle_activations.pkl files under data/raw/mint_extracted.")
    for key in ("babel_train", "smplh_female", "smplh_male"):
        if not raw[key]["exists"]:
            failures.append(f"Missing {raw[key]['path']}.")
    for split, status in checks["processed_splits"].items():
        if not status["ok"]:
            failures.append(
                f"Processed split {split} is incomplete: {status['clip_count']} clips, {status['muscle_count']} muscles. "
                "Build with: python -m egomuscle.data.build_manifests --mint-root data/raw/mint_extracted "
                "--babel-root data/raw/babel --output-dir data/processed/manifests && "
                "AMASS_ROOT=data/raw/amass SMPL_ROOT=data/raw/models "
                "python experiments/build_amass_pair.py data/processed data/processed_exo"
            )
    if checks["twente"]["clips"] == 0 or checks["twente"]["muscles"] == 0 or not checks["twente"]["metadata"]["exists"]:
        failures.append("Processed Twente eval dataset is incomplete under data/processed_real/twente.")
    for key in ("inferotemporal_rdm", "stimuli_list", "threat_rdm", "threat_manifest"):
        if not checks["bold5000"][key]["exists"]:
            failures.append(f"Missing BOLD5000 artifact: {checks['bold5000'][key]['path']}.")
    if args.require_checkpoints:
        if checks["checkpoints"]["scaling"]["count"] == 0:
            failures.append("No scaling checkpoints found under checkpoints/S_*/*.ckpt.")
        if checks["checkpoints"]["threat"]["count"] == 0:
            failures.append("No threat checkpoints found under checkpoints/threat_f*_s*/*.ckpt.")

    payload = {"ok": not failures, "failures": failures, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()

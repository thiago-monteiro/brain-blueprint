from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

from .amass_smpl import resolve_amass_path
SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(value: str) -> str:
    return SAFE_CHARS.sub("_", value).strip("_")


def load_pickled_frame(path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"numpy\.core\.numeric is deprecated.*",
            category=DeprecationWarning,
        )
        with path.open("rb") as handle:
            frame = pickle.load(handle)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame in {path}")
    return frame


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def primary_activity(record: dict[str, Any]) -> str | None:
    babel = record.get("babel")
    if not babel:
        return None
    seq_ann = babel.get("seq_ann")
    if isinstance(seq_ann, dict):
        labels = seq_ann.get("labels") or []
        if labels:
            return labels[0].get("proc_label") or labels[0].get("raw_label")
    return None


def split_for_activity(activity: str) -> str:
    digest = hashlib.md5(activity.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def fit_incremental_pca(records: list[dict[str, Any]], n_components: int, batch_size: int = 20000) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=n_components)
    buffer: list[np.ndarray] = []
    buffered_rows = 0

    for record in records:
        frame = load_pickled_frame(Path(record["muscle_activations"]))
        values = frame.to_numpy(dtype=np.float32)
        buffer.append(values)
        buffered_rows += values.shape[0]
        if buffered_rows >= batch_size:
            ipca.partial_fit(np.concatenate(buffer, axis=0))
            buffer = []
            buffered_rows = 0

    if buffer:
        ipca.partial_fit(np.concatenate(buffer, axis=0))
    return ipca


def transform_sequence(record: dict[str, Any], ipca: IncrementalPCA) -> np.ndarray:
    frame = load_pickled_frame(Path(record["muscle_activations"]))
    values = frame.to_numpy(dtype=np.float32)
    return ipca.transform(values).astype(np.float32)


def download_one(item: tuple[str, str, Path], timeout: float = 30.0) -> tuple[str, bool, str | None]:
    url, stem, output_path = item
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return stem, True, None
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with output_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)
        return stem, True, None
    except Exception as exc:
        if output_path.exists():
            output_path.unlink()
        return stem, False, str(exc)


def download_videos(records: list[dict[str, Any]], cache_dir: Path, workers: int = 16) -> dict[str, dict[str, Any]]:
    jobs = []
    for record in records:
        stem = safe_stem(record["mint_key"])
        url = record["babel"]["url"]
        jobs.append((url, stem, cache_dir / f"{stem}.mp4"))

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, job): job[1] for job in jobs}
        for future in as_completed(futures):
            stem = futures[future]
            _, ok, error = future.result()
            results[stem] = {"ok": ok, "error": error}
    return results


def default_video_cache(video_source: str) -> Path:
    if video_source == "babel":
        return Path("data/raw/babel_renders")
    if video_source == "babel_recenter":
        return Path("data/raw/babel_recentered")
    if video_source == "amass_exo":
        return Path("data/raw/amass_renders/exo")
    if video_source == "amass_ego":
        return Path("data/raw/amass_renders/ego")
    raise ValueError(f"Unsupported video source: {video_source}")


def opposite_view(video_source: str) -> str:
    if video_source == "babel":
        return "babel_recenter"
    if video_source == "babel_recenter":
        return "babel"
    if video_source == "amass_ego":
        return "amass_exo"
    if video_source == "amass_exo":
        return "amass_ego"
    raise ValueError(f"{video_source} does not have an AMASS paired view.")


def resolve_smpl_model_root(model_root: Path | None) -> Path | None:
    if model_root is None:
        env_value = os.environ.get("SMPL_MODEL_DIR")
        if env_value:
            model_root = Path(env_value)
    if model_root is None:
        return None
    if not model_root.exists():
        raise FileNotFoundError(f"SMPL model root does not exist: {model_root}")
    return model_root


def ensure_processed_layout(root: Path) -> None:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (root / split / "muscles").mkdir(parents=True, exist_ok=True)
        (root / split / "clips").mkdir(parents=True, exist_ok=True)


def attach_sidecars(src_video: Path, dst_video: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for suffix, key in (
        (".quality.json", "quality_path"),
        (".recenter.json", "camera_metadata_path"),
        (".head_pose.json", "camera_metadata_path"),
        (".render.json", "camera_metadata_path"),
    ):
        src_sidecar = src_video.with_suffix(suffix)
        if not src_sidecar.exists():
            continue
        dst_sidecar = dst_video.with_suffix(suffix)
        symlink_or_copy(src_sidecar, dst_sidecar)
        payload[key] = str(dst_sidecar)
    return payload


def write_split_metadata(root: Path, split_metadata: dict[str, list[dict[str, Any]]]) -> None:
    for split, rows in split_metadata.items():
        metadata_path = root / split / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(rows, indent=2))


def prepare_recentered_videos(
    records: list[dict[str, Any]],
    *,
    raw_cache_dir: Path,
    output_cache_dir: Path,
    workers: int,
    reframe_device: str,
    render_size: int,
    render_workers: int,
    detect_every: int,
    crop_scale: float,
    min_score: float,
    overall_progress: tqdm | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_results = download_videos(records, raw_cache_dir, workers=workers)
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    def reframe_one(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        stem = safe_stem(record["mint_key"])
        raw_state = raw_results.get(stem, {"ok": False, "error": "missing_download_result"})
        if not raw_state["ok"]:
            return stem, {"ok": False, "error": raw_state["error"]}
        input_path = raw_cache_dir / f"{stem}.mp4"
        output_path = output_cache_dir / f"{stem}.mp4"
        if output_path.exists() and output_path.stat().st_size > 0:
            return stem, {"ok": True, "error": None}
        try:
            metadata_path = output_path.with_suffix(".recenter.json")
            env = os.environ.copy()
            env["EGO_MUSCLE_RENDER_PROGRESS"] = "0"
            cmd = [
                sys.executable,
                "-m",
                "egomuscle.data.reframe_babel",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--size",
                str(render_size),
                "--device",
                str(reframe_device),
                "--detect-every",
                str(detect_every),
                "--crop-scale",
                str(crop_scale),
                "--min-score",
                str(min_score),
                "--metadata-out",
                str(metadata_path),
            ]
            subprocess.run(cmd, check=True, env=env)
            result = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"success": output_path.exists()}
            if not result.get("success", True):
                return stem, {"ok": False, "error": "recenter_failed"}
            return stem, {"ok": True, "error": None}
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            return stem, {"ok": False, "error": str(exc)}

    if render_workers <= 1:
        for record in records:
            stem = safe_stem(record["mint_key"])
            if overall_progress is not None:
                overall_progress.set_postfix_str(f"babel_recenter:{stem[:48]}", refresh=False)
            result_stem, result = reframe_one(record)
            results[result_stem] = result
            if overall_progress is not None:
                overall_progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=render_workers) as pool:
            futures = {pool.submit(reframe_one, record): safe_stem(record["mint_key"]) for record in records}
            for future in as_completed(futures):
                stem = futures[future]
                if overall_progress is not None:
                    overall_progress.set_postfix_str(f"babel_recenter:{stem[:48]}", refresh=False)
                result_stem, result = future.result()
                results[result_stem] = result
                if overall_progress is not None:
                    overall_progress.update(1)
    return results, raw_results


def prepare_rendered_videos(
    records: list[dict[str, Any]],
    *,
    video_source: str,
    cache_dir: Path,
    amass_root: Path,
    smpl_model_root: Path,
    render_fps: float,
    render_size: int,
    render_device: str,
    render_chunk_size: int,
    apply_quality_filter: bool,
    clip_model_name: str,
    clip_device: str,
    similarity_threshold: float,
    head_pose_threshold: float,
    render_workers: int = 1,
    overall_progress: tqdm | None = None,
) -> dict[str, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    def render_one(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        stem = safe_stem(record["mint_key"])
        output_path = cache_dir / f"{stem}.mp4"
        if not output_path.exists() or output_path.stat().st_size == 0:
            try:
                amass_path = resolve_amass_path(record["mint_key"], amass_root)
                metadata_path = output_path.with_suffix(".render.json" if video_source == "amass_exo" else ".head_pose.json")
                env = os.environ.copy()
                env["PYOPENGL_PLATFORM"] = "osmesa"
                env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
                env["EGO_MUSCLE_RENDER_PROGRESS"] = "0"
                if video_source == "amass_exo":
                    cmd = [
                        sys.executable,
                        "-m",
                        "egomuscle.data.render_amass",
                        "--input",
                        str(amass_path),
                        "--output",
                        str(output_path),
                        "--model-root",
                        str(smpl_model_root),
                        "--fps",
                        str(render_fps),
                        "--size",
                        str(render_size),
                        "--device",
                        str(render_device),
                        "--chunk-size",
                        str(render_chunk_size),
                        "--metadata-out",
                        str(metadata_path),
                    ]
                elif video_source == "amass_ego":
                    cmd = [
                        sys.executable,
                        "-m",
                        "egomuscle.data.egox_pipeline",
                        "--input",
                        str(amass_path),
                        "--output",
                        str(output_path),
                        "--model-root",
                        str(smpl_model_root),
                        "--fps",
                        str(render_fps),
                        "--size",
                        str(render_size),
                        "--device",
                        str(render_device),
                        "--chunk-size",
                        str(render_chunk_size),
                        "--metadata-out",
                        str(metadata_path),
                    ]
                else:
                    raise ValueError(f"Unsupported rendered source: {video_source}")
                subprocess.run(cmd, check=True, env=env)
                render_result = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"success": output_path.exists()}
            except Exception as exc:
                if output_path.exists():
                    output_path.unlink()
                return stem, {"ok": False, "error": str(exc)}
            if not render_result.get("success", True):
                return stem, {"ok": False, "error": "render_failed"}

        return stem, {"ok": True, "error": None}

    if render_workers <= 1:
        for record in records:
            stem = safe_stem(record["mint_key"])
            if overall_progress is not None:
                overall_progress.set_postfix_str(f"{video_source}:{stem[:48]}", refresh=False)
            result_stem, result = render_one(record)
            results[result_stem] = result
            if overall_progress is not None:
                overall_progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=render_workers) as pool:
            futures = {pool.submit(render_one, record): safe_stem(record["mint_key"]) for record in records}
            for future in as_completed(futures):
                stem = futures[future]
                if overall_progress is not None:
                    overall_progress.set_postfix_str(f"{video_source}:{stem[:48]}", refresh=False)
                result_stem, result = future.result()
                results[result_stem] = result
                if overall_progress is not None:
                    overall_progress.update(1)

    if not apply_quality_filter:
        return results

    quality_targets = []
    for record in records:
        stem = safe_stem(record["mint_key"])
        state = results.get(stem)
        if state is None or not state["ok"]:
            continue
        quality_targets.append(cache_dir / f"{stem}.mp4")

    if not quality_targets:
        return results

    targets_file = cache_dir / "_quality_targets.txt"
    output_file = cache_dir / "_quality_reports.json"
    targets_file.write_text("\n".join(str(path) for path in quality_targets) + "\n")
    cmd = [
        sys.executable,
        "-m",
        "egomuscle.data.quality_filter",
        "--clip-dir",
        str(cache_dir),
        "--clips-file",
        str(targets_file),
        "--output",
        str(output_file),
        "--similarity-threshold",
        str(similarity_threshold),
        "--head-pose-threshold",
        str(head_pose_threshold),
        "--clip-model",
        str(clip_model_name),
        "--device",
        str(clip_device),
    ]
    subprocess.run(cmd, check=True)
    reports = json.loads(output_file.read_text())
    reports_by_clip = {str(report["clip_id"]): report for report in reports}

    for record in records:
        stem = safe_stem(record["mint_key"])
        state = results.get(stem)
        if state is None or not state["ok"]:
            continue
        report = reports_by_clip.get(stem)
        if report is None:
            results[stem] = {"ok": False, "error": "missing_quality_report"}
            continue
        (cache_dir / f"{stem}.quality.json").write_text(json.dumps(report, indent=2))
        if not bool(report.get("accepted", False)):
            results[stem] = {"ok": False, "error": "quality_filter_rejected"}
    return results


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MinT processed dataset with real or legacy video sources.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifests/mint_sequences.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--video-source", choices=["amass_ego", "amass_exo", "babel", "babel_recenter"], default="amass_ego")
    parser.add_argument("--video-cache", type=Path)
    parser.add_argument(
        "--paired-output-root",
        type=Path,
        help="Optional second processed root for the opposite AMASS view (for E3/E4 paired exports).",
    )
    parser.add_argument("--paired-video-cache", type=Path)
    parser.add_argument("--amass-root", type=Path, default=Path("data/raw/amass"))
    parser.add_argument("--smpl-model-root", type=Path)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--render-device", default="cpu")
    parser.add_argument("--render-chunk-size", type=int, default=128)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--recenter-detect-every", type=int, default=6)
    parser.add_argument("--recenter-crop-scale", type=float, default=1.8)
    parser.add_argument("--recenter-min-score", type=float, default=0.65)
    parser.add_argument("--muscle-dim", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--skip-quality-filter", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", default="cuda")
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    parser.add_argument("--head-pose-threshold", type=float, default=0.25)
    args = parser.parse_args()

    records = [record for record in load_manifest(args.manifest) if record.get("babel") is not None]
    if args.max_records is not None:
        records = records[: args.max_records]

    if args.video_cache is None:
        args.video_cache = default_video_cache(args.video_source)

    paired_video_source: str | None = None
    if args.paired_output_root is not None:
        if args.paired_output_root.resolve() == args.output_root.resolve():
            raise ValueError("--paired-output-root must be different from --output-root.")
        paired_video_source = opposite_view(args.video_source)
        if args.paired_video_cache is None:
            args.paired_video_cache = default_video_cache(paired_video_source)
        if args.paired_video_cache.resolve() == args.video_cache.resolve():
            raise ValueError("--paired-video-cache must differ from --video-cache so the paired video exports do not collide.")

    ensure_processed_layout(args.output_root)
    if args.paired_output_root is not None:
        ensure_processed_layout(args.paired_output_root)

    activity_vocab = sorted({primary_activity(record) for record in records if primary_activity(record)})
    activity_to_id = {activity: idx for idx, activity in enumerate(activity_vocab)}

    ipca = fit_incremental_pca(records, n_components=args.muscle_dim)
    np.save(args.output_root / "manifests" / "mint_pca_components.npy", ipca.components_.astype(np.float32))
    np.save(args.output_root / "manifests" / "mint_pca_mean.npy", ipca.mean_.astype(np.float32))
    if args.paired_output_root is not None:
        np.save(args.paired_output_root / "manifests" / "mint_pca_components.npy", ipca.components_.astype(np.float32))
        np.save(args.paired_output_root / "manifests" / "mint_pca_mean.npy", ipca.mean_.astype(np.float32))

    if args.video_source == "babel":
        download_results = download_videos(records, args.video_cache, workers=args.workers)
        paired_download_results = None
        if paired_video_source == "babel_recenter" and args.paired_video_cache is not None:
            progress = tqdm(total=len(records), desc="recenter babel", unit="clip", dynamic_ncols=True)
            try:
                paired_download_results, _ = prepare_recentered_videos(
                    records,
                    raw_cache_dir=args.video_cache,
                    output_cache_dir=args.paired_video_cache,
                    workers=args.workers,
                    reframe_device=args.render_device,
                    render_size=args.render_size,
                    render_workers=args.render_workers,
                    detect_every=args.recenter_detect_every,
                    crop_scale=args.recenter_crop_scale,
                    min_score=args.recenter_min_score,
                    overall_progress=progress,
                )
            finally:
                progress.close()
    elif args.video_source == "babel_recenter":
        raw_cache_dir = args.paired_video_cache if paired_video_source == "babel" and args.paired_video_cache is not None else default_video_cache("babel")
        progress = tqdm(total=len(records), desc="recenter babel", unit="clip", dynamic_ncols=True)
        try:
            download_results, raw_download_results = prepare_recentered_videos(
                records,
                raw_cache_dir=raw_cache_dir,
                output_cache_dir=args.video_cache,
                workers=args.workers,
                reframe_device=args.render_device,
                render_size=args.render_size,
                render_workers=args.render_workers,
                detect_every=args.recenter_detect_every,
                crop_scale=args.recenter_crop_scale,
                min_score=args.recenter_min_score,
                overall_progress=progress,
            )
        finally:
            progress.close()
        paired_download_results = raw_download_results if paired_video_source == "babel" else None
        if paired_video_source == "babel" and args.paired_video_cache is None:
            args.paired_video_cache = raw_cache_dir
    else:
        args.smpl_model_root = resolve_smpl_model_root(args.smpl_model_root)
        if args.smpl_model_root is None:
            raise FileNotFoundError(
                "Real AMASS rendering requires SMPL assets. Pass --smpl-model-root or set SMPL_MODEL_DIR."
            )
        total_render_jobs = len(records) * (2 if paired_video_source is not None and args.paired_video_cache is not None else 1)
        overall_progress = tqdm(total=total_render_jobs, desc="render amass", unit="clip", dynamic_ncols=True)
        try:
            download_results = prepare_rendered_videos(
                records,
                video_source=args.video_source,
                cache_dir=args.video_cache,
                amass_root=args.amass_root,
                smpl_model_root=args.smpl_model_root,
                render_fps=args.render_fps,
                render_size=args.render_size,
                render_device=args.render_device,
                render_chunk_size=args.render_chunk_size,
                apply_quality_filter=not args.skip_quality_filter,
                clip_model_name=args.clip_model,
                clip_device=args.clip_device,
                similarity_threshold=args.similarity_threshold,
                head_pose_threshold=args.head_pose_threshold,
                render_workers=args.render_workers,
                overall_progress=overall_progress,
            )
            paired_download_results = None
            if paired_video_source is not None and args.paired_video_cache is not None:
                paired_download_results = prepare_rendered_videos(
                    records,
                    video_source=paired_video_source,
                    cache_dir=args.paired_video_cache,
                    amass_root=args.amass_root,
                    smpl_model_root=args.smpl_model_root,
                    render_fps=args.render_fps,
                    render_size=args.render_size,
                    render_device=args.render_device,
                    render_chunk_size=args.render_chunk_size,
                    apply_quality_filter=not args.skip_quality_filter,
                    clip_model_name=args.clip_model,
                    clip_device=args.clip_device,
                    similarity_threshold=args.similarity_threshold,
                    head_pose_threshold=args.head_pose_threshold,
                    render_workers=args.render_workers,
                    overall_progress=overall_progress,
                )
        finally:
            overall_progress.close()
    split_metadata: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    paired_split_metadata: dict[str, list[dict[str, Any]]] | None = None
    if args.paired_output_root is not None:
        paired_split_metadata = {"train": [], "val": [], "test": []}
    failures: list[dict[str, Any]] = []

    for record in records:
        activity = primary_activity(record)
        if activity is None:
            failures.append({"mint_key": record["mint_key"], "reason": "missing_activity"})
            continue

        split = split_for_activity(activity)
        stem = safe_stem(record["mint_key"])
        download_state = download_results.get(stem, {"ok": False, "error": "missing_download_result"})
        if not download_state["ok"]:
            failures.append({"mint_key": record["mint_key"], "reason": download_state["error"]})
            continue
        if paired_video_source is not None and paired_download_results is not None:
            paired_download_state = paired_download_results.get(stem, {"ok": False, "error": "missing_paired_download_result"})
            if not paired_download_state["ok"]:
                failures.append({"mint_key": record["mint_key"], "reason": paired_download_state["error"]})
                continue

        transformed = transform_sequence(record, ipca)
        amass_path = None
        if args.video_source != "babel":
            amass_path = str(resolve_amass_path(record["mint_key"], args.amass_root))

        muscle_out = args.output_root / split / "muscles" / f"{stem}.npy"
        np.save(muscle_out, transformed)
        video_src = args.video_cache / f"{stem}.mp4"
        video_out = args.output_root / split / "clips" / f"{stem}.mp4"
        symlink_or_copy(video_src, video_out)
        sidecar_metadata = attach_sidecars(video_src, video_out)

        split_metadata[split].append(
            {
                "clip_id": stem,
                "activity": activity,
                "activity_id": activity_to_id[activity],
                "mint_key": record["mint_key"],
                "babel_sid": record["babel"]["babel_sid"],
                "video_source": args.video_source,
                "video_url": record["babel"]["url"],
                "clip_path": str(video_out),
                "muscle_path": str(muscle_out),
                "amass_path": amass_path,
                "paired_video_source": paired_video_source,
                "paired_dataset_root": str(args.paired_output_root) if args.paired_output_root is not None else None,
                "paired_clip_path": (
                    str(args.paired_output_root / split / "clips" / f"{stem}.mp4")
                    if args.paired_output_root is not None
                    else None
                ),
                **sidecar_metadata,
            }
        )
        if args.paired_output_root is not None and paired_split_metadata is not None and paired_video_source is not None:
            paired_muscle_out = args.paired_output_root / split / "muscles" / f"{stem}.npy"
            symlink_or_copy(muscle_out, paired_muscle_out)
            paired_video_src = args.paired_video_cache / f"{stem}.mp4"
            paired_video_out = args.paired_output_root / split / "clips" / f"{stem}.mp4"
            symlink_or_copy(paired_video_src, paired_video_out)
            paired_sidecar_metadata = attach_sidecars(paired_video_src, paired_video_out)
            paired_split_metadata[split].append(
                {
                    "clip_id": stem,
                    "activity": activity,
                    "activity_id": activity_to_id[activity],
                    "mint_key": record["mint_key"],
                    "babel_sid": record["babel"]["babel_sid"],
                    "video_source": paired_video_source,
                    "video_url": record["babel"]["url"],
                    "clip_path": str(paired_video_out),
                    "muscle_path": str(paired_muscle_out),
                    "amass_path": amass_path,
                    "paired_video_source": args.video_source,
                    "paired_dataset_root": str(args.output_root),
                    "paired_clip_path": str(video_out),
                    **paired_sidecar_metadata,
                }
            )

    manifests_root = args.output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    write_split_metadata(args.output_root, split_metadata)
    if args.paired_output_root is not None and paired_split_metadata is not None:
        write_split_metadata(args.paired_output_root, paired_split_metadata)

    build_report = {
        "requested_records": len(records),
        "processed_records": sum(len(rows) for rows in split_metadata.values()),
        "failures": failures,
        "activity_count": len(activity_vocab),
        "split_counts": {split: len(rows) for split, rows in split_metadata.items()},
        "video_source": args.video_source,
        "output_root": str(args.output_root),
        "paired_output_root": str(args.paired_output_root) if args.paired_output_root is not None else None,
        "paired_video_source": paired_video_source,
    }
    (manifests_root / "mint_build_report.json").write_text(json.dumps(build_report, indent=2))
    if args.paired_output_root is not None and paired_split_metadata is not None:
        paired_build_report = {
            **build_report,
            "processed_records": sum(len(rows) for rows in paired_split_metadata.values()),
            "split_counts": {split: len(rows) for split, rows in paired_split_metadata.items()},
            "video_source": paired_video_source,
            "output_root": str(args.paired_output_root),
            "paired_output_root": str(args.output_root),
            "paired_video_source": args.video_source,
        }
        (args.paired_output_root / "manifests" / "mint_build_report.json").write_text(
            json.dumps(paired_build_report, indent=2)
        )
    print(json.dumps(build_report, indent=2))


if __name__ == "__main__":
    main()

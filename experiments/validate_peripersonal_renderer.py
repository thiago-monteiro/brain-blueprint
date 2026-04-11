from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from egomuscle.data.amass_smpl import (
    SMPLH_FACE_JOINTS,
    build_scene,
    egocentric_camera_poses,
    exocentric_camera_pose,
    load_smpl_sequence,
    look_at,
    make_trimesh,
    resolve_amass_path,
    subsample_sequence,
)


VALIDATION_TOKENS = (
    "walk",
    "run",
    "dance",
    "jump",
    "throw",
    "pick up",
    "sit",
    "reach",
    "stretch",
)

CONTACT_JOINTS = {
    "head": SMPLH_FACE_JOINTS["head"],
    "neck": SMPLH_FACE_JOINTS["neck"],
    "left_shoulder": SMPLH_FACE_JOINTS["left_shoulder"],
    "right_shoulder": SMPLH_FACE_JOINTS["right_shoulder"],
    "spine3": SMPLH_FACE_JOINTS["spine3"],
    "pelvis": SMPLH_FACE_JOINTS["pelvis"],
    "left_elbow": SMPLH_FACE_JOINTS["left_elbow"],
    "right_elbow": SMPLH_FACE_JOINTS["right_elbow"],
    "left_wrist": SMPLH_FACE_JOINTS["left_wrist"],
    "right_wrist": SMPLH_FACE_JOINTS["right_wrist"],
}

CONTACT_EDGES = (
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("neck", "spine3"),
    ("spine3", "pelvis"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)

KEY_JOINTS = {
    "head": SMPLH_FACE_JOINTS["head"],
    "left_shoulder": SMPLH_FACE_JOINTS["left_shoulder"],
    "right_shoulder": SMPLH_FACE_JOINTS["right_shoulder"],
    "left_elbow": SMPLH_FACE_JOINTS["left_elbow"],
    "right_elbow": SMPLH_FACE_JOINTS["right_elbow"],
    "left_wrist": SMPLH_FACE_JOINTS["left_wrist"],
    "right_wrist": SMPLH_FACE_JOINTS["right_wrist"],
    "pelvis": SMPLH_FACE_JOINTS["pelvis"],
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def primary_activity(record: dict[str, Any]) -> str:
    babel = record.get("babel") or {}
    seq_ann = babel.get("seq_ann") or {}
    labels = seq_ann.get("labels") or []
    if not labels:
        return ""
    return (labels[0].get("proc_label") or labels[0].get("raw_label") or "").strip()


def select_records(records: list[dict[str, Any]], tokens: tuple[str, ...]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        activity = primary_activity(record).lower()
        for token in tokens:
            if token in seen:
                continue
            if token in activity:
                selected.append(record)
                seen.add(token)
                break
        if len(seen) == len(tokens):
            break
    return selected


def project_points(points: np.ndarray, camera_pose: np.ndarray, yfov: float) -> tuple[np.ndarray, np.ndarray]:
    world_to_cam = np.linalg.inv(camera_pose)
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (world_to_cam @ hom.T).T[:, :3]
    depth = -cam[:, 2]
    valid_depth = depth > 1e-4
    tan_half = math.tan(yfov * 0.5)
    x_ndc = np.full(points.shape[0], np.nan, dtype=np.float32)
    y_ndc = np.full(points.shape[0], np.nan, dtype=np.float32)
    x_ndc[valid_depth] = cam[valid_depth, 0] / (depth[valid_depth] * tan_half)
    y_ndc[valid_depth] = cam[valid_depth, 1] / (depth[valid_depth] * tan_half)
    uv = np.stack([(x_ndc + 1.0) * 0.5, (1.0 - y_ndc) * 0.5], axis=1)
    in_frame = valid_depth & (np.abs(x_ndc) <= 1.0) & (np.abs(y_ndc) <= 1.0)
    return uv, in_frame


def camera_metrics(joints: np.ndarray, poses: list[np.ndarray], *, yfov: float) -> dict[str, Any]:
    joint_names = list(KEY_JOINTS.keys())
    joint_indices = [KEY_JOINTS[name] for name in joint_names]
    visibility = defaultdict(list)
    bbox_areas: list[float] = []
    full_body_visible = 0
    workspace_visible = 0
    torso_visible = 0

    for frame_joints, pose in zip(joints, poses):
        sample = frame_joints[joint_indices]
        uv, in_frame = project_points(sample, pose, yfov)
        for name, flag in zip(joint_names, in_frame):
            visibility[name].append(float(flag))

        workspace_flags = [in_frame[joint_names.index("left_elbow")], in_frame[joint_names.index("right_elbow")], in_frame[joint_names.index("left_wrist")], in_frame[joint_names.index("right_wrist")]]
        torso_flags = [in_frame[joint_names.index("head")], in_frame[joint_names.index("left_shoulder")], in_frame[joint_names.index("right_shoulder")], in_frame[joint_names.index("pelvis")]]
        workspace_visible += int(sum(workspace_flags) >= 3)
        torso_visible += int(sum(torso_flags) >= 3)
        full_body_visible += int(np.all(in_frame))

        visible_uv = uv[in_frame]
        if len(visible_uv) >= 2:
            width = float(np.clip(visible_uv[:, 0].max() - visible_uv[:, 0].min(), 0.0, 1.0))
            height = float(np.clip(visible_uv[:, 1].max() - visible_uv[:, 1].min(), 0.0, 1.0))
            bbox_areas.append(width * height)

    frame_count = max(len(poses), 1)
    metrics: dict[str, Any] = {
        "workspace_visible_rate": workspace_visible / frame_count,
        "torso_visible_rate": torso_visible / frame_count,
        "all_key_joints_visible_rate": full_body_visible / frame_count,
        "median_key_joint_bbox_area": float(np.median(bbox_areas)) if bbox_areas else 0.0,
    }
    for name, values in visibility.items():
        metrics[f"{name}_visible_rate"] = float(np.mean(values)) if values else 0.0
    return metrics


def _draw_projection_panel(
    frame_joints: np.ndarray,
    pose: np.ndarray,
    *,
    yfov: float,
    image_size: int,
) -> np.ndarray:
    panel = np.full((image_size, image_size, 3), (236, 240, 246), dtype=np.uint8)
    sample_names = list(CONTACT_JOINTS.keys())
    sample_indices = [CONTACT_JOINTS[name] for name in sample_names]
    projected, in_frame = project_points(frame_joints[sample_indices], pose, yfov)
    pixels = np.full((len(sample_names), 2), -1, dtype=np.int32)
    if np.any(in_frame):
        pixels[in_frame] = np.round(projected[in_frame] * float(image_size - 1)).astype(np.int32)
    index_by_name = {name: idx for idx, name in enumerate(sample_names)}

    for start_name, end_name in CONTACT_EDGES:
        start_idx = index_by_name[start_name]
        end_idx = index_by_name[end_name]
        if in_frame[start_idx] and in_frame[end_idx]:
            start_px = tuple(int(v) for v in pixels[start_idx])
            end_px = tuple(int(v) for v in pixels[end_idx])
            cv2.line(panel, start_px, end_px, (82, 96, 122), 2, cv2.LINE_AA)

    for name, pixel, visible in zip(sample_names, pixels, in_frame):
        if not visible:
            continue
        color = (62, 114, 224) if "wrist" not in name else (231, 122, 71)
        cv2.circle(panel, tuple(int(v) for v in pixel), 5, color, -1, cv2.LINE_AA)

    visible_uv = projected[in_frame]
    if len(visible_uv) >= 2:
        min_uv = np.clip(visible_uv.min(axis=0), 0.0, 1.0)
        max_uv = np.clip(visible_uv.max(axis=0), 0.0, 1.0)
        min_px = tuple(int(v) for v in np.round(min_uv * float(image_size - 1)))
        max_px = tuple(int(v) for v in np.round(max_uv * float(image_size - 1)))
        cv2.rectangle(panel, min_px, max_px, (156, 166, 184), 1, cv2.LINE_AA)
    return panel


def render_contact_sheet(
    *,
    joints: np.ndarray,
    exo_poses: list[np.ndarray],
    peri_poses: list[np.ndarray],
    output_path: Path,
    image_size: int,
    yfov_peri: float,
    num_frames: int = 6,
) -> None:
    total_frames = len(joints)
    indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    canvas = np.full((num_frames * image_size, image_size * 2, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row, frame_idx in enumerate(indices):
        y0 = row * image_size
        frame_exo = _draw_projection_panel(joints[frame_idx], exo_poses[frame_idx], yfov=math.pi / 3.0, image_size=image_size)
        frame_peri = _draw_projection_panel(joints[frame_idx], peri_poses[frame_idx], yfov=yfov_peri, image_size=image_size)
        canvas[y0 : y0 + image_size, :image_size] = frame_exo
        canvas[y0 : y0 + image_size, image_size : image_size * 2] = frame_peri
        cv2.putText(canvas, f"frame {frame_idx}", (8, y0 + 24), font, 0.7, (24, 28, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, "exo", (image_size - 56, y0 + 24), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "peri", (image_size * 2 - 60, y0 + 24), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def summarize_metrics(metrics_by_view: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for view, metrics in metrics_by_view.items():
        for key, value in metrics.items():
            summary[f"{view}_{key}"] = value
    summary["workspace_gain"] = metrics_by_view["peri"]["workspace_visible_rate"] - metrics_by_view["exo"]["workspace_visible_rate"]
    summary["torso_gain"] = metrics_by_view["peri"]["torso_visible_rate"] - metrics_by_view["exo"]["torso_visible_rate"]
    summary["bbox_area_gain"] = metrics_by_view["peri"]["median_key_joint_bbox_area"] - metrics_by_view["exo"]["median_key_joint_bbox_area"]
    exo_bbox = max(float(metrics_by_view["exo"]["median_key_joint_bbox_area"]), 1e-6)
    peri_bbox = float(metrics_by_view["peri"]["median_key_joint_bbox_area"])
    exo_workspace = float(metrics_by_view["exo"]["workspace_visible_rate"])
    peri_workspace = float(metrics_by_view["peri"]["workspace_visible_rate"])
    exo_torso = float(metrics_by_view["exo"]["torso_visible_rate"])
    peri_torso = float(metrics_by_view["peri"]["torso_visible_rate"])
    exo_wrist = 0.5 * (
        float(metrics_by_view["exo"]["left_wrist_visible_rate"]) + float(metrics_by_view["exo"]["right_wrist_visible_rate"])
    )
    peri_wrist = 0.5 * (
        float(metrics_by_view["peri"]["left_wrist_visible_rate"]) + float(metrics_by_view["peri"]["right_wrist_visible_rate"])
    )

    summary["bbox_area_ratio"] = peri_bbox / exo_bbox
    summary["workspace_focus_score_exo"] = exo_bbox * exo_workspace
    summary["workspace_focus_score_peri"] = peri_bbox * peri_workspace
    summary["workspace_focus_gain"] = summary["workspace_focus_score_peri"] - summary["workspace_focus_score_exo"]
    summary["wrist_focus_score_exo"] = exo_bbox * exo_wrist
    summary["wrist_focus_score_peri"] = peri_bbox * peri_wrist
    summary["wrist_focus_gain"] = summary["wrist_focus_score_peri"] - summary["wrist_focus_score_exo"]
    summary["torso_focus_score_exo"] = exo_bbox * exo_torso
    summary["torso_focus_score_peri"] = peri_bbox * peri_torso
    summary["torso_focus_gain"] = summary["torso_focus_score_peri"] - summary["torso_focus_score_exo"]
    summary["peripersonal_preferred"] = bool(
        summary["bbox_area_ratio"] >= 1.5
        and peri_workspace >= 0.9
        and peri_torso >= 0.95
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate peripersonal renderer against exocentric rendering on diverse AMASS clips.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifests/mint_sequences.jsonl"))
    parser.add_argument("--amass-root", type=Path, default=Path("data/raw/amass"))
    parser.add_argument("--smpl-model-root", type=Path, default=Path("data/raw/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/peripersonal_validation"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--peri-yfov-deg", type=float, default=100.0)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    selected = select_records(records, VALIDATION_TOKENS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    overall_metrics: dict[str, list[float]] = defaultdict(list)

    for record in selected:
        mint_key = record["mint_key"]
        activity = primary_activity(record)
        amass_path = resolve_amass_path(mint_key, args.amass_root)
        sequence = load_smpl_sequence(amass_path, model_root=args.smpl_model_root, device=args.device, chunk_size=args.chunk_size)
        sequence = subsample_sequence(sequence, target_fps=args.fps)
        frame_limit = min(args.max_frames, len(sequence.vertices))
        vertices = sequence.vertices[:frame_limit]
        joints = sequence.joints[:frame_limit]
        exo_poses = [exocentric_camera_pose(frame_joints) for frame_joints in joints]
        peri_poses = egocentric_camera_poses(joints)
        peri_yfov = math.radians(args.peri_yfov_deg)

        exo_metrics = camera_metrics(joints, exo_poses, yfov=math.pi / 3.0)
        peri_metrics = camera_metrics(joints, peri_poses, yfov=peri_yfov)
        combined = summarize_metrics({"exo": exo_metrics, "peri": peri_metrics})
        combined["mint_key"] = mint_key
        combined["activity"] = activity
        results.append(combined)

        sheet_path = args.output_dir / f"{mint_key.replace('/', '_')}.png"
        render_contact_sheet(
            joints=joints,
            exo_poses=exo_poses,
            peri_poses=peri_poses,
            output_path=sheet_path,
            image_size=args.image_size,
            yfov_peri=peri_yfov,
        )

        for key, value in combined.items():
            if isinstance(value, (int, float)):
                overall_metrics[key].append(float(value))

    aggregate = {key: float(np.mean(values)) for key, values in overall_metrics.items() if values}
    payload = {
        "clips": results,
        "aggregate": aggregate,
        "num_clips": len(results),
        "tokens": list(VALIDATION_TOKENS),
    }
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from tqdm.auto import tqdm


PERSON_CLASS_ID = 1
_DETECTORS: dict[str, torch.nn.Module] = {}


def progress_enabled() -> bool:
    return os.environ.get("EGO_MUSCLE_RENDER_PROGRESS", "1") not in {"0", "false", "False"}


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def get_detector(device: str) -> torch.nn.Module:
    device = resolve_device(device)
    model = _DETECTORS.get(device)
    if model is not None:
        return model
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval().to(device)
    _DETECTORS[device] = model
    return model


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps


def detect_person_box(
    frame_bgr: np.ndarray,
    *,
    detector: torch.nn.Module,
    device: str,
    min_score: float,
    previous_box: np.ndarray | None,
) -> np.ndarray | None:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div_(255.0).to(device)
    with torch.inference_mode():
        prediction = detector([tensor])[0]
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    mask = (labels == PERSON_CLASS_ID) & (scores >= min_score)
    if not np.any(mask):
        return previous_box.copy() if previous_box is not None else None

    boxes = boxes[mask]
    scores = scores[mask]
    if previous_box is None:
        return boxes[int(np.argmax(scores))].astype(np.float32)

    prev_center = np.array([(previous_box[0] + previous_box[2]) * 0.5, (previous_box[1] + previous_box[3]) * 0.5], dtype=np.float32)
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)
    distances = np.linalg.norm(centers - prev_center[None, :], axis=1)
    diag = max(float(np.linalg.norm(frame_bgr.shape[:2])), 1.0)
    combined = scores - 0.35 * (distances / diag)
    return boxes[int(np.argmax(combined))].astype(np.float32)


def interpolate_series(track_values: dict[int, float], n_frames: int) -> np.ndarray:
    known_indices = np.array(sorted(track_values.keys()), dtype=np.int32)
    known_values = np.array([track_values[idx] for idx in known_indices], dtype=np.float32)
    if len(known_indices) == 1:
        return np.full(n_frames, known_values[0], dtype=np.float32)
    return np.interp(np.arange(n_frames, dtype=np.float32), known_indices.astype(np.float32), known_values).astype(np.float32)


def smooth_series(values: np.ndarray, alpha: float = 0.82) -> np.ndarray:
    if values.size == 0:
        return values
    smoothed = values.astype(np.float32).copy()
    for idx in range(1, len(smoothed)):
        smoothed[idx] = alpha * smoothed[idx - 1] + (1.0 - alpha) * smoothed[idx]
    return smoothed


def build_crop_track(
    frames: list[np.ndarray],
    *,
    detector: torch.nn.Module,
    device: str,
    detect_every: int,
    crop_scale: float,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_frames = len(frames)
    detect_indices = sorted(set(list(range(0, n_frames, max(detect_every, 1))) + [n_frames - 1]))
    center_x_track: dict[int, float] = {}
    center_y_track: dict[int, float] = {}
    crop_size_track: dict[int, float] = {}
    previous_box: np.ndarray | None = None
    detections = 0

    for frame_idx in detect_indices:
        box = detect_person_box(
            frames[frame_idx],
            detector=detector,
            device=device,
            min_score=min_score,
            previous_box=previous_box,
        )
        frame_h, frame_w = frames[frame_idx].shape[:2]
        if box is None:
            center_x = frame_w * 0.5
            center_y = frame_h * 0.5
            crop_size = float(max(frame_w, frame_h))
        else:
            previous_box = box
            detections += 1
            x1, y1, x2, y2 = box
            width = max(float(x2 - x1), 1.0)
            height = max(float(y2 - y1), 1.0)
            center_x = float((x1 + x2) * 0.5)
            center_y = float(y1 + 0.42 * height)
            crop_size = max(width * crop_scale, height * crop_scale, min(frame_w, frame_h) * 0.7)

        center_x_track[frame_idx] = center_x
        center_y_track[frame_idx] = center_y
        crop_size_track[frame_idx] = crop_size

    center_x = smooth_series(interpolate_series(center_x_track, n_frames))
    center_y = smooth_series(interpolate_series(center_y_track, n_frames))
    crop_size = smooth_series(interpolate_series(crop_size_track, n_frames))
    detection_rate = float(detections / max(len(detect_indices), 1))
    return center_x, center_y, crop_size, detection_rate


def crop_square(frame: np.ndarray, center_x: float, center_y: float, crop_size: float, output_size: int) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    crop_size = max(int(round(crop_size)), 32)
    half = crop_size // 2
    x1 = int(round(center_x)) - half
    y1 = int(round(center_y)) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - frame_w)
    pad_bottom = max(0, y2 - frame_h)
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def reframe_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    image_size: int = 512,
    device: str = "auto",
    detect_every: int = 6,
    crop_scale: float = 1.8,
    min_score: float = 0.65,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    frames, fps = read_video_frames(input_path)
    detector = get_detector(device)
    center_x, center_y, crop_size, detection_rate = build_crop_track(
        frames,
        detector=detector,
        device=device,
        detect_every=detect_every,
        crop_scale=crop_scale,
        min_score=min_score,
    )

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (image_size, image_size))
    progress = tqdm(total=len(frames), desc=f"recenter {output_path.stem}", unit="frame", dynamic_ncols=True, disable=not progress_enabled())
    try:
        for frame_idx, frame in enumerate(frames):
            reframed = crop_square(frame, center_x[frame_idx], center_y[frame_idx], crop_size[frame_idx], image_size)
            writer.write(reframed)
            progress.update(1)
    finally:
        progress.close()
        writer.release()

    return {
        "success": True,
        "frames_written": int(len(frames)),
        "fps": float(fps),
        "image_size": int(image_size),
        "camera_mode": "body_centered_reframe",
        "detect_every": int(detect_every),
        "crop_scale": float(crop_scale),
        "min_score": float(min_score),
        "detection_rate": float(detection_rate),
        "device": device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an actor-centered reframed view from a real BABEL-linked video.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detect-every", type=int, default=6)
    parser.add_argument("--crop-scale", type=float, default=1.8)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--metadata-out", type=Path)
    args = parser.parse_args()

    result = reframe_video(
        args.input,
        args.output,
        image_size=args.size,
        device=args.device,
        detect_every=args.detect_every,
        crop_scale=args.crop_scale,
        min_score=args.min_score,
    )
    metadata_out = args.metadata_out or args.output.with_suffix(".recenter.json")
    metadata_out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

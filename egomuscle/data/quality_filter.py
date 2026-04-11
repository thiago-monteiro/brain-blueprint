from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .dataset import get_video_frame_count, load_video_frames_uint8


@dataclass
class QualityReport:
    clip_id: str
    frame_count: int
    mean_adjacent_clip_similarity: float
    min_adjacent_clip_similarity: float
    head_pose_ok: bool
    duration_ok: bool
    accepted: bool
    error: str | None = None


def _load_clip_model(model_name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def adjacent_clip_similarity(
    frames_uint8: torch.Tensor,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> tuple[float, float]:
    pil_frames = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in frames_uint8]
    encoded = processor(images=pil_frames, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        vision_outputs = model.vision_model(pixel_values=encoded["pixel_values"], return_dict=True)
        pooled = vision_outputs.pooler_output
        if pooled is None:
            raise TypeError("CLIP vision tower did not return pooler_output.")
        projection = getattr(model, "visual_projection", None)
        features = projection(pooled) if projection is not None else pooled
    features = torch.nn.functional.normalize(features, dim=1)
    sims = (features[1:] * features[:-1]).sum(dim=1)
    if sims.numel() == 0:
        return 1.0, 1.0
    return float(sims.mean().item()), float(sims.min().item())


def evaluate_clip(
    clip_path: str | Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    *,
    n_frames: int = 16,
    image_size: int = 224,
    similarity_threshold: float = 0.80,
    head_pose_threshold: float = 0.25,
) -> QualityReport:
    clip_path = Path(clip_path)
    try:
        frame_count = get_video_frame_count(clip_path)
        frames_uint8 = load_video_frames_uint8(clip_path, n_frames=n_frames, image_size=image_size)
        mean_similarity, min_similarity = adjacent_clip_similarity(frames_uint8, model=model, processor=processor, device=device)
        duration_ok = frame_count >= n_frames

        head_pose_meta = clip_path.with_suffix(".head_pose.json")
        head_pose_ok = True
        if head_pose_meta.exists():
            payload = json.loads(head_pose_meta.read_text())
            head_pose_ok = bool(payload.get("success", True)) and float(payload.get("max_camera_step", 0.0)) <= head_pose_threshold

        accepted = mean_similarity >= similarity_threshold and duration_ok and head_pose_ok
        return QualityReport(
            clip_id=clip_path.stem,
            frame_count=frame_count,
            mean_adjacent_clip_similarity=mean_similarity,
            min_adjacent_clip_similarity=min_similarity,
            head_pose_ok=head_pose_ok,
            duration_ok=duration_ok,
            accepted=accepted,
        )
    except Exception as exc:
        return QualityReport(
            clip_id=clip_path.stem,
            frame_count=0,
            mean_adjacent_clip_similarity=0.0,
            min_adjacent_clip_similarity=0.0,
            head_pose_ok=False,
            duration_ok=False,
            accepted=False,
            error=str(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CLIP-based quality scores for rendered clips.")
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips-file", type=Path, help="Optional newline-delimited list of clip paths to score.")
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    parser.add_argument("--head-pose-threshold", type=float, default=0.25)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, processor = _load_clip_model(args.clip_model, device)

    clip_paths: list[Path]
    if args.clips_file is not None:
        clip_paths = [Path(line.strip()) for line in args.clips_file.read_text().splitlines() if line.strip()]
    else:
        clip_paths = sorted(args.clip_dir.glob("*.mp4"))

    reports = [
        evaluate_clip(
            clip_path,
            model=model,
            processor=processor,
            device=device,
            n_frames=args.n_frames,
            image_size=args.image_size,
            similarity_threshold=args.similarity_threshold,
            head_pose_threshold=args.head_pose_threshold,
        )
        for clip_path in clip_paths
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(report) for report in reports], indent=2))

    acceptance_rate = np.mean([report.accepted for report in reports]) if reports else 0.0
    print(f"Scored {len(reports)} clips, acceptance rate: {acceptance_rate:.2%}")


if __name__ == "__main__":
    main()

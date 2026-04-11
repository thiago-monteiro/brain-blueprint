from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from .amass_smpl import (
    build_scene,
    exocentric_camera_pose,
    load_smpl_sequence,
    make_trimesh,
    subsample_sequence,
)


def render_progress_enabled() -> bool:
    return os.environ.get("EGO_MUSCLE_RENDER_PROGRESS", "1") not in {"0", "false", "False"}


def render_mesh_sequence(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    output_path: str | Path,
    fps: float,
    image_size: int = 512,
) -> dict[str, object]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene, camera = build_scene(image_size=image_size)
    import pyrender

    camera_node = scene.add(camera, pose=np.eye(4, dtype=np.float32))
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (image_size, image_size))

    mesh_node = None
    camera_poses = []
    progress = tqdm(
        total=len(vertices),
        desc=f"exo {output_path.stem}",
        unit="frame",
        dynamic_ncols=True,
        disable=not render_progress_enabled(),
    )
    try:
        for frame_vertices, frame_joints in zip(vertices, joints):
            if mesh_node is not None:
                scene.remove_node(mesh_node)
            tri_mesh = make_trimesh(frame_vertices, faces)
            mesh_node = scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=False))
            pose = exocentric_camera_pose(frame_joints)
            scene.set_pose(camera_node, pose)
            color, _ = renderer.render(scene)
            writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            camera_poses.append(pose.astype(np.float32))
            progress.update(1)
    finally:
        progress.close()
        writer.release()
        renderer.delete()

    return {
        "success": True,
        "frames_written": int(len(vertices)),
        "fps": float(fps),
        "image_size": int(image_size),
        "camera_mode": "exocentric",
        "camera_poses": len(camera_poses),
    }


def render_amass_sequence(
    input_path: str | Path,
    output_path: str | Path,
    model_root: str | Path | None,
    *,
    target_fps: float = 30.0,
    image_size: int = 512,
    device: str = "cpu",
    chunk_size: int = 128,
) -> dict[str, object]:
    sequence = load_smpl_sequence(input_path, model_root=model_root, device=device, chunk_size=chunk_size)
    sequence = subsample_sequence(sequence, target_fps=target_fps)
    return render_mesh_sequence(
        sequence.vertices,
        sequence.joints,
        sequence.faces,
        output_path=output_path,
        fps=sequence.mocap_framerate,
        image_size=image_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a real SMPL-H AMASS sequence to third-person video.")
    parser.add_argument("--input", type=Path, required=True, help="AMASS .npz file containing SMPL-H parameters.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None, help="Directory containing SMPL/SMPL-H assets.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--metadata-out", type=Path)
    args = parser.parse_args()

    result = render_amass_sequence(
        args.input,
        args.output,
        model_root=args.model_root,
        target_fps=args.fps,
        image_size=args.size,
        device=args.device,
        chunk_size=args.chunk_size,
    )
    metadata_out = args.metadata_out or args.output.with_suffix(".render.json")
    metadata_out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import ctypes
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

for lib_path in ("/usr/lib/x86_64-linux-gnu/libstdc++.so.6", "/lib/x86_64-linux-gnu/libstdc++.so.6"):
    if Path(lib_path).exists():
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue

import smplx

from smplx import joint_names
from smplx.utils import Struct


SMPLH_VERTEX_COLOR = np.array([231, 122, 71, 255], dtype=np.uint8)
WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
SMPLH_FACE_JOINTS = {
    "neck": joint_names.JOINT_NAMES.index("neck"),
    "head": joint_names.JOINT_NAMES.index("head"),
    "jaw": joint_names.JOINT_NAMES.index("jaw"),
    "left_eye": joint_names.JOINT_NAMES.index("left_eye_smplhf"),
    "right_eye": joint_names.JOINT_NAMES.index("right_eye_smplhf"),
    "left_shoulder": joint_names.JOINT_NAMES.index("left_shoulder"),
    "right_shoulder": joint_names.JOINT_NAMES.index("right_shoulder"),
    "left_elbow": joint_names.JOINT_NAMES.index("left_elbow"),
    "right_elbow": joint_names.JOINT_NAMES.index("right_elbow"),
    "left_wrist": joint_names.JOINT_NAMES.index("left_wrist"),
    "right_wrist": joint_names.JOINT_NAMES.index("right_wrist"),
    "spine3": joint_names.JOINT_NAMES.index("spine3"),
    "pelvis": joint_names.JOINT_NAMES.index("pelvis"),
}


@dataclass(frozen=True)
class SMPLSequence:
    vertices: np.ndarray
    joints: np.ndarray
    faces: np.ndarray
    mocap_framerate: float
    gender: str
    source_path: Path


@dataclass(frozen=True)
class CameraFrame:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray


def _normalize_gender(gender: object) -> str:
    value = str(gender).strip().lower()
    if value.startswith("f"):
        return "female"
    if value.startswith("m"):
        return "male"
    return "neutral"


def _require_model_root(model_root: str | Path | None) -> Path:
    if model_root is None:
        env_value = os.environ.get("SMPL_MODEL_DIR")
        if env_value:
            model_root = env_value
    if model_root is None:
        raise FileNotFoundError(
            "SMPL body model assets are required. Set --model-root or the SMPL_MODEL_DIR environment variable."
        )
    model_root = Path(model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"SMPL model root does not exist: {model_root}")
    return model_root


def _normalize(vector: np.ndarray, default_vector: np.ndarray) -> np.ndarray:
    if not np.isfinite(vector).all():
        return default_vector.astype(np.float32).copy()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return default_vector.astype(np.float32).copy()
    return (vector / norm).astype(np.float32)


def _smplh_model_file(model_root: Path, gender: str) -> Path:
    candidate = model_root / "smplh" / f"SMPLH_{gender.upper()}.pkl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing SMPL+H model file for gender={gender}: {candidate}")


def _load_smplh_data_struct(model_root: Path, gender: str) -> tuple[Struct, int]:
    model_path = _smplh_model_file(model_root, gender)
    with model_path.open("rb") as handle:
        model_data = pickle.load(handle, encoding="latin1")
    shapedirs = np.asarray(model_data["shapedirs"], dtype=np.float32)
    available_betas = int(shapedirs.shape[-1])
    target_dim = int(smplx.body_models.SMPLH.SHAPE_SPACE_DIM)
    if available_betas < target_dim:
        pad_shape = shapedirs.shape[:2] + (target_dim - available_betas,)
        shapedirs = np.concatenate([shapedirs, np.zeros(pad_shape, dtype=np.float32)], axis=2)
        model_data["shapedirs"] = shapedirs
    return Struct(**model_data), available_betas


def resolve_amass_path(mint_key: str, amass_root: str | Path) -> Path:
    amass_root = Path(amass_root)
    parts = mint_key.split("/")
    candidates = [
        amass_root / f"{mint_key}.npz",
        amass_root / Path(*parts[1:]).with_suffix(".npz") if len(parts) > 1 else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve AMASS path for {mint_key} under {amass_root}")


def load_amass_motion(npz_path: str | Path) -> dict[str, object]:
    npz_path = Path(npz_path)
    payload = np.load(npz_path)
    required = {"poses", "trans", "betas", "gender", "mocap_framerate"}
    missing = required.difference(payload.files)
    if missing:
        raise KeyError(f"Missing AMASS keys in {npz_path}: {sorted(missing)}")
    return {
        "poses": payload["poses"].astype(np.float32),
        "trans": payload["trans"].astype(np.float32),
        "betas": payload["betas"].astype(np.float32),
        "gender": _normalize_gender(payload["gender"]),
        "mocap_framerate": float(payload["mocap_framerate"]),
    }


def _chunked_forward(
    motion: dict[str, object],
    model_root: Path,
    device: torch.device,
    chunk_size: int,
    num_betas: int,
) -> SMPLSequence:
    poses = motion["poses"]
    trans = motion["trans"]
    betas = motion["betas"]
    gender = motion["gender"]
    n_frames = poses.shape[0]
    data_struct, available_betas = _load_smplh_data_struct(model_root, str(gender))
    requested_betas = min(int(num_betas), int(betas.shape[0]), int(available_betas))

    model = smplx.create(
        str(model_root),
        model_type="smplh",
        gender=gender,
        use_pca=False,
        batch_size=min(chunk_size, n_frames),
        num_betas=requested_betas,
        data_struct=data_struct,
    ).to(device)
    faces = model.faces.astype(np.int32)
    effective_num_betas = min(int(getattr(model, "num_betas", requested_betas)), int(betas.shape[0]), requested_betas)

    vertices_parts: list[np.ndarray] = []
    joints_parts: list[np.ndarray] = []

    betas_tensor = torch.from_numpy(betas[:effective_num_betas]).float().to(device)
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk_len = end - start
        pose_chunk = torch.from_numpy(poses[start:end]).float().to(device)
        transl_chunk = torch.from_numpy(trans[start:end]).float().to(device)
        output = model(
            betas=betas_tensor.unsqueeze(0).expand(chunk_len, -1),
            global_orient=pose_chunk[:, :3],
            body_pose=pose_chunk[:, 3:66],
            left_hand_pose=pose_chunk[:, 66:111],
            right_hand_pose=pose_chunk[:, 111:156],
            transl=transl_chunk,
            return_verts=True,
        )
        vertices_parts.append(output.vertices.detach().cpu().numpy().astype(np.float32))
        joints_parts.append(output.joints.detach().cpu().numpy().astype(np.float32))

    return SMPLSequence(
        vertices=np.concatenate(vertices_parts, axis=0),
        joints=np.concatenate(joints_parts, axis=0),
        faces=faces,
        mocap_framerate=float(motion["mocap_framerate"]),
        gender=str(gender),
        source_path=Path(""),
    )


def load_smpl_sequence(
    npz_path: str | Path,
    model_root: str | Path | None = None,
    device: str | torch.device = "cpu",
    chunk_size: int = 128,
    num_betas: int = 16,
) -> SMPLSequence:
    npz_path = Path(npz_path)
    model_root_path = _require_model_root(model_root)
    motion = load_amass_motion(npz_path)
    sequence = _chunked_forward(motion, model_root_path, torch.device(device), chunk_size=chunk_size, num_betas=num_betas)
    return SMPLSequence(
        vertices=sequence.vertices,
        joints=sequence.joints,
        faces=sequence.faces,
        mocap_framerate=sequence.mocap_framerate,
        gender=sequence.gender,
        source_path=npz_path,
    )


def subsample_sequence(sequence: SMPLSequence, target_fps: float) -> SMPLSequence:
    source_fps = sequence.mocap_framerate
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if source_fps <= target_fps:
        return sequence
    step = max(int(round(source_fps / target_fps)), 1)
    indices = np.arange(0, sequence.vertices.shape[0], step)
    return SMPLSequence(
        vertices=sequence.vertices[indices],
        joints=sequence.joints[indices],
        faces=sequence.faces,
        mocap_framerate=source_fps / step,
        gender=sequence.gender,
        source_path=sequence.source_path,
    )


def make_trimesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    colors = np.broadcast_to(SMPLH_VERTEX_COLOR, (vertices.shape[0], 4)).copy()
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)


def exocentric_camera_pose(joints: np.ndarray, distance_scale: float = 2.2) -> np.ndarray:
    pelvis = joints[SMPLH_FACE_JOINTS["pelvis"]]
    head = joints[SMPLH_FACE_JOINTS["head"]]
    left_shoulder = joints[SMPLH_FACE_JOINTS["left_shoulder"]]
    right_shoulder = joints[SMPLH_FACE_JOINTS["right_shoulder"]]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    body_height = np.linalg.norm(head - pelvis)
    distance = max(distance_scale * max(body_height, shoulder_width), 1.8)
    target = pelvis + np.array([0.0, body_height * 0.45, 0.0], dtype=np.float32)
    eye = target + np.array([distance * 0.35, body_height * 0.2, distance], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return look_at(eye=eye, target=target, up=up)


def egocentric_camera_frame(
    joints: np.ndarray,
    eye_height_offset: float = 0.28,
    back_offset: float = 0.42,
    shoulder_margin: float = 0.34,
) -> CameraFrame:
    neck = joints[SMPLH_FACE_JOINTS["neck"]]
    head = joints[SMPLH_FACE_JOINTS["head"]]
    left_eye = joints[SMPLH_FACE_JOINTS["left_eye"]]
    right_eye = joints[SMPLH_FACE_JOINTS["right_eye"]]
    left_shoulder = joints[SMPLH_FACE_JOINTS["left_shoulder"]]
    right_shoulder = joints[SMPLH_FACE_JOINTS["right_shoulder"]]
    left_elbow = joints[SMPLH_FACE_JOINTS["left_elbow"]]
    right_elbow = joints[SMPLH_FACE_JOINTS["right_elbow"]]
    left_wrist = joints[SMPLH_FACE_JOINTS["left_wrist"]]
    right_wrist = joints[SMPLH_FACE_JOINTS["right_wrist"]]
    spine3 = joints[SMPLH_FACE_JOINTS["spine3"]]
    pelvis = joints[SMPLH_FACE_JOINTS["pelvis"]]

    eye_center = (left_eye + right_eye) * 0.5
    if not np.isfinite(eye_center).all() or np.linalg.norm(left_eye - right_eye) < 1e-6:
        eye_center = head

    body_up = _normalize(head - pelvis, WORLD_UP)
    eye_right = right_eye - left_eye
    if float(np.linalg.norm(eye_right)) < 1e-5:
        eye_right = right_shoulder - left_shoulder
    right = _normalize(eye_right, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    forward = _normalize(np.cross(right, body_up), np.array([0.0, 0.0, 1.0], dtype=np.float32))

    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    elbow_center = (left_elbow + right_elbow) * 0.5
    chest_center = spine3 * 0.55 + neck * 0.45
    left_reach = float(np.linalg.norm(left_wrist - chest_center))
    right_reach = float(np.linalg.norm(right_wrist - chest_center))
    total_reach_sq = max(left_reach**2 + right_reach**2, 1e-6)
    wl = (left_reach**2) / total_reach_sq
    wr = (right_reach**2) / total_reach_sq
    wrist_center = left_wrist * wl + right_wrist * wr
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

    workspace_center = (
        chest_center * 0.26
        + elbow_center * 0.22
        + wrist_center * 0.26
        + shoulder_center * 0.16
        + (chest_center + forward * max(0.28, shoulder_width * 0.8)) * 0.08
    )

    eye = chest_center + body_up * eye_height_offset - forward * back_offset
    target = workspace_center + forward * shoulder_margin - body_up * 0.02
    up = _normalize(body_up * 0.65 + WORLD_UP * 0.35, WORLD_UP)
    cam_forward = _normalize(target - eye, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    cam_right = _normalize(np.cross(cam_forward, up), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    cam_true_up = _normalize(np.cross(cam_right, cam_forward), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    margin_tan = 0.78
    max_pullback = 0.0
    for wrist in (left_wrist, right_wrist):
        w_rel = wrist - eye
        z = float(np.dot(w_rel, cam_forward))
        if z < 0.1: 
            max_pullback = max(max_pullback, 0.1 - z)
            z = 0.1
        
        x = float(np.dot(w_rel, cam_right))
        y = float(np.dot(w_rel, cam_true_up))
        
        req_z_x = abs(x) / margin_tan
        req_z_y = abs(y) / margin_tan
        
        max_pullback = max(max_pullback, req_z_x - z, req_z_y - z)
            
    if max_pullback > 0.0:
        eye = eye - cam_forward * max_pullback

    return CameraFrame(eye=eye.astype(np.float32), target=target.astype(np.float32), up=up.astype(np.float32))


def egocentric_camera_poses(joints_sequence: np.ndarray, smoothing: float = 0.88) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    prev_eye: np.ndarray | None = None
    prev_forward: np.ndarray | None = None
    prev_up: np.ndarray | None = None
    prev_distance: float | None = None

    for frame_joints in joints_sequence:
        frame = egocentric_camera_frame(frame_joints)
        raw_forward = _normalize(frame.target - frame.eye, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        raw_up = _normalize(frame.up, WORLD_UP)
        raw_distance = float(np.linalg.norm(frame.target - frame.eye))
        eye = frame.eye

        if prev_forward is not None and float(np.dot(raw_forward, prev_forward)) < 0.0:
            raw_forward = -raw_forward

        if prev_eye is not None and prev_forward is not None and prev_up is not None and prev_distance is not None:
            eye = prev_eye * smoothing + eye * (1.0 - smoothing)
            raw_forward = _normalize(prev_forward * smoothing + raw_forward * (1.0 - smoothing), prev_forward)
            raw_up = _normalize(prev_up * smoothing + raw_up * (1.0 - smoothing), prev_up)
            raw_distance = prev_distance * smoothing + raw_distance * (1.0 - smoothing)

        target = eye + raw_forward * raw_distance
        poses.append(look_at(eye=eye, target=target, up=raw_up))
        prev_eye = eye
        prev_forward = raw_forward
        prev_up = raw_up
        prev_distance = raw_distance

    return poses


def egocentric_camera_pose(
    joints: np.ndarray,
    eye_height_offset: float = 0.28,
    back_offset: float = 0.42,
    shoulder_margin: float = 0.34,
) -> np.ndarray:
    frame = egocentric_camera_frame(
        joints,
        eye_height_offset=eye_height_offset,
        back_offset=back_offset,
        shoulder_margin=shoulder_margin,
    )
    return look_at(eye=frame.eye, target=frame.target, up=frame.up)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-6)
    true_up = np.cross(right, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-6)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def build_scene(image_size: int = 512, yfov: float = np.pi / 3.0) -> tuple[Any, Any]:
    import pyrender

    scene = pyrender.Scene(
        bg_color=np.array([213, 223, 235, 255], dtype=np.uint8),
        ambient_light=np.array([0.18, 0.20, 0.24], dtype=np.float32),
    )
    camera = pyrender.PerspectiveCamera(yfov=yfov)
    key_light_pose = look_at(
        eye=np.array([2.5, 3.8, 4.0], dtype=np.float32),
        target=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        up=WORLD_UP,
    )
    fill_light_pose = look_at(
        eye=np.array([-3.5, 2.2, -1.5], dtype=np.float32),
        target=np.array([0.0, 0.8, 0.0], dtype=np.float32),
        up=WORLD_UP,
    )
    rim_light_pose = np.eye(4, dtype=np.float32)
    rim_light_pose[:3, 3] = np.array([0.0, 4.0, -4.0], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=np.array([1.0, 0.96, 0.92], dtype=np.float32), intensity=4.2), pose=key_light_pose)
    scene.add(pyrender.DirectionalLight(color=np.array([0.72, 0.82, 1.0], dtype=np.float32), intensity=2.0), pose=fill_light_pose)
    scene.add(pyrender.PointLight(color=np.array([1.0, 1.0, 1.0], dtype=np.float32), intensity=8.0), pose=rim_light_pose)

    floor = trimesh.creation.box(extents=(12.0, 0.03, 12.0))
    floor.apply_translation(np.array([0.0, -0.015, 0.0], dtype=np.float32))
    floor.visual.face_colors = np.tile(np.array([[61, 70, 84, 255]], dtype=np.uint8), (len(floor.faces), 1))
    floor_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.24, 0.27, 0.33, 1.0),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    scene.add(pyrender.Mesh.from_trimesh(floor, material=floor_material, smooth=False))
    return scene, camera

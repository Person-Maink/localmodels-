import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PREFERRED_PHASES = ("smooth_fit", "prior")


@dataclass(frozen=True)
class DynHAMRRun:
    clip_id: str
    date_text: str
    run_name: str
    run_root: Path
    phase_name: str
    mesh_root: Path
    world_results_path: Path
    cameras_json_path: Path
    track_info_path: Path


def _extract_clip_id(run_name):
    marker = "-all-shot-"
    if marker in run_name:
        return run_name.split(marker, 1)[0]
    return run_name


def parse_obj_vertices(obj_path):
    vertices = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                break

    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {obj_path}")

    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Invalid OBJ vertex array shape in {obj_path}: {verts.shape}")
    return verts


def load_camera_json(cameras_json_path):
    with open(cameras_json_path, "r", encoding="utf-8") as f:
        cam_data = json.load(f)

    rotation = np.asarray(cam_data["rotation"], dtype=np.float32).reshape(-1, 3, 3)
    translation = np.asarray(cam_data["translation"], dtype=np.float32).reshape(-1, 3)
    intrinsics = np.asarray(cam_data["intrinsics"], dtype=np.float32)
    if intrinsics.ndim == 1:
        intrinsics = np.broadcast_to(intrinsics.reshape(1, 4), (len(rotation), 4)).copy()
    else:
        intrinsics = intrinsics.reshape(-1, 4)

    return {
        "rotation": rotation,
        "translation": translation,
        "intrinsics": intrinsics,
    }


def _resolve_phase_assets(run_root, clip_id):
    for phase_name in PREFERRED_PHASES:
        phase_root = run_root / phase_name
        if not phase_root.is_dir():
            continue

        mesh_dirs = sorted(phase_root.glob(f"{clip_id}_*_meshes"))
        world_files = sorted(phase_root.glob(f"{clip_id}_*_world_results.npz"))
        if not mesh_dirs or not world_files:
            continue

        mesh_by_prefix = {path.name[: -len("_meshes")]: path for path in mesh_dirs}
        world_by_prefix = {path.name[: -len("_world_results.npz")]: path for path in world_files}
        shared_prefixes = sorted(set(mesh_by_prefix) & set(world_by_prefix))
        if shared_prefixes:
            prefix = shared_prefixes[-1]
            return phase_name, mesh_by_prefix[prefix], world_by_prefix[prefix]

        return phase_name, mesh_dirs[-1], world_files[-1]

    return None, None, None


def discover_complete_dynhamr_runs(logs_root):
    logs_root = Path(logs_root)
    runs = []
    if not logs_root.exists():
        return runs

    for date_dir in sorted([path for path in logs_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        for run_root in sorted([path for path in date_dir.iterdir() if path.is_dir()], key=lambda path: path.name):
            clip_id = _extract_clip_id(run_root.name)
            phase_name, mesh_root, world_results_path = _resolve_phase_assets(run_root, clip_id)
            cameras_json_path = run_root / "cameras.json"
            track_info_path = run_root / "track_info.json"

            if phase_name is None:
                continue
            if not cameras_json_path.is_file() or not track_info_path.is_file():
                continue

            runs.append(
                DynHAMRRun(
                    clip_id=clip_id,
                    date_text=date_dir.name,
                    run_name=run_root.name,
                    run_root=run_root,
                    phase_name=phase_name,
                    mesh_root=mesh_root,
                    world_results_path=world_results_path,
                    cameras_json_path=cameras_json_path,
                    track_info_path=track_info_path,
                )
            )

    return runs


def resolve_latest_complete_dynhamr_runs(logs_root):
    resolved = {}
    for run in discover_complete_dynhamr_runs(logs_root):
        current = resolved.get(run.clip_id)
        if current is None or (run.date_text, run.run_name) > (current.date_text, current.run_name):
            resolved[run.clip_id] = run
    return resolved


def _iter_mesh_entries(mesh_root):
    entries = []
    for obj_path in sorted(mesh_root.glob("*.obj")):
        stem = obj_path.stem
        try:
            frame_text, track_text = stem.rsplit("_", 1)
            frame_idx = int(frame_text)
            track_idx = int(track_text)
        except ValueError:
            continue
        entries.append((frame_idx, track_idx, obj_path))

    if not entries:
        raise RuntimeError(f"No OBJ frame meshes found under {mesh_root}")
    return entries


def _img_res_from_intrinsics(intrinsics_row):
    fx, fy, cx, cy = np.asarray(intrinsics_row, dtype=np.float32).reshape(4)
    width = int(round(float(cx) * 2.0))
    height = int(round(float(cy) * 2.0))
    return np.asarray([width, height], dtype=np.int32), float(fx), float(fy)


def _pose_wc(rotation, translation):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    pose[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return pose


def export_dynhamr_run(run, output_root, overwrite=False):
    output_root = Path(output_root)
    clip_root = output_root / run.clip_id
    meshes_root = clip_root / "meshes"
    camera_poses_path = clip_root / "camera_poses.npz"

    if clip_root.exists() and overwrite:
        shutil.rmtree(clip_root)

    meshes_root.mkdir(parents=True, exist_ok=True)

    world_results = np.load(run.world_results_path, allow_pickle=True)
    is_right = np.asarray(world_results["is_right"], dtype=np.float32)
    cam_t = np.asarray(world_results["cam_t"], dtype=np.float32)

    cam_data = load_camera_json(run.cameras_json_path)
    mesh_entries = _iter_mesh_entries(run.mesh_root)

    pose_rows = []
    frame_rows = []
    right_rows = []
    track_rows = []
    intrinsics_rows = []

    for frame_idx, track_idx, obj_path in mesh_entries:
        if frame_idx >= len(cam_data["rotation"]):
            raise IndexError(f"Frame {frame_idx} out of range for camera data in {run.cameras_json_path}")
        if track_idx >= is_right.shape[0] or frame_idx >= is_right.shape[1]:
            raise IndexError(f"Track/frame ({track_idx}, {frame_idx}) out of range for {run.world_results_path}")

        verts_world = parse_obj_vertices(obj_path)
        right_value = int(float(is_right[track_idx, frame_idx]))
        cam_t_value = np.asarray(cam_t[track_idx, frame_idx], dtype=np.float32).reshape(3)
        intrinsics_row = cam_data["intrinsics"][frame_idx]
        img_res, focal_length_x, _ = _img_res_from_intrinsics(intrinsics_row)

        frame_dir = meshes_root / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        out_file = frame_dir / f"frame_{frame_idx:06d}_{track_idx}_{float(right_value):.1f}_verts.npy"
        record = {
            "verts_world": verts_world,
            "right": np.asarray(right_value, dtype=np.int32),
            "frame_id": np.asarray(frame_idx, dtype=np.int32),
            "track_id": np.asarray(track_idx, dtype=np.int32),
            "cam_t": cam_t_value,
            "focal_length": float(focal_length_x),
            "img_res": img_res,
            "source_run": str(run.run_root),
            "source_phase": run.phase_name,
        }
        np.save(out_file, record, allow_pickle=True)

        pose_rows.append(_pose_wc(cam_data["rotation"][frame_idx], cam_data["translation"][frame_idx]))
        frame_rows.append(frame_idx)
        right_rows.append(right_value)
        track_rows.append(track_idx)
        intrinsics_rows.append(np.asarray(intrinsics_row, dtype=np.float32).reshape(4))

    np.savez(
        camera_poses_path,
        poses_wc=np.asarray(pose_rows, dtype=np.float32),
        frame_id=np.asarray(frame_rows, dtype=np.int32),
        right=np.asarray(right_rows, dtype=np.int32),
        track_id=np.asarray(track_rows, dtype=np.int32),
        intrinsics=np.asarray(intrinsics_rows, dtype=np.float32),
    )

    metadata = {
        "clip_id": run.clip_id,
        "date": run.date_text,
        "run_name": run.run_name,
        "run_root": str(run.run_root),
        "phase": run.phase_name,
        "mesh_root": str(run.mesh_root),
        "world_results": str(run.world_results_path),
        "camera_poses": str(camera_poses_path),
        "records_exported": len(mesh_entries),
    }
    with open(clip_root / "dynhamr_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def export_latest_dynhamr_runs(logs_root, output_root, clip_ids=None, overwrite=False):
    resolved = resolve_latest_complete_dynhamr_runs(logs_root)
    selected_clip_ids = sorted(resolved)
    if clip_ids is not None:
        wanted = {str(item) for item in clip_ids}
        selected_clip_ids = [clip_id for clip_id in selected_clip_ids if clip_id in wanted]

    summaries = []
    for clip_id in selected_clip_ids:
        summaries.append(export_dynhamr_run(resolved[clip_id], output_root, overwrite=overwrite))
    return summaries

from pathlib import Path

import numpy as np


def load_vipe_pose_artifact(pose_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not pose_path.exists():
        raise FileNotFoundError(f"ViPE pose file not found: {pose_path}")

    data = np.load(pose_path)
    if "inds" not in data or "data" not in data:
        raise ValueError(f"{pose_path} must contain 'inds' and 'data'.")

    inds = np.asarray(data["inds"], dtype=np.int64)
    poses = np.asarray(data["data"], dtype=np.float32)
    if inds.ndim != 1 or poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"Unexpected ViPE pose shapes in {pose_path}: inds={inds.shape}, data={poses.shape}"
        )

    order = np.argsort(inds)
    return inds[order], poses[order]


def load_vipe_intrinsics_artifact(intrinsics_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"ViPE intrinsics file not found: {intrinsics_path}")

    data = np.load(intrinsics_path)
    if "inds" not in data or "data" not in data:
        raise ValueError(f"{intrinsics_path} must contain 'inds' and 'data'.")

    inds = np.asarray(data["inds"], dtype=np.int64)
    intrinsics = np.asarray(data["data"], dtype=np.float32)
    if inds.ndim != 1 or intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
        raise ValueError(
            f"Unexpected ViPE intrinsics shapes in {intrinsics_path}: inds={inds.shape}, data={intrinsics.shape}"
        )

    order = np.argsort(inds)
    return inds[order], intrinsics[order]


def pick_value_for_frame(
    frame_idx: int,
    inds: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    pos = np.searchsorted(inds, frame_idx, side="right") - 1
    if pos >= 0:
        source_idx = int(inds[pos])
        return values[pos], source_idx, frame_idx != source_idx
    return values[0], int(inds[0]), True


def resolve_vipe_camera_sequence(video_name: str, frame_ids, vipe_output_root: str | Path) -> dict:
    root = Path(vipe_output_root)
    pose_inds, poses_c2w = load_vipe_pose_artifact(root / "pose" / f"{video_name}.npz")
    intr_inds, intrinsics_all = load_vipe_intrinsics_artifact(root / "intrinsics" / f"{video_name}.npz")

    cam_r = []
    cam_t = []
    intrinsics = []
    pose_frame_ids = []
    intr_frame_ids = []
    pose_fallback = []
    intr_fallback = []

    for frame_id in np.asarray(frame_ids, dtype=np.int64):
        pose_c2w, pose_source_idx, pose_used_fallback = pick_value_for_frame(
            int(frame_id), pose_inds, poses_c2w
        )
        intr, intr_source_idx, intr_used_fallback = pick_value_for_frame(
            int(frame_id), intr_inds, intrinsics_all
        )
        pose_w2c = np.linalg.inv(pose_c2w)

        cam_r.append(pose_w2c[:3, :3].astype(np.float32))
        cam_t.append(pose_w2c[:3, 3].astype(np.float32))
        intrinsics.append(np.asarray(intr, dtype=np.float32))
        pose_frame_ids.append(np.int64(pose_source_idx))
        intr_frame_ids.append(np.int64(intr_source_idx))
        pose_fallback.append(bool(pose_used_fallback))
        intr_fallback.append(bool(intr_used_fallback))

    return {
        "cam_R": np.asarray(cam_r, dtype=np.float32),
        "cam_t": np.asarray(cam_t, dtype=np.float32),
        "intrinsics": np.asarray(intrinsics, dtype=np.float32),
        "pose_frame_ids": np.asarray(pose_frame_ids, dtype=np.int64),
        "intrinsics_frame_ids": np.asarray(intr_frame_ids, dtype=np.int64),
        "pose_fallback_mask": np.asarray(pose_fallback, dtype=bool),
        "intrinsics_fallback_mask": np.asarray(intr_fallback, dtype=bool),
    }

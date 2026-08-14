from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from npy_io import iter_model_frame_records, resolve_model_camera_poses_file

from .mano import coerce_rotations, load_mano_assets, mano_forward
from .settings import AppSettings, hand_label_to_value, hand_value_to_label


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
]


def frame_in_range(frame_id: int, frame_start: Optional[int], frame_end: Optional[int]) -> bool:
    if frame_start is not None and int(frame_id) < int(frame_start):
        return False
    if frame_end is not None and int(frame_id) > int(frame_end):
        return False
    return True


def source_kind(source_path: str) -> str:
    return "mediapipe" if str(source_path).lower().endswith(".csv") else "model"


def source_mtime_key(path_text: str) -> str:
    path = Path(path_text)
    if not path.exists():
        return "missing"
    if path.is_file():
        stat = path.stat()
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"

    digest = hashlib.sha1()
    digest.update(str(path).encode("utf-8"))
    for child in sorted(path.rglob("*")):
        if child.is_file():
            stat = child.stat()
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def figure_to_svg(fig) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    svg = buffer.getvalue()
    plt.close(fig)
    return svg


def extract_frame_id_from_path(path: Path) -> int:
    for name in (path.name, path.parent.name):
        match = re.match(r"frame_(\d+)$", name)
        if match:
            return int(match.group(1))
    return -1


def _load_mediapipe_frames(csv_path: str, frame_start: Optional[int] = None, frame_end: Optional[int] = None) -> List[dict]:
    df = pd.read_csv(csv_path)
    frames = []
    for frame_id in sorted(df["frame_id"].unique()):
        if not frame_in_range(int(frame_id), frame_start, frame_end):
            continue
        frame_df = df[df["frame_id"] == frame_id]
        hands = []
        for hand_id, hand_df in frame_df.groupby("hand_id"):
            hand_df = hand_df.sort_values("joint_id")
            points = hand_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            hands.append(
                {
                    "right": hand_label_to_value(str(hand_id)),
                    "points": points,
                    "joint_ids": hand_df["joint_id"].astype(int).tolist(),
                }
            )
        frames.append({"frame_id": int(frame_id), "hands": hands})
    return frames


def _record_global_orient_matrix(record: dict, source_path: Path) -> np.ndarray:
    pred_mano_params = record.get("pred_mano_params", {})
    global_orient = pred_mano_params.get("global_orient")
    if global_orient is None:
        return np.eye(3, dtype=np.float32)
    return coerce_rotations(global_orient, 1, "global_orient", source_path)[0]


def _model_record_to_hand(record: dict, j_regressor: np.ndarray, include_camera_space: bool = False, wrist_joint_idx: int = 0) -> dict:
    verts_world = np.asarray(record["verts_world"], dtype=np.float32)
    joints_world = np.asarray(j_regressor @ verts_world, dtype=np.float32)
    right = int(record.get("right", -1))
    hand = {
        "right": right,
        "verts_world": verts_world,
        "joints_world": joints_world,
        "cam_t": np.asarray(record.get("cam_t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3),
        "box_center": None if record.get("box_center") is None else np.asarray(record["box_center"], dtype=np.float32).reshape(2),
        "box_size": None if record.get("box_size") is None else float(record["box_size"]),
        "track_id": int(record.get("track_id", 0) or 0),
        "path": str(record.get("path", "")),
    }
    if include_camera_space:
        wrist = joints_world[int(wrist_joint_idx)]
        global_orient = _record_global_orient_matrix(record, Path(hand["path"]) if hand["path"] else Path("."))
        verts = np.asarray(record.get("verts", verts_world), dtype=np.float32)
        cam_t = hand["cam_t"]
        verts_world_camera = verts + cam_t[None, :]
        joints_world_camera = np.asarray(j_regressor @ verts_world_camera, dtype=np.float32)
        wrist_camera = joints_world_camera[int(wrist_joint_idx)]
        verts_no_global = (verts_world_camera - wrist_camera[None, :]) @ global_orient.T
        hand["verts_camera_space"] = verts_no_global + wrist_camera[None, :]
        hand["verts_camera_centered"] = verts_no_global
    return hand


def load_model_frames(
    source_path: str,
    settings: AppSettings,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    include_camera_space: bool = False,
    wrist_joint_idx: int = 0,
) -> List[dict]:
    mano = load_mano_assets(str(settings.mano_right_path))
    j_regressor = mano["j_regressor"]
    frames = []
    for discovered_frame_id, records in iter_model_frame_records(source_path, pattern="*.npy"):
        if not records:
            continue
        frame_id = int(records[0].get("frame_id", discovered_frame_id) or discovered_frame_id)
        if not frame_in_range(frame_id, frame_start, frame_end):
            continue
        hands = [
            _model_record_to_hand(
                record,
                j_regressor=j_regressor,
                include_camera_space=include_camera_space,
                wrist_joint_idx=wrist_joint_idx,
            )
            for record in records
        ]
        frames.append({"frame_id": frame_id, "hands": hands})
    return frames


def load_source_frames(
    source: dict,
    settings: AppSettings,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    include_camera_space: bool = False,
    wrist_joint_idx: int = 0,
) -> List[dict]:
    if source["family"] == "mediapipe":
        return _load_mediapipe_frames(source["path"], frame_start=frame_start, frame_end=frame_end)
    return load_model_frames(
        source["path"],
        settings=settings,
        frame_start=frame_start,
        frame_end=frame_end,
        include_camera_space=include_camera_space,
        wrist_joint_idx=wrist_joint_idx,
    )


def source_metadata(source: dict, settings: AppSettings) -> dict:
    frames = load_source_frames(source, settings=settings)
    hands = sorted(
        {hand_value_to_label(hand["right"]) for frame in frames for hand in frame["hands"]}
    )
    bbox_capable = any(
        hand.get("box_center") is not None and hand.get("box_size") is not None
        for frame in frames
        for hand in frame["hands"]
        if isinstance(hand, dict)
    )
    frame_ids = [int(frame["frame_id"]) for frame in frames]
    return {
        "source_id": source["id"],
        "frame_count": len(frame_ids),
        "frame_ids": frame_ids,
        "first_frame_id": frame_ids[0] if frame_ids else None,
        "last_frame_id": frame_ids[-1] if frame_ids else None,
        "available_hands": hands,
        "bounding_boxes": bbox_capable,
        "path": source["path"],
    }


def _load_camera_pose_npz(camera_poses_file: Path) -> dict:
    data = np.load(camera_poses_file, allow_pickle=True)
    try:
        if "poses_wc" in data:
            poses_wc = np.asarray(data["poses_wc"], dtype=np.float32)
        elif "cam_R" in data and "cam_t" in data:
            cam_t = np.asarray(data["cam_t"], dtype=np.float32).reshape(-1, 3)
            cam_r = np.asarray(data["cam_R"], dtype=np.float32).reshape(-1, 3, 3)
            poses_wc = np.repeat(np.eye(4, dtype=np.float32)[None], len(cam_t), axis=0)
            poses_wc[:, :3, :3] = cam_r
            poses_wc[:, :3, 3] = cam_t
        else:
            raise KeyError(f"Unsupported camera pose payload: {camera_poses_file}")
        frame_id = np.asarray(data["frame_id"], dtype=np.int32).reshape(-1) if "frame_id" in data else np.arange(len(poses_wc))
        right = np.asarray(data["right"], dtype=np.int32).reshape(-1) if "right" in data else np.full((len(poses_wc),), -1, dtype=np.int32)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32) if "intrinsics" in data else None
    finally:
        data.close()
    return {"poses_wc": poses_wc, "frame_id": frame_id, "right": right, "intrinsics": intrinsics}


def load_model_camera_poses(source: dict, settings: AppSettings, invert_cam_t: bool = True) -> dict:
    camera_poses_file = resolve_model_camera_poses_file(source["path"])
    if camera_poses_file is not None:
        payload = _load_camera_pose_npz(Path(camera_poses_file))
        return payload

    frames = load_model_frames(source["path"], settings=settings)
    rows = []
    rights = []
    frame_ids = []
    for frame in frames:
        for hand in frame["hands"]:
            pose = np.eye(4, dtype=np.float32)
            cam_t = np.asarray(hand["cam_t"], dtype=np.float32).reshape(3)
            pose[:3, 3] = -cam_t if invert_cam_t else cam_t
            rows.append(pose)
            rights.append(int(hand["right"]))
            frame_ids.append(int(frame["frame_id"]))

    return {
        "poses_wc": np.asarray(rows, dtype=np.float32),
        "frame_id": np.asarray(frame_ids, dtype=np.int32),
        "right": np.asarray(rights, dtype=np.int32),
        "intrinsics": None,
    }


def load_vipe_overlay(overlay: dict) -> dict:
    data = np.load(overlay["pose_path"], allow_pickle=True)
    try:
        frame_ids = np.asarray(data["inds"], dtype=np.int32).reshape(-1)
        poses = np.asarray(data["data"], dtype=np.float32).reshape(-1, 4, 4)
    finally:
        data.close()
    return {"frame_id": frame_ids, "poses_wc": poses}


def extract_beta_average_records(source: dict, settings: AppSettings, hand_filter: Optional[int] = None) -> dict:
    mano = load_mano_assets(str(settings.mano_right_path))
    records = []
    source_path = Path(source["path"]).expanduser().resolve()
    for _, recs in iter_model_frame_records(str(source_path), pattern="*.npy"):
        for rec in recs:
            pred = rec.get("pred_mano_params")
            if not isinstance(pred, dict):
                continue
            if hand_filter is not None and int(rec.get("right", -1)) != int(hand_filter):
                continue
            if "betas" not in pred or "global_orient" not in pred or "hand_pose" not in pred:
                continue
            path_hint = Path(str(rec.get("path", source_path)))
            records.append(
                {
                    "frame_id": int(rec.get("frame_id", extract_frame_id_from_path(path_hint))),
                    "right": int(rec.get("right", -1)),
                    "score": float(rec.get("detection_confidence", 1.0) or 1.0),
                    "betas": np.asarray(pred["betas"], dtype=np.float32).reshape(-1),
                    "global_orient": coerce_rotations(pred["global_orient"], 1, "global_orient", path_hint),
                    "hand_pose": coerce_rotations(pred["hand_pose"], 15, "hand_pose", path_hint),
                    "cam_t": np.asarray(rec.get("cam_t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3),
                }
            )
    if not records:
        raise RuntimeError(f"No MANO parameterized records available for beta-average reconstruction in {source['path']}")

    average_betas = np.stack([record["betas"] for record in records], axis=0).mean(axis=0)
    betas_batch = np.broadcast_to(average_betas[None], (len(records), len(average_betas))).copy()
    global_orient = np.stack([record["global_orient"] for record in records], axis=0)
    hand_pose = np.stack([record["hand_pose"] for record in records], axis=0)
    right = np.asarray([record["right"] for record in records], dtype=np.int32)
    cam_t = np.stack([record["cam_t"] for record in records], axis=0)
    verts_local = mano_forward(mano, betas_batch, global_orient, hand_pose, right)
    verts_camera = verts_local + cam_t[:, None, :]

    frames: Dict[int, List[dict]] = {}
    for record, verts in zip(records, verts_camera):
        frames.setdefault(record["frame_id"], []).append(
            {
                "right": record["right"],
                "score": record["score"],
                "verts_world": np.asarray(verts, dtype=np.float32),
            }
        )
    return {
        "mano": mano,
        "average_betas": average_betas,
        "frames": [{"frame_id": frame_id, "hands": frames[frame_id]} for frame_id in sorted(frames)],
    }


def json_clone(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clone(item) for item in value]
    return value

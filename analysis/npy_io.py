import glob
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np


STRIDE_SEQUENCE_FILENAMES = (
    "refined_sequence.npz",
)


def resolve_model_record_root(source_path):
    path = Path(source_path)
    if path.is_file():
        path = path.parent

    candidates = [path, path / "meshes"]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(frame_dir.is_dir() for frame_dir in candidate.glob("frame_*")):
            return candidate

    return None


def list_frame_folders(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "frame_*")))


def load_wilor_record(npy_path):
    """
    Parse one model record saved as:
      np.save(path, {"verts": verts, "cam_t": cam_t, "right": is_right, ...})
    or
      np.save(path, {"verts_world": verts_world, "right": is_right, ...})
    """
    arr = np.load(npy_path, allow_pickle=True)

    if not (isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ()):
        raise ValueError(f"Unsupported npy format: {npy_path}")

    item = arr.item()
    if not isinstance(item, dict) or ("verts" not in item and "verts_world" not in item):
        raise ValueError(f"Missing dict/verts or dict/verts_world in: {npy_path}")

    cam_t = np.asarray(item.get("cam_t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    verts = item.get("verts", None)
    verts_world = item.get("verts_world", None)

    if verts is not None:
        verts = np.asarray(verts, dtype=np.float32)
    if verts_world is not None:
        verts_world = np.asarray(verts_world, dtype=np.float32)

    if verts_world is None:
        if verts is None:
            raise ValueError(f"Missing vertices in: {npy_path}")
        verts_world = verts + cam_t.reshape(1, 3)
    elif verts is None:
        # Some normalized exports store only world-space vertices.
        verts = verts_world.copy()

    right = int(float(np.asarray(item.get("right", -1)).item()))
    box_center = item.get("box_center", None)
    box_size = item.get("box_size", None)
    frame_id = item.get("frame_id", None)
    track_id = item.get("track_id", None)

    if box_center is not None:
        box_center = np.asarray(box_center, dtype=np.float32).reshape(2)
    if box_size is not None:
        box_size = float(np.asarray(box_size, dtype=np.float32).reshape(()))
    if frame_id is not None:
        frame_id = int(np.asarray(frame_id, dtype=np.int32).reshape(()))
    if track_id is not None:
        track_id = int(np.asarray(track_id, dtype=np.int32).reshape(()))

    return {
        "verts": verts,
        "cam_t": cam_t,
        "verts_world": verts_world,
        "right": right,
        "box_center": box_center,
        "box_size": box_size,
        "frame_id": frame_id,
        "track_id": track_id,
        "path": npy_path,
    }


def load_frame_records(frame_dir, pattern="*_verts.npy"):
    records = []
    npy_paths = sorted(glob.glob(os.path.join(frame_dir, pattern)))

    # Some exports write plain *.npy files instead of *_verts.npy.
    if not npy_paths and pattern == "*_verts.npy":
        npy_paths = sorted(glob.glob(os.path.join(frame_dir, "*.npy")))

    for npy_path in npy_paths:
        try:
            records.append(load_wilor_record(npy_path))
        except Exception as exc:
            print(f"[npy_io] Failed to load {npy_path}: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue
    return records


def parse_frame_index(path):
    path = Path(path)
    for name in (path.name, path.parent.name):
        match = re.match(r"frame_(\d+)$", name)
        if match:
            return int(match.group(1))
    return -1


def discover_frame_files(frames_root, frame_dirs_glob="frame_*", file_glob="*.npy"):
    root = Path(frames_root)
    if not root.exists():
        raise FileNotFoundError(f"frames_root does not exist: {root}")

    frame_dirs = [path for path in root.glob(frame_dirs_glob) if path.is_dir()]
    frame_dirs = sorted(frame_dirs, key=lambda path: (parse_frame_index(path), path.name))

    discovered = []
    if frame_dirs:
        for frame_dir in frame_dirs:
            frame_idx = parse_frame_index(frame_dir)
            for file_path in sorted(frame_dir.glob(file_glob)):
                if file_path.is_file():
                    discovered.append((frame_idx, file_path))
    else:
        for file_path in sorted(root.glob(file_glob)):
            if file_path.is_file():
                discovered.append((parse_frame_index(file_path), file_path))

    return discovered


def resolve_stride_sequence_file(source_path):
    path = Path(source_path)

    if path.is_dir():
        for filename in STRIDE_SEQUENCE_FILENAMES:
            candidate = path / filename
            if candidate.is_file():
                return candidate
        return None

    if path.is_file() and path.suffix.lower() == ".npz":
        if path.name in STRIDE_SEQUENCE_FILENAMES:
            return path
    return None


def is_stride_source(source_path):
    return resolve_stride_sequence_file(source_path) is not None


def resolve_model_camera_poses_file(source_path):
    path = Path(source_path)
    if path.is_file():
        path = path.parent

    if not is_stride_source(path):
        return None

    camera_poses_file = path / "camera_poses.npz"
    if camera_poses_file.is_file():
        return camera_poses_file
    return None


def _normalize_stride_hand_value(value):
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return -1

    item = arr[0]
    if isinstance(item, (bool, np.bool_)):
        return 1 if bool(item) else 0

    try:
        return int(float(item))
    except (TypeError, ValueError):
        return -1


def _load_stride_right_values(clip_root, n_frames):
    track_info_path = clip_root / "track_info.json"
    if track_info_path.is_file():
        try:
            payload = json.loads(track_info_path.read_text(encoding="utf-8"))
            tracks = payload.get("tracks") or {}
            if tracks:
                first_track = next(iter(tracks.values()))
                hand_value = _normalize_stride_hand_value(first_track.get("right", -1))
                return np.full((n_frames,), hand_value, dtype=np.int32)
        except Exception:
            pass

    metadata_path = clip_root / "stride_metadata.json"
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            hand_value = _normalize_stride_hand_value(payload.get("right", -1))
            return np.full((n_frames,), hand_value, dtype=np.int32)
        except Exception:
            pass

    handedness_npz = clip_root / "refined_world_results.npz"
    if handedness_npz.is_file():
        try:
            data = np.load(handedness_npz, allow_pickle=True)
            if "is_right" in data:
                right = np.asarray(data["is_right"]).reshape(-1)
                if right.size == 1 and n_frames > 1:
                    right = np.full((n_frames,), _normalize_stride_hand_value(right[0]), dtype=np.int32)
                elif right.size == n_frames:
                    right = np.asarray(
                        [_normalize_stride_hand_value(value) for value in right],
                        dtype=np.int32,
                    )
                else:
                    right = None
                data.close()
                if right is not None:
                    return right
            else:
                data.close()
        except Exception:
            pass

    return np.full((n_frames,), -1, dtype=np.int32)


def load_stride_right_values(clip_root, n_frames):
    return _load_stride_right_values(Path(clip_root), n_frames)


def _frame_major_stride_array(array, expected_name, n_frames=None):
    arr = np.asarray(array)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]

    if n_frames is not None and arr.shape[0] != int(n_frames):
        raise ValueError(
            f"Invalid frame-major shape for {expected_name}: expected first axis {n_frames}, got {arr.shape}"
        )
    return arr


def iter_stride_frame_records(source_path):
    sequence_file = resolve_stride_sequence_file(source_path)
    if sequence_file is None:
        raise FileNotFoundError(f"No stride sequence file found under: {source_path}")

    clip_root = sequence_file.parent
    data = np.load(sequence_file, allow_pickle=True)
    try:
        if "verts" not in data:
            raise KeyError(f"'verts' not found in stride sequence: {sequence_file}")

        verts = _frame_major_stride_array(data["verts"], "verts")
        if verts.ndim != 3 or verts.shape[2] != 3:
            raise ValueError(f"Invalid stride verts shape in {sequence_file}: {verts.shape}")

        n_frames = int(verts.shape[0])
        frame_ids = _frame_major_stride_array(data["frame_id"], "frame_id", n_frames) if "frame_id" in data else np.arange(n_frames)
        cam_t = _frame_major_stride_array(data["cam_t"], "cam_t", n_frames) if "cam_t" in data else None
        box_center = _frame_major_stride_array(data["box_center"], "box_center", n_frames) if "box_center" in data else None
        box_size = _frame_major_stride_array(data["box_size"], "box_size", n_frames) if "box_size" in data else None
        right = _load_stride_right_values(clip_root, n_frames)
    finally:
        data.close()

    order = np.argsort(np.asarray(frame_ids).reshape(-1), kind="stable")
    for index in order.tolist():
        frame_id = int(np.asarray(frame_ids[index]).reshape(()))
        verts_frame = np.asarray(verts[index], dtype=np.float32)
        cam_t_frame = np.zeros((3,), dtype=np.float32)
        if cam_t is not None:
            cam_t_frame = np.asarray(cam_t[index], dtype=np.float32).reshape(3)

        box_center_frame = None
        if box_center is not None:
            box_center_frame = np.asarray(box_center[index], dtype=np.float32).reshape(2)
            if not np.all(np.isfinite(box_center_frame)):
                box_center_frame = None

        box_size_frame = None
        if box_size is not None:
            box_size_frame = float(np.asarray(box_size[index], dtype=np.float32).reshape(()))
            if not np.isfinite(box_size_frame):
                box_size_frame = None

        record = {
            "verts": verts_frame,
            "cam_t": cam_t_frame,
            "verts_world": verts_frame + cam_t_frame.reshape(1, 3),
            "right": int(right[index]),
            "box_center": box_center_frame,
            "box_size": box_size_frame,
            "frame_id": frame_id,
            "track_id": 0,
            "path": f"{sequence_file}#frame_{frame_id:06d}",
        }
        yield frame_id, [record]


def iter_model_frame_records(source_path, pattern="*.npy"):
    if is_stride_source(source_path):
        yield from iter_stride_frame_records(source_path)
        return

    discovered = discover_frame_files(source_path, frame_dirs_glob="frame_*", file_glob=pattern)
    current_frame = None
    current_records = []

    for frame_idx, file_path in discovered:
        try:
            record = load_wilor_record(file_path)
        except Exception as exc:
            print(f"[npy_io] Failed to load {file_path}: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue

        if current_frame is None:
            current_frame = frame_idx
        elif frame_idx != current_frame:
            if current_records:
                yield current_frame, current_records
            current_frame = frame_idx
            current_records = []

        current_records.append(record)

    if current_records:
        yield current_frame, current_records


def parse_obj_faces(obj_path):
    faces = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            toks = line.strip().split()[1:]
            idx = [int(tok.split("/")[0]) - 1 for tok in toks]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) > 3:
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    if not faces:
        return None
    return np.asarray(faces, dtype=np.int32)


def load_template_faces_from_root(root_dir):
    """
    Load MANO topology from any existing obj in the output tree.
    Works even when per-frame visualization uses npy-only vertex data.
    """
    obj_files = sorted(glob.glob(os.path.join(root_dir, "frame_*", "*.obj")))
    if not obj_files:
        return None
    try:
        return parse_obj_faces(obj_files[0])
    except Exception:
        return None

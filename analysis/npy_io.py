import glob
import os
import re
from pathlib import Path

import numpy as np


def list_frame_folders(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "frame_*")))


def load_wilor_record(npy_path):
    """
    Parse one WiLoR inference record saved as:
      np.save(path, {"verts": verts, "cam_t": cam_t, "right": is_right, ...})
    """
    arr = np.load(npy_path, allow_pickle=True)

    if not (isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ()):
        raise ValueError(f"Unsupported npy format: {npy_path}")

    item = arr.item()
    if not isinstance(item, dict) or "verts" not in item:
        raise ValueError(f"Missing dict/verts in: {npy_path}")

    verts = np.asarray(item["verts"], dtype=np.float32)
    cam_t = np.asarray(item.get("cam_t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    right = int(float(np.asarray(item.get("right", -1)).item()))
    box_center = item.get("box_center", None)
    box_size = item.get("box_size", None)

    if box_center is not None:
        box_center = np.asarray(box_center, dtype=np.float32).reshape(2)
    if box_size is not None:
        box_size = float(np.asarray(box_size, dtype=np.float32).reshape(()))

    return {
        "verts": verts,
        "cam_t": cam_t,
        "verts_world": verts + cam_t.reshape(1, 3),
        "right": right,
        "box_center": box_center,
        "box_size": box_size,
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
        except Exception:
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

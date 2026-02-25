import glob
import os

import numpy as np


def list_frame_folders(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "frame_*")))


def load_wilor_record(npy_path):
    """
    Parse one WiLoR inference record saved as:
      np.save(path, {"verts": verts, "cam_t": cam_t, "right": is_right})
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

    return {
        "verts": verts,
        "cam_t": cam_t,
        "verts_world": verts + cam_t.reshape(1, 3),
        "right": right,
        "path": npy_path,
    }


def load_frame_records(frame_dir, pattern="*_verts.npy"):
    records = []
    for npy_path in sorted(glob.glob(os.path.join(frame_dir, pattern))):
        try:
            records.append(load_wilor_record(npy_path))
        except Exception:
            continue
    return records


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

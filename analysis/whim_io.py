from pathlib import Path

import numpy as np

from _path_setup import PROJECT_ROOT  # ensures root imports work


DEFAULT_WHIM_DATA_ROOT = PROJECT_ROOT.parent / "data" / "whim"
DEFAULT_WHIM_TEST_VIDEO_DIR = DEFAULT_WHIM_DATA_ROOT / "test" / "anno" / "NXRHcCScubA"


def normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return Path(text)


def to_numpy(value, dtype=np.float32):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        arr = value.numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def to_scalar_int(value, default=-1):
    if value is None:
        return int(default)
    arr = to_numpy(value, dtype=np.float32)
    if arr.size == 0:
        return int(default)
    return int(round(float(arr.reshape(-1)[0])))


def load_whim_frame_items(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    if not (isinstance(arr, np.ndarray) and arr.dtype == object):
        raise ValueError(f"Unsupported WHIM npy format: {npy_path}")

    items = arr.tolist()
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise ValueError(f"Unsupported WHIM object payload in: {npy_path}")

    parsed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parsed.append(
            {
                "bbox": to_numpy(item.get("bbox"), dtype=np.float32),
                "vertices": to_numpy(item.get("vertices"), dtype=np.float32),
                "points_2d": to_numpy(item.get("joints_2d"), dtype=np.float32),
                "joints_3d": to_numpy(item.get("joints_3d"), dtype=np.float32),
                "side": to_scalar_int(item.get("side"), default=-1),
                "trans": to_numpy(item.get("trans"), dtype=np.float32),
                "K": to_numpy(item.get("K"), dtype=np.float32),
                "mano": item.get("mano"),
            }
        )
    return parsed


def iter_whim_npy_paths(video_dir, frame_step=1, max_frames=None):
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"WHIM video directory does not exist: {video_dir}")

    npy_paths = sorted(video_dir.glob("*.npy"))
    if not npy_paths:
        raise RuntimeError(f"No .npy files found under: {video_dir}")

    step = max(1, int(frame_step))
    if step > 1:
        npy_paths = npy_paths[::step]
    if max_frames is not None:
        npy_paths = npy_paths[: max(1, int(max_frames))]
    return npy_paths

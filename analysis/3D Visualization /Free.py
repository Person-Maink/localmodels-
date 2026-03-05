import pickle

import numpy as np
from vedo import Lines, Mesh, Points
from vedo.applications import AnimationPlayer

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from npy_io import list_frame_folders, load_frame_records


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
]


def _normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _resolve_source():
    return _normalize_optional_path(
        getattr(CONFIG, "FREE_SOURCE", None)
        or getattr(CONFIG, "MODEL_COMP", None)
        or getattr(CONFIG, "MODEL_ROOT", None)
        or getattr(CONFIG, "MEDIAPIPE_ROOT", None)
    )


def _is_mediapipe_source(path_text):
    return str(path_text).lower().endswith(".csv")


def _parse_hand_id(value):
    text = str(value).strip().lower()
    if text in {"right", "r", "1", "1.0"}:
        return 1
    if text in {"left", "l", "0", "0.0"}:
        return 0
    try:
        as_int = int(float(text))
        if as_int in {0, 1}:
            return as_int
    except ValueError:
        pass
    return -1


def _color_for_hand(hand_id):
    if hand_id == 1:
        return "crimson"
    if hand_id == 0:
        return "royalblue"
    return "gray"


def _load_faces_from_mano(mano_right_path):
    alias_map = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
        "nan": float("nan"),
        "inf": float("inf"),
    }
    for name, value in alias_map.items():
        if not hasattr(np, name):
            setattr(np, name, value)

    with open(mano_right_path, "rb") as f:
        mano = pickle.load(f, encoding="latin1")

    faces = np.asarray(mano["f"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Invalid MANO face topology in {mano_right_path}: shape={faces.shape}")
    return faces


def _build_mesh_actor(verts_world, faces_right, right, apply_x180=True, mesh_alpha=0.5):
    verts = verts_world.copy()
    if apply_x180:
        verts[:, 1] *= -1.0
        verts[:, 2] *= -1.0

    faces = faces_right if right == 1 else faces_right[:, [0, 2, 1]]
    actor = Mesh([verts, faces])
    actor.color(_color_for_hand(right))
    actor.alpha(mesh_alpha)
    return actor


def _build_mediapipe_actor(points, hand_id):
    color = _color_for_hand(hand_id)
    joints = Points(points, r=8, c=color)

    valid_connections = [(i, j) for i, j in HAND_CONNECTIONS if i < len(points) and j < len(points)]
    if not valid_connections:
        return joints

    lines = Lines(
        points[[i for i, _ in valid_connections]],
        points[[j for _, j in valid_connections]],
        lw=2,
        c=color,
    )
    return joints + lines


def _load_frames_from_model(root_dir, apply_x180=True, mesh_alpha=0.5):
    faces_right = _load_faces_from_mano(CONFIG.MANO_RIGHT_PATH)

    frames = []
    loaded_records = 0

    for folder in list_frame_folders(root_dir):
        records = load_frame_records(folder, pattern="*.npy")
        if not records:
            continue

        actors = []
        for rec in records:
            actors.append(_build_mesh_actor(rec["verts_world"], faces_right, rec["right"], apply_x180, mesh_alpha))
            loaded_records += 1

        if not actors:
            continue

        frame_actor = actors[0] if len(actors) == 1 else sum(actors[1:], actors[0])
        frames.append(frame_actor)

    return frames, loaded_records


def _load_frames_from_mediapipe(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "frame_id" not in df.columns:
        raise ValueError(f"MediaPipe CSV missing required column 'frame_id': {csv_path}")

    hand_col = "hand_id" if "hand_id" in df.columns else None
    if hand_col is None:
        raise ValueError(f"MediaPipe CSV missing required column 'hand_id': {csv_path}")

    frames = []
    loaded_records = 0

    for frame_id in sorted(df.frame_id.unique()):
        frame_df = df[df.frame_id == frame_id]
        frame_actors = []

        for hid, hand_df in frame_df.groupby(hand_col):
            hand_df = hand_df.sort_values("joint_id")
            pts = hand_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            if len(pts) == 0:
                continue
            frame_actors.append(_build_mediapipe_actor(pts, _parse_hand_id(hid)))
            loaded_records += 1

        if frame_actors:
            frame_actor = frame_actors[0] if len(frame_actors) == 1 else sum(frame_actors[1:], frame_actors[0])
            frames.append(frame_actor)

    return frames, loaded_records


def main():
    source_path = _resolve_source()
    if source_path is None:
        raise RuntimeError(
            "No source configured. Set FREE_SOURCE (or MODEL_COMP/MODEL_ROOT/MEDIAPIPE_ROOT) in FILENAME.py."
        )

    if _is_mediapipe_source(source_path):
        frames, n_records = _load_frames_from_mediapipe(source_path)
        print(f"Loaded {len(frames)} frames from MediaPipe CSV ({n_records} hand records)")
    else:
        frames, n_records = _load_frames_from_model(source_path, apply_x180=True, mesh_alpha=0.5)
        print(f"Loaded {len(frames)} frames from model npy records ({n_records} hand records)")

    if not frames:
        raise RuntimeError(f"No valid frames found for source: {source_path}")

    actor = frames[0].clone()

    def update_scene(i: int):
        nonlocal actor
        plt.remove(actor)
        actor = frames[i].clone()
        plt.add(actor)
        plt.render()

    plt = AnimationPlayer(update_scene, irange=[0, len(frames) - 1])
    plt += actor

    plt.set_frame(0)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

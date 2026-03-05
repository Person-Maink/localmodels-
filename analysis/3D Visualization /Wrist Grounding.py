import pickle

import numpy as np
from vedo import Lines, Mesh, Points
from vedo.applications import AnimationPlayer

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from FILENAME import HAND_IDX as DEFAULT_HAND_IDX, WRIST_JOINT_IDX
from npy_io import list_frame_folders, load_frame_records

np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.unicode = str
np.nan = float("nan")
np.inf = float("inf")

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
        getattr(CONFIG, "WRIST_GROUNDING_SOURCE", None)
        or getattr(CONFIG, "MODEL_ROOT", None)
        or getattr(CONFIG, "MODEL_COMP", None)
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


def _load_frames_from_model(root_dir):
    with open(CONFIG.MANO_RIGHT_PATH, "rb") as f:
        mano = pickle.load(f, encoding="latin1")

    j_reg = mano["J_regressor"]
    faces_right = np.asarray(mano["f"], dtype=np.int32)

    frames = []

    for folder in list_frame_folders(root_dir):
        records = load_frame_records(folder, pattern="*.npy")
        if not records:
            continue

        frame_hands = []
        for rec in records:
            verts_world = rec["verts_world"]
            if verts_world.shape[0] != j_reg.shape[1]:
                continue

            joints = j_reg @ verts_world
            wrist = joints[WRIST_JOINT_IDX]
            verts_centered = verts_world - wrist

            frame_hands.append(
                {
                    "mode": "model",
                    "verts": verts_centered,
                    "is_right": int(rec["right"] == 1),
                    "faces_right": faces_right,
                }
            )

        if frame_hands:
            frames.append(frame_hands)

    return frames


def _load_frames_from_mediapipe(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "frame_id" not in df.columns:
        raise ValueError(f"MediaPipe CSV missing required column 'frame_id': {csv_path}")
    if "hand_id" not in df.columns:
        raise ValueError(f"MediaPipe CSV missing required column 'hand_id': {csv_path}")

    frames = []
    for frame_id in sorted(df.frame_id.unique()):
        frame_df = df[df.frame_id == frame_id]
        frame_hands = []

        for hid, hand_df in frame_df.groupby("hand_id"):
            hand_df = hand_df.sort_values("joint_id")
            pts = hand_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            if len(pts) == 0:
                continue

            wrist = pts[0]
            pts_centered = pts - wrist

            frame_hands.append(
                {
                    "mode": "mediapipe",
                    "points": pts_centered,
                    "is_right": _parse_hand_id(hid),
                }
            )

        if frame_hands:
            frames.append(frame_hands)

    return frames


def _build_actor(frame_hands, hand_idx):
    selected = [h for h in frame_hands if h["is_right"] == hand_idx]
    if not selected:
        return None

    actors = []
    for hand in selected:
        if hand["mode"] == "model":
            faces_right = hand["faces_right"]
            faces = faces_right if hand["is_right"] == 1 else faces_right[:, [0, 2, 1]]
            actor = Mesh([hand["verts"], faces])
            actor.color(_color_for_hand(hand["is_right"]))
            actor.alpha(0.55)
            actors.append(actor)
            continue

        pts = hand["points"]
        joints = Points(pts, r=8, c=_color_for_hand(hand["is_right"]))

        valid_connections = [(i, j) for i, j in HAND_CONNECTIONS if i < len(pts) and j < len(pts)]
        if valid_connections:
            lines = Lines(
                pts[[i for i, _ in valid_connections]],
                pts[[j for _, j in valid_connections]],
                lw=2,
                c=_color_for_hand(hand["is_right"]),
            )
            actors.append(joints + lines)
        else:
            actors.append(joints)

    if len(actors) == 1:
        return actors[0]
    return sum(actors[1:], actors[0])


def main():
    source_path = _resolve_source()
    if source_path is None:
        raise RuntimeError(
            "No source configured. Set WRIST_GROUNDING_SOURCE (or MODEL_ROOT/MODEL_COMP/MEDIAPIPE_ROOT) in FILENAME.py."
        )

    if _is_mediapipe_source(source_path):
        frames = _load_frames_from_mediapipe(source_path)
        print(f"Loaded {len(frames)} frames from MediaPipe CSV")
    else:
        frames = _load_frames_from_model(source_path)
        print(f"Loaded {len(frames)} frames from model npy records")

    if not frames:
        raise RuntimeError(f"No valid records found for source: {source_path}")

    hand_idx = int(DEFAULT_HAND_IDX)
    actor = _build_actor(frames[0], hand_idx)

    def update_scene(i: int):
        nonlocal actor
        if actor is not None:
            plt.remove(actor)

        actor = _build_actor(frames[i], hand_idx)
        if actor is not None:
            plt.add(actor)

        plt.render()

    def toggle_hand(*args, **kwargs):
        nonlocal hand_idx, actor
        hand_idx = 1 - hand_idx

        if actor is not None:
            plt.remove(actor)

        actor = _build_actor(frames[plt.frame], hand_idx)
        if actor is not None:
            plt.add(actor)

        plt.render()

    plt = AnimationPlayer(update_scene, irange=[0, len(frames) - 1])
    if actor is not None:
        plt += actor

    plt.add_button(
        toggle_hand,
        pos=(0.02, 0.92),
        states=["Left", "Right"],
        c=["white", "white"],
        bc=["#444444", "#444444"],
    )

    plt.set_frame(0)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

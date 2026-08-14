import argparse
from pathlib import Path

import numpy as np

import FILENAME as CONFIG
from mano_pickle import load_mano_pickle
from whim_io import (
    DEFAULT_WHIM_TEST_VIDEO_DIR,
    iter_whim_npy_paths,
    load_whim_frame_items,
    normalize_optional_path,
)

try:
    from vedo import Mesh
    from vedo.applications import AnimationPlayer
except ModuleNotFoundError:
    Mesh = AnimationPlayer = None


RIGHT_COLOR = "crimson"
LEFT_COLOR = "royalblue"
UNKNOWN_COLOR = "gray"
DEFAULT_VIDEO_DIR = DEFAULT_WHIM_TEST_VIDEO_DIR
DEFAULT_MESH_ALPHA = 0.5


def _color_for_hand(side):
    if side == 1:
        return RIGHT_COLOR
    if side == 0:
        return LEFT_COLOR
    return UNKNOWN_COLOR

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

    mano = load_mano_pickle(mano_right_path)
    faces = np.asarray(mano["f"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Invalid MANO face topology in {mano_right_path}: shape={faces.shape}")
    return faces

def _build_mesh_actor(verts_world, faces_right, side, apply_x180=True, mesh_alpha=DEFAULT_MESH_ALPHA):
    verts = verts_world.copy()
    if apply_x180:
        verts[:, 1] *= -1.0
        verts[:, 2] *= -1.0

    faces = faces_right if side == 1 else faces_right[:, [0, 2, 1]]
    actor = Mesh([verts, faces])
    actor.color(_color_for_hand(side))
    actor.alpha(mesh_alpha)
    return actor


def _load_frames_from_whim(video_dir, apply_x180=True, mesh_alpha=DEFAULT_MESH_ALPHA):
    video_dir = Path(video_dir)
    faces_right = _load_faces_from_mano(CONFIG.MANO_RIGHT_PATH)
    npy_paths = iter_whim_npy_paths(video_dir)

    frames = []
    loaded_records = 0

    for npy_path in npy_paths:
        hands = load_whim_frame_items(npy_path)
        if not hands:
            continue

        actors = []
        for hand in hands:
            verts = hand["vertices"]
            trans = hand["trans"]
            if verts is None or trans is None:
                continue
            verts_world = np.asarray(verts, dtype=np.float32).reshape(-1, 3) + np.asarray(trans, dtype=np.float32).reshape(1, 3)
            actors.append(_build_mesh_actor(verts_world, faces_right, hand["side"], apply_x180, mesh_alpha))
            loaded_records += 1

        if not actors:
            continue

        frame_actor = actors[0] if len(actors) == 1 else sum(actors[1:], actors[0])
        frames.append(frame_actor)

    return frames, loaded_records


def main():
    if AnimationPlayer is None or Mesh is None:
        raise ModuleNotFoundError("vedo is required for visualization. Install it in this environment first.")

    parser = argparse.ArgumentParser(description="Visualize WHIM hand meshes in free view.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(DEFAULT_VIDEO_DIR),
        help="WHIM per-video annotation directory containing *.npy files.",
    )
    args = parser.parse_args()

    video_dir = normalize_optional_path(args.video_dir)
    if video_dir is None:
        raise ValueError("No WHIM video directory configured.")

    frames, n_records = _load_frames_from_whim(video_dir, apply_x180=True, mesh_alpha=DEFAULT_MESH_ALPHA)
    print(f"Loaded {len(frames)} frames from WHIM npy records ({n_records} hand records)")

    if not frames:
        raise RuntimeError(f"No valid WHIM frames found under: {video_dir}")

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

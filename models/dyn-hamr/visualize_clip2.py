from pathlib import Path

from vedo import Mesh
from vedo.applications import AnimationPlayer


OUTPUTS_ROOT = Path(
    "/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/dynhamr/logs/video-custom"
)
CLIP_RUN_NAME = "clip_2-all-shot-0-0--1"
CLIP_PREFIX = "clip_2"
PREFERRED_PHASES = ("smooth_fit", "prior")

LEFT_COLOR = "royalblue"
RIGHT_COLOR = "crimson"
MESH_ALPHA = 0.55


def _resolve_run_root():
    matches = sorted(OUTPUTS_ROOT.glob(f"*/{CLIP_RUN_NAME}"))
    if not matches:
        raise FileNotFoundError(f"No Dyn-HaMR runs found for {CLIP_RUN_NAME} under {OUTPUTS_ROOT}")
    return matches[-1]


def _resolve_mesh_root(run_root):
    for phase in PREFERRED_PHASES:
        candidates = sorted((run_root / phase).glob(f"{CLIP_PREFIX}_*_meshes"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"No mesh export folders found in {run_root}")


def _group_frame_meshes(mesh_root):
    grouped = {}
    for obj_path in sorted(mesh_root.glob("*.obj")):
        try:
            frame_text, hand_text = obj_path.stem.rsplit("_", 1)
            frame_idx = int(frame_text)
            hand_idx = int(hand_text)
        except ValueError:
            continue
        grouped.setdefault(frame_idx, []).append((hand_idx, obj_path))

    if not grouped:
        raise RuntimeError(f"No OBJ frames found in {mesh_root}")

    return [(frame_idx, sorted(entries, key=lambda item: item[0])) for frame_idx, entries in sorted(grouped.items())]


def _color_for_hand(hand_idx):
    if hand_idx == 1:
        return RIGHT_COLOR
    if hand_idx == 0:
        return LEFT_COLOR
    return "gray"


def _build_frame_actor(frame_entries):
    actors = []
    for hand_idx, obj_path in frame_entries:
        actor = Mesh(str(obj_path))
        actor.color(_color_for_hand(hand_idx))
        actor.alpha(MESH_ALPHA)
        actors.append(actor)

    if not actors:
        return None
    if len(actors) == 1:
        return actors[0]
    return sum(actors[1:], actors[0])


def main():
    run_root = _resolve_run_root()
    mesh_root = _resolve_mesh_root(run_root)
    frames = _group_frame_meshes(mesh_root)

    print(f"Using Dyn-HaMR run: {run_root}")
    print(f"Using mesh folder: {mesh_root}")
    print(f"Loaded {len(frames)} frames")

    actor = _build_frame_actor(frames[0][1])

    def update_scene(i):
        nonlocal actor
        if actor is not None:
            plt.remove(actor)

        actor = _build_frame_actor(frames[i][1])
        if actor is not None:
            plt.add(actor)

        plt.render()

    plt = AnimationPlayer(update_scene, irange=[0, len(frames) - 1])
    if actor is not None:
        plt += actor

    plt.show(bg="white", axes=1, viewup="z")
    plt.close()


if __name__ == "__main__":
    main()

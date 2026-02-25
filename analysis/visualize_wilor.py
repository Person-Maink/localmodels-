import numpy as np
from vedo import Mesh
from vedo.applications import AnimationPlayer

from visualizing_files import WILOR_ROOT
from wilor_npy_io import list_frame_folders, load_frame_records, load_template_faces_from_root


ROOT_DIR = WILOR_ROOT
APPLY_X180 = True
MESH_ALPHA = 0.5


def build_mesh_actor(verts_world, faces_right, right):
    verts = verts_world.copy()
    if APPLY_X180:
        verts[:, 1] *= -1.0
        verts[:, 2] *= -1.0

    faces = faces_right if right == 1 else faces_right[:, [0, 2, 1]]
    actor = Mesh([verts, faces])
    if right == 1:
        actor.color("crimson")
    elif right == 0:
        actor.color("royalblue")
    else:
        actor.color("gray")
    actor.alpha(MESH_ALPHA)
    return actor


def load_frames_from_npy(root_dir):
    faces_right = load_template_faces_from_root(root_dir)
    if faces_right is None:
        raise RuntimeError(
            f"Could not find mesh faces from obj template under {root_dir}. "
            "Need at least one .obj file to recover topology."
        )

    frames = []
    loaded_records = 0

    for folder in list_frame_folders(root_dir):
        records = load_frame_records(folder)
        if not records:
            continue

        actors = []
        for rec in records:
            actors.append(build_mesh_actor(rec["verts_world"], faces_right, rec["right"]))
            loaded_records += 1

        if not actors:
            continue

        frame_actor = actors[0] if len(actors) == 1 else sum(actors[1:], actors[0])
        frames.append(frame_actor)

    return frames, loaded_records


frames, n_records = load_frames_from_npy(ROOT_DIR)
print(f"Loaded {len(frames)} frames from npy records ({n_records} hand records)")

if not frames:
    raise RuntimeError(f"No valid *_verts.npy records found under {ROOT_DIR}")


actor = frames[0].clone(deep=True)


def update_scene(i: int):
    global actor
    plt.remove(actor)
    actor = frames[i].clone(deep=True)
    plt.add(actor)
    plt.render()


plt = AnimationPlayer(update_scene, irange=[0, len(frames) - 1])
plt += actor

plt.set_frame(0)
plt.show()
plt.close()

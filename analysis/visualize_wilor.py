import glob
import os

import numpy as np
from vedo import Mesh, Points, merge
from vedo.applications import AnimationPlayer

from visualizing_files import WILOR_ROOT


ROOT_DIR = WILOR_ROOT
APPLY_X180 = True  # Match the orientation used in obj export path.


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
                # Fan triangulation for polygon faces.
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    if not faces:
        return None
    return np.asarray(faces, dtype=np.int32)


def try_get_template_faces(root_dir):
    obj_files = sorted(glob.glob(os.path.join(root_dir, "frame_*", "*.obj")))
    if not obj_files:
        return None
    try:
        return parse_obj_faces(obj_files[0])
    except Exception:
        return None


def load_record_from_npy(npy_path):
    arr = np.load(npy_path, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        item = arr.item()
        if isinstance(item, dict):
            if "verts" not in item:
                raise KeyError(f"Missing 'verts' in {npy_path}")
            verts = np.asarray(item["verts"], dtype=np.float32)
            cam_t = np.asarray(item.get("cam_t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            right = int(float(np.asarray(item.get("right", -1)).item()))
            return verts, cam_t, right

    raise ValueError(f"Unsupported npy format for {npy_path}")


def make_actor(verts, faces, right):
    if APPLY_X180:
        verts = verts.copy()
        verts[:, 1] *= -1.0
        verts[:, 2] *= -1.0

    if faces is not None:
        actor = Mesh([verts, faces])
    else:
        actor = Points(verts, r=3)

    if right == 1:
        actor.color("crimson")
    elif right == 0:
        actor.color("royalblue")
    else:
        actor.color("gray")

    if hasattr(actor, "alpha"):
        actor.alpha(0.5)
    return actor


def load_frames_from_npy(root_dir):
    frame_folders = sorted(glob.glob(os.path.join(root_dir, "frame_*")))
    faces = try_get_template_faces(root_dir)

    frames = []
    loaded_records = 0

    for folder in frame_folders:
        npy_files = sorted(glob.glob(os.path.join(folder, "*_verts.npy")))
        if not npy_files:
            continue

        frame_actors = []
        for npy_path in npy_files:
            try:
                verts, cam_t, right = load_record_from_npy(npy_path)
            except Exception:
                continue

            # Reconstruct the translated mesh space from inference output.
            verts_world = verts + cam_t.reshape(1, 3)
            frame_actors.append(make_actor(verts_world, faces, right))
            loaded_records += 1

        if not frame_actors:
            continue

        merged = merge(frame_actors) if len(frame_actors) > 1 else frame_actors[0]
        frames.append(merged)

    return frames, loaded_records, faces is not None


frames, n_records, used_faces = load_frames_from_npy(ROOT_DIR)
print(f"Loaded {len(frames)} frames from npy records ({n_records} hand records)")
print(f"Mesh mode: {'on' if used_faces else 'off (points fallback)'}")

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

import pickle

import numpy as np
from vedo import Mesh
from vedo.applications import AnimationPlayer

from _path_setup import PROJECT_ROOT  # ensures root imports work
from FILENAME import *
from wilor_npy_io import list_frame_folders, load_frame_records

np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.unicode = str
np.nan = float("nan")
np.inf = float("inf")

# =========================
# CONFIG
# =========================
ROOT_DIR = WILOR_ROOT

WRIST_JOINT_IDX = 0
HAND_IDX = 1  # 1=right, 0=left

# =========================
# LOAD MANO REGRESSOR
# =========================

with open(MANO_RIGHT_PATH, "rb") as f:
    mano = pickle.load(f, encoding="latin1")

J_reg = mano["J_regressor"]
FACES_RIGHT = np.asarray(mano["f"], dtype=np.int32)

# =========================
# LOAD + WRIST CENTER HANDS (NPY)
# =========================

frames = []

for folder in list_frame_folders(ROOT_DIR):
    records = load_frame_records(folder, pattern="*.npy")
    if not records:
        continue

    frame_hands = []

    for rec in records:
        V = rec["verts_world"]
        assert V.shape[0] == J_reg.shape[1]

        J = J_reg @ V
        wrist = J[WRIST_JOINT_IDX]
        V_centered = V - wrist

        frame_hands.append({
            "verts": V_centered,
            "is_right": int(rec["right"] == 1),
        })

    if frame_hands:
        frames.append(frame_hands)

print(f"Loaded {len(frames)} frames from npy")

if not frames:
    raise RuntimeError(f"No valid WiLoR npy records found under {ROOT_DIR}")


# =========================
# VISUALIZATION
# =========================
def build_actor(frame_hands):
    selected = [h for h in frame_hands if h["is_right"] == HAND_IDX]
    if not selected:
        return None

    actors = []
    for h in selected:
        faces = FACES_RIGHT if h["is_right"] == 1 else FACES_RIGHT[:, [0, 2, 1]]
        a = Mesh([h["verts"], faces])
        a.color("crimson" if h["is_right"] == 1 else "royalblue")
        a.alpha(0.55)
        actors.append(a)

    if len(actors) == 1:
        return actors[0]
    return sum(actors[1:], actors[0])


actor = build_actor(frames[0])


def update_scene(i: int):
    global actor
    if actor is not None:
        plt.remove(actor)

    actor = build_actor(frames[i])
    if actor is not None:
        plt.add(actor)

    plt.render()


def toggle_hand(*args, **kwargs):
    global HAND_IDX, actor
    HAND_IDX = 1 - HAND_IDX

    if actor is not None:
        plt.remove(actor)

    actor = build_actor(frames[plt.frame])
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

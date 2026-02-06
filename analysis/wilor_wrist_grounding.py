import glob
import os
import pickle
from vedo import load, merge
from vedo.applications import AnimationPlayer
from visualizing_files import *

import numpy as np
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

# Root directory containing frame_XXXXX folders
ROOT_DIR = WILOR_ROOT

WRIST_JOINT_IDX = 0   # MANO wrist index
HAND_IDX = 1  # for right, 0 for left 
# HAND_IDX = 1  # for right, 0 for left 

# =========================
# LOAD MANO REGRESSOR
# =========================

with open(MANO_RIGHT_PATH, "rb") as f:
    mano = pickle.load(f, encoding="latin1")

J_reg = mano["J_regressor"]   # (21, 778), sparse OK

# =========================
# LOAD + CENTER HANDS
# =========================

frame_folders = sorted(glob.glob(os.path.join(ROOT_DIR, "frame_*")))
frames = []   # frames[i] = [hand0_mesh, hand1_mesh]

for folder in frame_folders:
    all_objs = sorted(glob.glob(os.path.join(folder, "*.obj")))
    if not all_objs:
        continue

    frame_hands = []

    for fpath in all_objs:
        # infer handedness from filename
        is_right = fpath.endswith("_1.0.obj")
        
        m = load(fpath)
        V = m.points

        assert V.shape[0] == J_reg.shape[1]
        J = J_reg @ V

        wrist = J[WRIST_JOINT_IDX]
        m.shift(-wrist)

        frame_hands.append({
            "mesh": m,
            "is_right": is_right
        })

    frames.append(frame_hands)

print(f"Loaded {len(frames)} frames with 2 hands each")

# =========================
# VISUALIZATION
# =========================

def build_actor(frame_hands):
    selected = [
        h["mesh"] for h in frame_hands
        if h["is_right"] == HAND_IDX 
    ]

    if not selected:
        return None

    return merge(selected) if len(selected) > 1 else selected[0]

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
    HAND_IDX = not HAND_IDX 

    plt.remove(actor)
    actor = build_actor(frames[plt.frame])
    if actor is not None:
        plt.add(actor)
    plt.render()

plt = AnimationPlayer(update_scene, irange=[0, len(frames) - 1])
plt += actor

plt.add_button(
    toggle_hand,
    pos=(0.02, 0.92),          # top-left
    states=["Hand 0", "Hand 1"],
    c=["white", "white"],
    bc=["#444444", "#444444"],
)

plt.set_frame(0)
plt.show()
plt.close()
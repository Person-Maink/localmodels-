import os
import glob
import pandas as pd
from vedo import Points, Lines, load, merge
from vedo.applications import AnimationPlayer
import numpy as np
from copy import copy

def center_points(pts):
    if len(pts) == 0:
        return pts
    return pts - pts.mean(axis=0)

def center_mesh(mesh):
    center = mesh.points.mean(axis=0)
    mesh.shift(-center)
    return mesh

# =========================
# CONFIG
# =========================

CSV_PATH = "/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/mediapipe/keypoints/163 (2) FU_keypoints.csv"
WILOR_ROOT = "/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/wilor/120-2_clip_5/meshes"
SHOW_HAND= "Left" # or "Right"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
]

# =========================
# LOAD MEDIAPIPE KEYPOINTS
# =========================

df = pd.read_csv(CSV_PATH)
mp_frames = sorted(df.frame_id.unique())

actors_mp = []

for fid in mp_frames:
    frame_df = df[df.frame_id == fid]

    frame_actors = []

    for hid, hdf in frame_df.groupby("hand_id"):
        if hid != SHOW_HAND:
            continue

        hdf = hdf.sort_values("joint_id")
        pts = hdf[["x", "y", "z"]].to_numpy()
        pts = center_points(pts)


        joints = Points(pts, r=8)

        lines = Lines(
            pts[[i for i, _ in HAND_CONNECTIONS]],
            pts[[j for _, j in HAND_CONNECTIONS]],
            lw=2,
        )

        frame_actors.append(joints + lines)

    # actors_mp.append(sum(frame_actors[1:], frame_actors[0]))
    if not frame_actors:
        continue
    actors_mp.append(sum(frame_actors[1:], frame_actors[0]))

print(f"Loaded {len(actors_mp)} MediaPipe frames")

# =========================
# LOAD WILOR MESHES
# =========================

frame_folders = sorted(glob.glob(os.path.join(WILOR_ROOT, "frame_*")))
actors_wilor = []

for folder in frame_folders:
    obj_files = sorted(glob.glob(os.path.join(folder, "*.obj")))

    if SHOW_HAND == "Left":
        obj_files = [f for f in obj_files if "_0" in f.split("/")[-1].lower()]
    elif SHOW_HAND == "Right":
        obj_files = [f for f in obj_files if "_1" in f.split("/")[-1].lower()]

    if not obj_files:
        continue
    print(obj_files[0].split('/')[-1])

    meshes = [load(f) for f in obj_files]
    merged = merge(meshes) if len(meshes) > 1 else meshes[0]
    merged = center_mesh(merged)

    merged.color("lightgray").alpha(0.4)

    actors_wilor.append(merged)

print(f"Loaded {len(actors_wilor)} Wilor frames")

# =========================
# ALIGN + COMBINE
# =========================

num_frames = min(len(actors_mp), len(actors_wilor))

combined_actors = []
for i in range(num_frames):
    # combined = actors_mp[i].clone() + actors_wilor[i].clone(deep=True)
    combined = actors_mp[i] + actors_wilor[i]
    combined_actors.append(combined)

print(f"Combined {len(combined_actors)} frames")

# =========================
# ANIMATION
# =========================

# actor = combined_actors[0].clone()
actor = copy(combined_actors)

def update_scene(i: int):
    global actor
    plt.remove(actor)
    actor = combined_actors[i].clone()
    plt.add(actor)
    plt.render()

plt = AnimationPlayer(update_scene, irange=[0, len(combined_actors) - 1])
plt += actor
plt.set_frame(0)
plt.show()
plt.close()

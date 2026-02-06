import pandas as pd
import numpy as np
from vedo import Points, Lines
from vedo.applications import AnimationPlayer
from visualizing_files import *

CSV_PATH = MEDIAPIPE_ROOT

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

df = pd.read_csv(CSV_PATH)
frames = sorted(df.frame_id.unique())

actors = []

for fid in frames:
    frame_df = df[df.frame_id == fid]

    frame_actors = []

    for hid, hdf in frame_df.groupby("hand_id"):
        hdf = hdf.sort_values("joint_id")
        pts = hdf[["x", "y", "z"]].to_numpy()

        joints = Points(pts, r=8)

        lines = Lines(
            pts[[i for i, _ in HAND_CONNECTIONS]],
            pts[[j for _, j in HAND_CONNECTIONS]],
            lw=2,
        )

        frame_actors.append(joints + lines)

    actors.append(sum(frame_actors[1:], frame_actors[0]))


actor = actors[0].clone()

def update_scene(i: int):
    global actor
    plt.remove(actor)
    actor = actors[i].clone()
    plt.add(actor)
    plt.render()

plt = AnimationPlayer(update_scene, irange=[0, len(actors) - 1])
plt += actor
plt.set_frame(0)
plt.show()
plt.close()

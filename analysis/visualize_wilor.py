from vedo import load, merge
import glob
import os
from vedo.applications import AnimationPlayer
from visualizing_files import *

root_dir = WILOR_ROOT

# Get sorted list of frame folders
frame_folders = sorted(glob.glob(os.path.join(root_dir, "frame_*")))

# Load meshes for each frame (merged if multiple hands)
frames = []
for folder in frame_folders:
    obj_files = sorted(glob.glob(os.path.join(folder, "*.obj")))
    if not obj_files:
        continue
    meshes = [load(f) for f in obj_files]
    merged = merge(meshes) if len(meshes) > 1 else meshes[0]
    frames.append(merged)

print(f"Loaded {len(frames)} frames")


actor = frames[0].clone(deep=True)

def update_scene(i: int):
    global actor
    plt.remove(actor)
    actor = frames[i].clone(deep=True)
    plt.add(actor)
    plt.render()

plt = AnimationPlayer(update_scene, irange=[0, len(frames)-1])
plt += actor

plt.set_frame(0)
plt.show()
plt.close()
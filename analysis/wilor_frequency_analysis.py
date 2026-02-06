import glob
import os
import pickle
from vedo import load, merge
from vedo.applications import AnimationPlayer
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
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
ROOT_DIR = WILOR_ROOT

WRIST_JOINT_IDX = 0   # MANO wrist index
HAND_IDX = 0  # for right, 0 for left 
# HAND_IDX = 1  # for right, 0 for left 
LOWPASS_CUTOFF = 6.0     # Hz
FILTER_ORDER = 3
FPS = 30

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
# BUILD MEAN HAND TRAJECTORY
# =========================

centroids = []

for frame_hands in frames:
    meshes = [h["mesh"] for h in frame_hands]
    mesh = merge(meshes) if len(meshes) > 1 else meshes[0]
    V = mesh.points
    centroids.append(V.mean(axis=0))

centroids = np.stack(centroids, axis=0)  # (T, 3)
print("Trajectory shape:", centroids.shape)

# =========================
# LOW-PASS BUTTERWORTH FILTER
# =========================

def lowpass_filter(signal, fs, cutoff, order):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal, axis=0)

filtered = lowpass_filter(
    centroids,
    fs=FPS,
    cutoff=LOWPASS_CUTOFF,
    order=FILTER_ORDER
)

# =========================
# FREQUENCY ANALYSIS
# =========================

motion_mag = np.linalg.norm(filtered, axis=1)
motion_mag -= motion_mag.mean()

f, Pxx = welch(
    motion_mag,
    fs=FPS,
    nperseg=min(256, len(motion_mag))
)

dominant_freq = f[np.argmax(Pxx)]
rms_amplitude = np.sqrt(np.mean(motion_mag**2))

# =========================
# RESULTS
# =========================

print(f"Dominant frequency: {dominant_freq:.2f} Hz")
print(f"RMS amplitude: {rms_amplitude:.6f}")

t = np.arange(len(motion_mag)) / FPS

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

# =====================
# 1) Displacement over time
# =====================
axes[0].plot(t, motion_mag, lw=1.5)
axes[0].set_title("Filtered hand displacement over time")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Displacement magnitude")
axes[0].grid(True)

# =====================
# 2) Frequency spectrum
# =====================
axes[1].semilogy(f, Pxx, lw=1.5)
axes[1].axvline(
    dominant_freq,
    color="r",
    ls="--",
    label=f"Dominant freq = {dominant_freq:.2f} Hz"
)
axes[1].set_title("Frequency spectrum of hand motion")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power spectral density")
axes[1].legend()
axes[1].grid(True)

# =====================
# 3) Per-axis displacement
# =====================
labels = ["x", "y", "z"]
for i in range(3):
    axes[2].plot(t, filtered[:, i], label=labels[i])

axes[2].set_title("Filtered hand displacement per axis")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Displacement")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

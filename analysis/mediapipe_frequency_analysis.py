import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from visualizing_files import *

CSV_PATH = MEDIAPIPE_ROOT

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


FPS = 30.0
LOWPASS_CUTOFF = 6.0   # Hz
FILTER_ORDER = 3

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_PATH)

# Expected columns:
# frame_id, hand_id, joint_id, x, y, z

frames = sorted(df.frame_id.unique())

# =========================
# BUILD CENTROID TRAJECTORY
# =========================

centroids = []

for fid in frames:
    frame_df = df[df.frame_id == fid]
    if frame_df.empty:
        continue

    hand_centers = []

    for hid, hdf in frame_df.groupby("hand_id"):
        pts = hdf.sort_values("joint_id")[["x", "y", "z"]].to_numpy()
        hand_centers.append(pts.mean(axis=0))

    if hand_centers:
        centroids.append(np.mean(hand_centers, axis=0))

centroids = np.stack(centroids, axis=0)  # (T, 3)

print("Trajectory shape:", centroids.shape)

# =========================
# LOW-PASS FILTER
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

print(f"Dominant frequency: {dominant_freq:.2f} Hz")
print(f"RMS amplitude: {rms_amplitude:.6f}")

# =========================
# PLOTS (ALL TOGETHER)
# =========================

t = np.arange(len(motion_mag)) / FPS

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 1) Displacement over time
axes[0].plot(t, motion_mag, lw=1.5)
axes[0].set_title("MediaPipe: filtered hand displacement over time")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Displacement magnitude")
axes[0].grid(True)

# 2) Frequency spectrum
axes[1].semilogy(f, Pxx, lw=1.5)
axes[1].axvline(
    dominant_freq,
    color="r",
    ls="--",
    label=f"Dominant freq = {dominant_freq:.2f} Hz"
)
axes[1].set_title("MediaPipe: frequency spectrum of hand motion")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power spectral density")
axes[1].legend()
axes[1].grid(True)

# 3) Per-axis displacement
labels = ["x", "y", "z"]
for i in range(3):
    axes[2].plot(t, filtered[:, i], label=labels[i])

axes[2].set_title("MediaPipe: filtered displacement per axis")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Displacement")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()


# df = pd.read_csv(CSV_PATH)
# frames = sorted(df.frame_id.unique())

# actors = []

# for fid in frames:
#     frame_df = df[df.frame_id == fid]

#     frame_actors = []

#     for hid, hdf in frame_df.groupby("hand_id"):
#         hdf = hdf.sort_values("joint_id")
#         pts = hdf[["x", "y", "z"]].to_numpy()

#         joints = Points(pts, r=8)

#         lines = Lines(
#             pts[[i for i, _ in HAND_CONNECTIONS]],
#             pts[[j for _, j in HAND_CONNECTIONS]],
#             lw=2,
#         )

#         frame_actors.append(joints + lines)

#     actors.append(sum(frame_actors[1:], frame_actors[0]))


# actor = actors[0].clone()

# def update_scene(i: int):
#     global actor
#     plt.remove(actor)
#     actor = actors[i].clone()
#     plt.add(actor)
#     plt.render()

# plt = AnimationPlayer(update_scene, irange=[0, len(actors) - 1])
# plt += actor
# plt.set_frame(0)
# plt.show()
# plt.close()

import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.signal import butter, filtfilt, welch

from visualizing_files import *
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
HAND_IDX = 0  # 1 for right, 0 for left
LOWPASS_CUTOFF = 6.0
FILTER_ORDER = 3
FPS = 30

# =========================
# LOAD MANO REGRESSOR
# =========================

with open(MANO_RIGHT_PATH, "rb") as f:
    mano = pickle.load(f, encoding="latin1")

J_reg = mano["J_regressor"]

# =========================
# LOAD + CENTER HANDS (NPY)
# =========================

frames = []

for folder in list_frame_folders(ROOT_DIR):
    records = load_frame_records(folder)
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
            "is_right": rec["right"] == 1,
        })

    if frame_hands:
        frames.append(frame_hands)

print(f"Loaded {len(frames)} frames from npy")

# =========================
# BUILD MEAN HAND TRAJECTORY
# =========================

centroids = []

for frame_hands in frames:
    selected = [h["verts"] for h in frame_hands if int(h["is_right"]) == HAND_IDX]
    if not selected:
        continue

    V = np.concatenate(selected, axis=0)
    centroids.append(V.mean(axis=0))

centroids = np.stack(centroids, axis=0)
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
    order=FILTER_ORDER,
)

# =========================
# FREQUENCY ANALYSIS
# =========================

motion_mag = np.linalg.norm(filtered, axis=1)
motion_mag -= motion_mag.mean()

f, Pxx = welch(
    motion_mag,
    fs=FPS,
    nperseg=min(256, len(motion_mag)),
)

dominant_freq = f[np.argmax(Pxx)]
rms_amplitude = np.sqrt(np.mean(motion_mag ** 2))

# =========================
# RESULTS
# =========================

print(f"Dominant frequency: {dominant_freq:.2f} Hz")
print(f"RMS amplitude: {rms_amplitude:.6f}")

t = np.arange(len(motion_mag)) / FPS

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

axes[0].plot(t, motion_mag, lw=1.5)
axes[0].set_title("Filtered hand displacement over time")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Displacement magnitude")
axes[0].grid(True)

axes[1].semilogy(f, Pxx, lw=1.5)
axes[1].axvline(
    dominant_freq,
    color="r",
    ls="--",
    label=f"Dominant freq = {dominant_freq:.2f} Hz",
)
axes[1].set_title("Frequency spectrum of hand motion")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power spectral density")
axes[1].legend()
axes[1].grid(True)

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

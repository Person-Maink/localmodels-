import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from vedo import load, merge
from visualizing_files import *

# =========================
# NUMPY LEGACY PATCH (MANO)
# =========================
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
FPS = 30.0
HIGHPASS_CUTOFF = 8.0
FILTER_ORDER = 3

WILOR_ROOT_A = WILOR_ROOT
WILOR_ROOT_B = WILOR_COMP

HAND_ID = 1   # change to the hand you want

# =========================
# FILTER
# =========================
def highpass(signal):
    nyq = 0.5 * FPS
    b, a = butter(FILTER_ORDER, HIGHPASS_CUTOFF / nyq, btype="high")
    return filtfilt(b, a, signal, axis=0)

# =========================
# LOAD MANO
# =========================
with open(MANO_RIGHT_PATH, "rb") as f:
    mano = pickle.load(f, encoding="latin1")

J_reg = mano["J_regressor"]

# =========================
# WILOR PIPELINE (FUNCTION)
# =========================
def process_wilor(root_dir, HAND_ID):
    frame_folders = sorted(glob.glob(os.path.join(root_dir, "frame_*")))
    centroids = []

    for folder in frame_folders:
        objs = glob.glob(os.path.join(folder, "*.obj"))
        if not objs:
            continue

        # select single hand by is_right flag encoded in filename
        selected = None
        for fpath in objs:
            is_right = float(os.path.splitext(fpath)[0].split("_")[-1])
            if int(is_right) == HAND_ID:
                selected = fpath
                break

        if selected is None:
            continue

        m = load(selected)
        V = m.points
        J = J_reg @ V
        m.shift(-J[0])  # wrist center

        centroids.append(m.points.mean(axis=0))

    centroids = np.stack(centroids)
    filt = highpass(centroids)
    mag = np.linalg.norm(filt, axis=1)
    mag -= mag.mean()

    f, P = welch(mag, fs=FPS, nperseg=min(512, len(mag)))
    dom = f[np.argmax(P)]

    return {
        "centroids": centroids,
        "filtered": filt,
        "magnitude": mag,
        "freqs": f,
        "psd": P,
        "dominant": dom,
    }


# =========================
# RUN BOTH WILOR VARIANTS
# =========================
wilor_A = process_wilor(WILOR_ROOT_A, HAND_ID)
wilor_B = process_wilor(WILOR_ROOT_B, HAND_ID)

tA = np.arange(len(wilor_A["magnitude"])) / FPS
tB = np.arange(len(wilor_B["magnitude"])) / FPS

# =========================
# PLOTS (SAME FIGURE)
# =========================
fig, axes = plt.subplots(3, 1, figsize=(13, 11))

# 1) Displacement over time
axes[0].plot(tA, wilor_A["magnitude"], label=f"Wilor ({wilor_A['dominant']:.2f} Hz)")
axes[0].plot(tB, wilor_B["magnitude"], "--", label=f"Amplified ({wilor_B['dominant']:.2f} Hz)")
axes[0].set_title("Wilor displacement over time")
axes[0].set_ylabel("Displacement magnitude")
axes[0].legend()
axes[0].grid(True)

# 2) Frequency spectrum
axes[1].semilogy(wilor_A["freqs"], wilor_A["psd"], label="Wilor")
axes[1].semilogy(wilor_B["freqs"], wilor_B["psd"], "--", label="Amplified")
axes[1].axvline(wilor_A["dominant"], color="C0", ls=":")
axes[1].axvline(wilor_B["dominant"], color="C1", ls=":")
axes[1].set_title("Frequency spectrum (PSD)")
axes[1].set_ylabel("Power")
axes[1].legend()
axes[1].grid(True)

# 3) Per-axis displacement
labels = ["x", "y", "z"]
for i, lab in enumerate(labels):
    axes[2].plot(tA, wilor_A["filtered"][:, i], label=f"Normal {lab}")
    axes[2].plot(tB, wilor_B["filtered"][:, i], "--", label=f"Amplified {lab}")

axes[2].set_title("Per-axis filtered displacement")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Displacement")
axes[2].legend(ncol=3)
axes[2].grid(True)

plt.tight_layout()
plt.show()

# =========================
# SUMMARY
# =========================
print(f"Wilor dominant frequency: {wilor_A['dominant']:.2f} Hz")
print(f"Amplified dominant frequency: {wilor_B['dominant']:.2f} Hz")

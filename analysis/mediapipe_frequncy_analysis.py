import glob
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from vedo import load, merge
from visualizing_files import *

FPS = 30.0
FILTER_ORDER = 3
HIGHPASS_CUTOFF = 6.0  # Hz (physiological tremor ~8–12 Hz)

MEDIAPIPE_CSV_A = MEDIAPIPE_ROOT          # e.g. original
MEDIAPIPE_CSV_B = MEDIAPIPE_COMP      # e.g. amplified

HAND_ID = 1   # change to the hand you want

# =========================
# FILTER UTILITY
# =========================
def highpass(signal):
    nyq = 0.5 * FPS
    b, a = butter(FILTER_ORDER, HIGHPASS_CUTOFF / nyq, btype="high")
    return filtfilt(b, a, signal, axis=0)

# =========================
# MEDIAPIPE PIPELINE
# =========================
def process_mediapipe(csv_path, HAND_ID):
    df = pd.read_csv(csv_path)
    centroids = []

    hand = "Right" if HAND_ID else "Left"

    for fid in sorted(df.frame_id.unique()):
        frame = df[(df.frame_id == fid) & (df.hand_id == hand)]
        if frame.empty:
            continue

        pts = frame.sort_values("joint_id")[["x", "y", "z"]].to_numpy()
        centroids.append(pts.mean(axis=0))

    centroids = np.stack(centroids)
    filt = highpass(centroids)
    mag = np.linalg.norm(filt, axis=1)
    mag -= mag.mean()

    f, P = welch(mag, fs=FPS, nperseg=min(512, len(mag)))
    dom = f[np.argmax(P)]

    return centroids, filt, mag, f, P, dom


# =========================
# PLOTTING (SAME FIGURE)
# =========================

mpA_cent, mpA_filt, mpA_mag, fA, PA, mpA_dom = process_mediapipe(
    MEDIAPIPE_CSV_A, HAND_ID
)

mpB_cent, mpB_filt, mpB_mag, fB, PB, mpB_dom = process_mediapipe(
    MEDIAPIPE_CSV_B, HAND_ID
)

fig, axes = plt.subplots(3, 1, figsize=(13, 11))

# 1) Displacement over time
axes[0].plot(
    np.arange(len(mpA_mag)) / FPS, mpA_mag, label="MediaPipe original", lw=1.5
)
axes[0].plot(
    np.arange(len(mpB_mag)) / FPS, mpB_mag, "--", label="MediaPipe amplified", lw=1.5
)
axes[0].set_title("Hand displacement over time")
axes[0].set_ylabel("Displacement magnitude")
axes[0].legend()
axes[0].grid(True)

# 2) Frequency spectrum
axes[1].semilogy(fA, PA, label=f"MP original ({mpA_dom:.2f} Hz)")
axes[1].semilogy(fB, PB, "--", label=f"MP amplified ({mpB_dom:.2f} Hz)")
axes[1].axvline(mpA_dom, color="C0", ls=":")
axes[1].axvline(mpB_dom, color="C1", ls=":")
axes[1].set_title("Frequency spectrum (PSD)")
axes[1].set_ylabel("Power")
axes[1].legend()
axes[1].grid(True)

# 3) Per-axis displacement
labels = ["x", "y", "z"]
for i, lab in enumerate(["x", "y", "z"]):
    axes[2].plot(mpA_filt[:, i], label=f"MP orig {lab}")
    axes[2].plot(mpB_filt[:, i], "--", label=f"MP amp {lab}")

axes[2].set_title("Per-axis filtered displacement")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Displacement")
axes[2].legend(ncol=3)
axes[2].grid(True)

plt.tight_layout()
plt.show()

# =========================
# PRINT SUMMARY
# =========================
print(f"MediaPipe original dominant freq:  {mpA_dom:.2f} Hz")
print(f"MediaPipe amplified dominant freq: {mpB_dom:.2f} Hz")


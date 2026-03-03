import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch

from _path_setup import PROJECT_ROOT  # ensures root imports work
from FILENAME import *
from wilor_npy_io import list_frame_folders, load_frame_records

# =========================
# NUMPY LEGACY PATCH
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
FILTER_ORDER = 3
HIGHPASS_CUTOFF = 2.0
LOWPASS_CUTOFF = 12.0

WILOR_ROOT = WILOR_ROOT
MEDIAPIPE_CSV = MEDIAPIPE_ROOT

HAND_ID = 1  # right=1, left=0


# =========================
# FILTER UTILITY
# =========================
def highpass(signal):
    nyq = 0.5 * FPS

    b_high, a_high = butter(FILTER_ORDER, HIGHPASS_CUTOFF / nyq, btype="high")
    signal = filtfilt(b_high, a_high, signal, axis=0)

    b_low, a_low = butter(FILTER_ORDER, LOWPASS_CUTOFF / nyq, btype="low")
    signal = filtfilt(b_low, a_low, signal, axis=0)

    return signal


# =========================
# WILOR PIPELINE (NPY)
# =========================
wilor_centroids = []

for folder in list_frame_folders(WILOR_ROOT):
    records = load_frame_records(folder)
    records = [r for r in records if r["right"] == HAND_ID]
    if not records:
        continue

    # Pick first matched hand in this frame.
    wilor_centroids.append(records[0]["verts_world"].mean(axis=0))

wilor_centroids = np.stack(wilor_centroids)
wilor_filt = highpass(wilor_centroids)
wilor_mag = np.linalg.norm(wilor_filt, axis=1)
wilor_mag -= wilor_mag.mean()

fw, Pw = welch(wilor_mag, fs=FPS, nperseg=min(512, len(wilor_mag)))
wilor_dom = fw[np.argmax(Pw)]

# =========================
# MEDIAPIPE PIPELINE
# =========================
df = pd.read_csv(MEDIAPIPE_CSV)
mp_centroids = []

for fid in sorted(df.frame_id.unique()):
    hand = "Right" if HAND_ID else "Left"
    frame = df[(df.frame_id == fid) & (df.hand_id == hand)]
    if frame.empty:
        continue

    pts = frame.sort_values("joint_id")[["x", "y", "z"]].to_numpy()
    mp_centroids.append(pts.mean(axis=0))

mp_centroids = np.stack(mp_centroids)
mp_filt = highpass(mp_centroids)
mp_mag = np.linalg.norm(mp_filt, axis=1)
mp_mag -= mp_mag.mean()

fm, Pm = welch(mp_mag, fs=FPS, nperseg=min(512, len(mp_mag)))
mp_dom = fm[np.argmax(Pm)]

# =========================
# PLOTTING
# =========================
t_w = np.arange(len(wilor_mag)) / FPS
t_m = np.arange(len(mp_mag)) / FPS

fig, axes = plt.subplots(3, 1, figsize=(13, 11))

axes[0].plot(t_w, wilor_mag, label="Wilor", lw=1.5)
axes[0].plot(t_m, mp_mag, "--", label="MediaPipe", lw=1.5)
axes[0].set_title("Hand displacement over time")
axes[0].set_ylabel("Displacement magnitude")
axes[0].legend()
axes[0].grid(True)

axes[1].semilogy(fw, Pw, label=f"Wilor ({wilor_dom:.2f} Hz)")
axes[1].semilogy(fm, Pm, "--", label=f"MediaPipe ({mp_dom:.2f} Hz)")
axes[1].axvline(wilor_dom, color="C0", ls=":")
axes[1].axvline(mp_dom, color="C1", ls=":")
axes[1].set_title("Frequency spectrum (PSD)")
axes[1].set_ylabel("Power")
axes[1].legend()
axes[1].grid(True)

labels = ["x", "y", "z"]
for i, lab in enumerate(labels):
    axes[2].plot(t_w, wilor_filt[:, i], label=f"Wilor {lab}")
    axes[2].plot(t_m, mp_filt[:, i], "--", label=f"MP {lab}")

axes[2].set_title("Per-axis filtered displacement")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Displacement")
axes[2].legend(ncol=3)
axes[2].grid(True)

plt.tight_layout()
plt.show()

print(f"Wilor dominant frequency:     {wilor_dom:.2f} Hz")

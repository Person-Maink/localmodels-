import numpy as np
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
HIGHPASS_CUTOFF = 2.0
LOWPASS_CUTOFF = 12.0
FILTER_ORDER = 3

WILOR_ROOT_A = WILOR_ROOT
WILOR_ROOT_B = WILOR_COMP

HAND_ID = 1  # right=1, left=0


# =========================
# FILTER
# =========================
def highpass(signal):
    nyq = 0.5 * FPS

    b_high, a_high = butter(FILTER_ORDER, HIGHPASS_CUTOFF / nyq, btype="high")
    signal = filtfilt(b_high, a_high, signal, axis=0)

    b_low, a_low = butter(FILTER_ORDER, LOWPASS_CUTOFF / nyq, btype="low")
    signal = filtfilt(b_low, a_low, signal, axis=0)

    return signal


# =========================
# WILOR PIPELINE (FUNCTION)
# =========================
def process_wilor(root_dir, hand_id):
    centroids = []

    for folder in list_frame_folders(root_dir):
        records = load_frame_records(folder)
        records = [r for r in records if r["right"] == hand_id]
        if not records:
            continue

        centroids.append(records[0]["verts_world"].mean(axis=0))

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
# PLOTS
# =========================
fig, axes = plt.subplots(3, 1, figsize=(13, 11))

axes[0].plot(tA, wilor_A["magnitude"], label=f"Wilor ({wilor_A['dominant']:.2f} Hz)")
axes[0].plot(tB, wilor_B["magnitude"], "--", label=f"Amplified ({wilor_B['dominant']:.2f} Hz)")
axes[0].set_title("Wilor displacement over time")
axes[0].set_ylabel("Displacement magnitude")
axes[0].legend()
axes[0].grid(True)

axes[1].semilogy(wilor_A["freqs"], wilor_A["psd"], label="Wilor")
axes[1].semilogy(wilor_B["freqs"], wilor_B["psd"], "--", label="Amplified")
axes[1].axvline(wilor_A["dominant"], color="C0", ls=":")
axes[1].axvline(wilor_B["dominant"], color="C1", ls=":")
axes[1].set_title("Frequency spectrum (PSD)")
axes[1].set_ylabel("Power")
axes[1].legend()
axes[1].grid(True)

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

print(f"Wilor dominant frequency: {wilor_A['dominant']:.2f} Hz")
print(f"Amplified dominant frequency: {wilor_B['dominant']:.2f} Hz")

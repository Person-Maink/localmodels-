import pickle
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FILENAME import (  # noqa: E402
    MANO_RIGHT_PATH,
    WILOR_ROOT,
    WILOR_SPECIFIC_VERTEX_A,
    WILOR_SPECIFIC_VERTEX_B,
)
from wilor_npy_io import list_frame_folders, load_frame_records  # noqa: E402


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
N_NEIGHBORS = 3

LOWPASS_CUTOFF = 6.0
FILTER_ORDER = 3
FPS = 30


# =========================
# LOAD MANO DATA
# =========================
with open(MANO_RIGHT_PATH, "rb") as f:
    mano = pickle.load(f, encoding="latin1")

# J_reg = np.asarray(mano["J_regressor"], dtype=np.float32)
J_reg = mano["J_regressor"]
FACES = np.asarray(mano["f"], dtype=np.int32)
N_VERTS = int(J_reg.shape[1])


# =========================
# TOPOLOGY HELPERS
# =========================
def validate_config_indices(n_verts, seed_a, seed_b):
    if not (0 <= seed_a < n_verts):
        raise ValueError(f"WILOR_SPECIFIC_VERTEX_A={seed_a} out of range [0, {n_verts-1}]")
    if not (0 <= seed_b < n_verts):
        raise ValueError(f"WILOR_SPECIFIC_VERTEX_B={seed_b} out of range [0, {n_verts-1}]")
    if seed_a == seed_b:
        raise ValueError("WILOR_SPECIFIC_VERTEX_A and WILOR_SPECIFIC_VERTEX_B must be different")
    if N_NEIGHBORS <= 0:
        raise ValueError(f"N_NEIGHBORS must be > 0, got {N_NEIGHBORS}")


def build_vertex_adjacency(n_verts, faces):
    adj = [set() for _ in range(n_verts)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    return [sorted(list(nei)) for nei in adj]


def select_graph_neighbors(seed, adjacency, n_neighbors):
    """
    Return up to n_neighbors nearest neighbors by graph hop distance.
    Deterministic order: lower hop first, then lower vertex id.
    """
    visited = {seed}
    q = deque([(seed, 0)])
    ranked = []

    while q:
        node, dist = q.popleft()
        for nb in adjacency[node]:
            if nb in visited:
                continue
            visited.add(nb)
            nd = dist + 1
            ranked.append((nd, nb))
            q.append((nb, nd))

    ranked.sort(key=lambda x: (x[0], x[1]))
    return [vid for _, vid in ranked[:n_neighbors]]


def build_region_indices(seed, adjacency, n_neighbors):
    selected = select_graph_neighbors(seed, adjacency, n_neighbors)
    if len(selected) < n_neighbors:
        print(
            f"[warn] seed={seed}: requested {n_neighbors} neighbors, "
            f"got {len(selected)} available"
        )
    return np.asarray([seed] + selected, dtype=np.int32)


# =========================
# FILTER
# =========================
def lowpass_filter(signal, fs, cutoff, order):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal, axis=0)


# =========================
# LOAD + CENTER HANDS
# =========================
validate_config_indices(N_VERTS, WILOR_SPECIFIC_VERTEX_A, WILOR_SPECIFIC_VERTEX_B)

adjacency = build_vertex_adjacency(N_VERTS, FACES)
region_a = build_region_indices(WILOR_SPECIFIC_VERTEX_A, adjacency, N_NEIGHBORS)
region_b = build_region_indices(WILOR_SPECIFIC_VERTEX_B, adjacency, N_NEIGHBORS)

print(f"Using vertex A={WILOR_SPECIFIC_VERTEX_A}, region size={len(region_a)}")
print(f"Using vertex B={WILOR_SPECIFIC_VERTEX_B}, region size={len(region_b)}")
print(f"Region A indices: {region_a.tolist()}")
print(f"Region B indices: {region_b.tolist()}")

frames = []
for folder in list_frame_folders(ROOT_DIR):
    records = load_frame_records(folder)
    if not records:
        continue

    frame_hands = []
    for rec in records:
        V = rec["verts_world"]
        if V.shape[0] != N_VERTS:
            raise ValueError(
                f"Vertex count mismatch in {rec['path']}: got {V.shape[0]}, expected {N_VERTS}"
            )

        J = J_reg @ V
        wrist = J[WRIST_JOINT_IDX]
        V_centered = V - wrist

        frame_hands.append(
            {
                "verts": V_centered,
                "is_right": rec["right"] == 1,
            }
        )

    if frame_hands:
        frames.append(frame_hands)

print(f"Loaded {len(frames)} frames from npy")


# =========================
# BUILD DIFFERENCE TRAJECTORY
# =========================
trajectory = []

for frame_hands in frames:
    selected = [h["verts"] for h in frame_hands if int(h["is_right"]) == HAND_IDX]
    if not selected:
        continue

    frame_diffs = []
    for verts in selected:
        centroid_a = verts[region_a].mean(axis=0)
        centroid_b = verts[region_b].mean(axis=0)
        frame_diffs.append(centroid_a - centroid_b)

    frame_diff = np.mean(np.stack(frame_diffs, axis=0), axis=0)
    trajectory.append(frame_diff)

if not trajectory:
    raise RuntimeError(
        f"No valid frames for HAND_IDX={HAND_IDX} under {ROOT_DIR}. "
        "Check hand side and input clip."
    )

trajectory = np.stack(trajectory, axis=0)
print("Trajectory shape:", trajectory.shape)


# =========================
# FILTER + FREQUENCY ANALYSIS
# =========================
filtered = lowpass_filter(
    trajectory,
    fs=FPS,
    cutoff=LOWPASS_CUTOFF,
    order=FILTER_ORDER,
)

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
axes[0].set_title("Filtered region-difference displacement over time")
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
axes[1].set_title("Frequency spectrum of region-difference motion")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power spectral density")
axes[1].legend()
axes[1].grid(True)

labels = ["x", "y", "z"]
for i in range(3):
    axes[2].plot(t, filtered[:, i], label=labels[i])

axes[2].set_title("Filtered region-difference displacement per axis")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Displacement")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

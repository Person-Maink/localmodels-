import argparse
import pickle
from collections import deque
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from mano_pickle import load_mano_pickle
from npy_io import list_frame_folders, load_frame_records


# NumPy legacy aliases for old pickle compatibility.
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.unicode = str
np.nan = float("nan")
np.inf = float("inf")


LOWPASS_CUTOFF = 6.0
FILTER_ORDER = 3
FPS = 30.0
DEFAULT_SLOT_NAMES = ("A", "B", "C", "D")
LINE_STYLES = ("-", "--", "-.", ":")


@lru_cache(maxsize=1)
def _load_mano_assets(mano_right_path):
    mano = load_mano_pickle(mano_right_path)

    j_reg = mano["J_regressor"]
    faces = np.asarray(mano["f"], dtype=np.int32)
    return j_reg, faces


def _normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _default_source_a():
    if hasattr(CONFIG, "POINT_SOURCE_A"):
        return getattr(CONFIG, "POINT_SOURCE_A")
    return getattr(CONFIG, "MODEL_ROOT", None)


def _default_source_b():
    if hasattr(CONFIG, "POINT_SOURCE_B"):
        return getattr(CONFIG, "POINT_SOURCE_B")
    return getattr(CONFIG, "MODEL_COMP", None)


def _default_all_model_sources():
    return [
        getattr(CONFIG, "WILOR_ROOT", None),
        getattr(CONFIG, "HAMBA_ROOT", None),
        getattr(CONFIG, "DYNHAMR_ROOT", None),
        getattr(CONFIG, "MEDIAPIPE_ROOT", None),
    ]


def _default_all_model_labels():
    return ["WILOR", "HAMBA", "DYNHAMR", "MEDIAPIPE"]


def _infer_label(root_dir, fallback):
    if root_dir is None:
        return fallback

    path = Path(root_dir)
    if path.suffix.lower() == ".csv":
        return path.stem
    if path.name.lower() in {"meshes", "mesh", "npy"} and path.parent.name:
        return path.parent.name
    if path.name:
        return path.name
    return fallback


def _source_kind(path_text):
    path = Path(path_text)
    if path.suffix.lower() == ".csv":
        return "mediapipe"
    return "model"


def _validate_config_indices(total_verts, seed_a, seed_b, n_neighbors):
    if not (0 <= seed_a < total_verts):
        raise ValueError(f"MODEL_SPECIFIC_VERTEX_A={seed_a} out of range [0, {total_verts-1}]")
    if not (0 <= seed_b < total_verts):
        raise ValueError(f"MODEL_SPECIFIC_VERTEX_B={seed_b} out of range [0, {total_verts-1}]")
    if seed_a == seed_b:
        raise ValueError("MODEL_SPECIFIC_VERTEX_A and MODEL_SPECIFIC_VERTEX_B must be different")
    if n_neighbors <= 0:
        raise ValueError(f"N_NEIGHBORS must be > 0, got {n_neighbors}")


def _validate_mediapipe_indices(point_a, point_b):
    if point_a < 0:
        raise ValueError(f"MEDIAPIPE_POINT_COORD_A must be >= 0, got {point_a}")
    if point_b < 0:
        raise ValueError(f"MEDIAPIPE_POINT_COORD_B must be >= 0, got {point_b}")
    if point_a == point_b:
        raise ValueError("MEDIAPIPE_POINT_COORD_A and MEDIAPIPE_POINT_COORD_B must be different")


def _build_vertex_adjacency(total_verts, tri_faces):
    adjacency = [set() for _ in range(total_verts)]
    for tri in tri_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adjacency[a].add(b)
        adjacency[a].add(c)
        adjacency[b].add(a)
        adjacency[b].add(c)
        adjacency[c].add(a)
        adjacency[c].add(b)
    return [sorted(list(nei)) for nei in adjacency]


def _select_graph_neighbors(seed, adjacency, n_neighbors):
    visited = {seed}
    queue = deque([(seed, 0)])
    ranked = []

    while queue:
        node, dist = queue.popleft()
        for nb in adjacency[node]:
            if nb in visited:
                continue
            visited.add(nb)
            next_dist = dist + 1
            ranked.append((next_dist, nb))
            queue.append((nb, next_dist))

    ranked.sort(key=lambda x: (x[0], x[1]))
    return [vertex_id for _, vertex_id in ranked[:n_neighbors]]


def _build_region_indices(seed, adjacency, n_neighbors):
    selected = _select_graph_neighbors(seed, adjacency, n_neighbors)
    if len(selected) < n_neighbors:
        print(
            f"[warn] seed={seed}: requested {n_neighbors} neighbors, "
            f"got {len(selected)} available"
        )
    return np.asarray([seed] + selected, dtype=np.int32)


def _lowpass_filter(signal, fs, cutoff, order):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal, axis=0)


def _finish_analysis(trajectory):
    trajectory = np.stack(trajectory, axis=0)
    filtered = _lowpass_filter(
        trajectory,
        fs=FPS,
        cutoff=LOWPASS_CUTOFF,
        order=FILTER_ORDER,
    )

    magnitude = np.linalg.norm(filtered, axis=1)
    magnitude -= magnitude.mean()

    freqs, psd = welch(
        magnitude,
        fs=FPS,
        nperseg=min(256, len(magnitude)),
    )

    dominant_freq = float(freqs[np.argmax(psd)])
    rms_amplitude = float(np.sqrt(np.mean(magnitude**2)))

    return {
        "trajectory": trajectory,
        "filtered": filtered,
        "magnitude": magnitude,
        "freqs": freqs,
        "psd": psd,
        "dominant": dominant_freq,
        "rms": rms_amplitude,
    }


def _analyze_model(root_dir, j_reg, region_a, region_b, n_verts, hand_idx, wrist_joint_idx):
    frames = []
    for folder in list_frame_folders(root_dir):
        records = load_frame_records(folder)
        if not records:
            continue

        frame_hands = []
        for rec in records:
            verts_world = rec["verts_world"]
            if verts_world.shape[0] != n_verts:
                raise ValueError(
                    f"Vertex count mismatch in {rec['path']}: got {verts_world.shape[0]}, expected {n_verts}"
                )

            joints = j_reg @ verts_world
            wrist = joints[wrist_joint_idx]
            centered = verts_world - wrist

            frame_hands.append(
                {
                    "verts": centered,
                    "is_right": rec["right"] == 1,
                }
            )

        if frame_hands:
            frames.append(frame_hands)

    print(f"Loaded {len(frames)} frames from npy for: {root_dir}")

    trajectory = []
    for frame_hands in frames:
        selected = [h["verts"] for h in frame_hands if int(h["is_right"]) == hand_idx]
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
            f"No valid frames for HAND_IDX={hand_idx} under {root_dir}. "
            "Check hand side and input clip."
        )

    print(f"Loaded {len(trajectory)} usable model frames for: {root_dir}")
    return _finish_analysis(trajectory)


def _analyze_mediapipe(csv_path, point_a, point_b, hand_idx):
    hand_label = "Right" if int(hand_idx) == 1 else "Left"
    df = pd.read_csv(csv_path)
    hand_df = df[df["hand_id"] == hand_label]

    if hand_df.empty:
        raise ValueError(f"No usable MediaPipe records in '{csv_path}' for hand='{hand_label}'.")

    available_joint_ids = set(hand_df["joint_id"].astype(int).unique().tolist())
    missing = [joint_id for joint_id in (point_a, point_b) if joint_id not in available_joint_ids]
    if missing:
        missing_text = ", ".join(str(joint_id) for joint_id in missing)
        raise ValueError(
            f"MediaPipe point selector(s) {missing_text} not present in '{csv_path}' for hand='{hand_label}'."
        )

    trajectory = []
    skipped_frames = 0

    for frame_id in sorted(hand_df["frame_id"].unique()):
        frame = hand_df[hand_df["frame_id"] == frame_id].sort_values("joint_id")
        points = frame.set_index("joint_id")[["x", "y", "z"]]

        if point_a not in points.index or point_b not in points.index:
            skipped_frames += 1
            continue

        point_a_xyz = points.loc[point_a].to_numpy(dtype=float)
        point_b_xyz = points.loc[point_b].to_numpy(dtype=float)
        trajectory.append(point_a_xyz - point_b_xyz)

    if not trajectory:
        raise ValueError(
            f"No usable MediaPipe frames in '{csv_path}' for hand='{hand_label}' "
            f"with joint ids {point_a} and {point_b}."
        )

    print(
        f"Loaded {len(trajectory)} usable MediaPipe frames for: {csv_path}"
        + (f" (skipped {skipped_frames} incomplete frames)" if skipped_frames else "")
    )
    return _finish_analysis(trajectory)


def _analyze_source(
    source_path,
    hand_idx,
    wrist_joint_idx,
    j_reg=None,
    region_a=None,
    region_b=None,
    n_verts=None,
    mediapipe_point_a=None,
    mediapipe_point_b=None,
):
    kind = _source_kind(source_path)
    if kind == "mediapipe":
        return _analyze_mediapipe(source_path, mediapipe_point_a, mediapipe_point_b, hand_idx)
    return _analyze_model(source_path, j_reg, region_a, region_b, n_verts, hand_idx, wrist_joint_idx)


def _resolve_entries(overrides):
    all_models = bool(overrides.get("all_models", False))
    sources_override = overrides.get("sources", None)
    labels_override = overrides.get("labels", None)

    if sources_override is not None:
        raw_sources = list(sources_override)
    elif all_models:
        raw_sources = _default_all_model_sources()
    else:
        raw_sources = [
            overrides.get("source_a", _default_source_a()),
            overrides.get("source_b", _default_source_b()),
        ]

    normalized_sources = [_normalize_optional_path(source) for source in raw_sources]
    if not any(source is not None for source in normalized_sources):
        raise ValueError(
            "No point-to-point sources resolved. Set POINT_SOURCE_A/B in FILENAME.py, "
            "pass explicit sources, or use --all-models with configured WILOR/HAMBA/MEDIAPIPE roots."
        )

    if labels_override is not None:
        if len(labels_override) != len(raw_sources):
            raise ValueError("labels length must match sources length when both are provided.")
        raw_labels = list(labels_override)
    elif all_models and sources_override is None:
        raw_labels = _default_all_model_labels()
    else:
        raw_labels = []
        if len(raw_sources) >= 1:
            raw_labels.append(
                overrides.get("label_a") or getattr(CONFIG, "POINT_LABEL_A", None) or _infer_label(raw_sources[0], "Source A")
            )
        if len(raw_sources) >= 2:
            raw_labels.append(
                overrides.get("label_b") or getattr(CONFIG, "POINT_LABEL_B", None) or _infer_label(raw_sources[1], "Source B")
            )
        for idx in range(2, len(raw_sources)):
            raw_labels.append(_infer_label(raw_sources[idx], f"Source {idx + 1}"))

    entries = []
    for idx, source in enumerate(normalized_sources):
        if source is None:
            continue
        label = raw_labels[idx] if idx < len(raw_labels) else None
        entries.append(
            {
                "slot": DEFAULT_SLOT_NAMES[idx] if idx < len(DEFAULT_SLOT_NAMES) else f"S{idx + 1}",
                "source": source,
                "kind": _source_kind(source),
                "label": label or _infer_label(source, f"Source {idx + 1}"),
            }
        )
    return entries


def run_point_to_point_analysis(config_overrides=None):
    overrides = config_overrides or {}
    entries = _resolve_entries(overrides)

    hand_idx = int(overrides.get("hand_idx", CONFIG.HAND_IDX))
    wrist_joint_idx = int(overrides.get("wrist_joint_idx", CONFIG.WRIST_JOINT_IDX))
    n_neighbors = int(overrides.get("n_neighbors", CONFIG.N_NEIGHBORS))
    vertex_a = int(overrides.get("vertex_a", CONFIG.MODEL_SPECIFIC_VERTEX_A))
    vertex_b = int(overrides.get("vertex_b", CONFIG.MODEL_SPECIFIC_VERTEX_B))
    mediapipe_point_a = int(overrides.get("mediapipe_point_a", CONFIG.MEDIAPIPE_POINT_COORD_A))
    mediapipe_point_b = int(overrides.get("mediapipe_point_b", CONFIG.MEDIAPIPE_POINT_COORD_B))

    _validate_mediapipe_indices(mediapipe_point_a, mediapipe_point_b)

    kinds = [entry["kind"] for entry in entries]

    j_reg = None
    n_verts = None
    region_a = None
    region_b = None

    if "model" in kinds:
        try:
            j_reg, faces = _load_mano_assets(str(CONFIG.MANO_RIGHT_PATH))
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Point-to-point analysis needs MANO pickle dependencies for model sources "
                "(missing module while loading MANO_RIGHT_PATH). Install the missing module "
                "(commonly 'chumpy') or switch POINT_SOURCE_A/B to MediaPipe CSV inputs."
            ) from exc

        n_verts = int(j_reg.shape[1])
        _validate_config_indices(n_verts, vertex_a, vertex_b, n_neighbors)

        adjacency = _build_vertex_adjacency(n_verts, faces)
        region_a = _build_region_indices(vertex_a, adjacency, n_neighbors)
        region_b = _build_region_indices(vertex_b, adjacency, n_neighbors)

        print(f"Using vertex A={vertex_a}, region size={len(region_a)}")
        print(f"Using vertex B={vertex_b}, region size={len(region_b)}")
        print(f"Region A indices: {region_a.tolist()}")
        print(f"Region B indices: {region_b.tolist()}")

    resolved_entries = []
    for entry in entries:
        resolved_entries.append(
            {
                **entry,
                "result": _analyze_source(
                    entry["source"],
                    hand_idx,
                    wrist_joint_idx,
                    j_reg=j_reg,
                    region_a=region_a,
                    region_b=region_b,
                    n_verts=n_verts,
                    mediapipe_point_a=mediapipe_point_a,
                    mediapipe_point_b=mediapipe_point_b,
                ),
            }
        )

    return {
        "analysis": "point_to_point",
        "hand_idx": hand_idx,
        "wrist_joint_idx": wrist_joint_idx,
        "vertex_a": vertex_a,
        "vertex_b": vertex_b,
        "mediapipe_point_a": mediapipe_point_a,
        "mediapipe_point_b": mediapipe_point_b,
        "n_neighbors": n_neighbors,
        "entries": resolved_entries,
    }


def build_point_to_point_figure(analysis_data, figsize_inches=(12, 10), dpi=100):
    fig, axes = plt.subplots(3, 1, figsize=figsize_inches, dpi=dpi, sharex=False)

    for i, entry in enumerate(analysis_data["entries"]):
        label = entry["label"]
        result = entry["result"]

        style = LINE_STYLES[i % len(LINE_STYLES)]
        color = f"C{i}"
        t = np.arange(len(result["magnitude"])) / FPS

        axes[0].plot(t, result["magnitude"], style, color=color, lw=1.5, label=label)

        axes[1].semilogy(
            result["freqs"],
            result["psd"],
            style,
            color=color,
            lw=1.5,
            label=f"{label} ({result['dominant']:.2f} Hz)",
        )
        axes[1].axvline(result["dominant"], color=color, ls=":")

        for axis_i, axis_name in enumerate(["x", "y", "z"]):
            axes[2].plot(
                t,
                result["filtered"][:, axis_i],
                style,
                color=color,
                lw=1.2,
                label=f"{label} {axis_name}",
            )

    axes[0].set_title("Filtered region-difference displacement over time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Displacement magnitude")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Frequency spectrum of region-difference motion")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power spectral density")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_title("Filtered region-difference displacement per axis")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Displacement")
    axes[2].legend(ncol=3)
    axes[2].grid(True)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Run point-to-point frequency analysis across configured sources.")
    parser.add_argument("--all-models", action="store_true", help="Compare WiLoR, Hamba, DynHAMR, and MediaPipe together.")
    args = parser.parse_args()

    analysis_data = run_point_to_point_analysis({"all_models": args.all_models})
    fig = build_point_to_point_figure(analysis_data)
    plt.show()
    plt.close(fig)

    for entry in analysis_data["entries"]:
        label = entry["label"]
        result = entry["result"]
        print(f"{label} dominant frequency: {result['dominant']:.2f} Hz")
        print(f"{label} RMS amplitude: {result['rms']:.6f}")


if __name__ == "__main__":
    main()

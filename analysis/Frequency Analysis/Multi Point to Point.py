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
SOURCE_STYLE = {
    "wilor": "-",
    "hamba": "--",
    "dynhamr": "-.",
    "mediapipe": ":",
}
AXIS_ALPHA = {
    "x": 1.0,
    "y": 0.75,
    "z": 0.5,
}

# Add optional per-run extras here without touching FILENAME.py.
EXTRA_MANO_PAIRS = []
EXTRA_MEDIAPIPE_PAIRS = []


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


def _pair_tuple(pair):
    if len(pair) != 2:
        raise ValueError(f"Pair must contain exactly two indices, got: {pair}")
    return (int(pair[0]), int(pair[1]))


def _dedupe_pairs(pairs):
    seen = set()
    resolved = []
    for pair in pairs:
        item = _pair_tuple(pair)
        if item in seen:
            continue
        seen.add(item)
        resolved.append(item)
    return resolved


def _default_mano_pairs():
    raw = getattr(CONFIG, "MULTI_POINT_MANO_PAIRS", None)
    if raw:
        return list(raw)
    return [(int(CONFIG.MODEL_SPECIFIC_VERTEX_A), int(CONFIG.MODEL_SPECIFIC_VERTEX_B))]


def _default_mediapipe_pairs():
    raw = getattr(CONFIG, "MULTI_POINT_MEDIAPIPE_PAIRS", None)
    if raw:
        return list(raw)
    return [(int(CONFIG.MEDIAPIPE_POINT_COORD_A), int(CONFIG.MEDIAPIPE_POINT_COORD_B))]


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
            f"[warn] seed={seed}: requested {n_neighbors} neighbors, got {len(selected)} available"
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


def _validate_mano_pair(total_verts, pair):
    a, b = pair
    if not (0 <= a < total_verts):
        raise ValueError(f"MANO pair index A={a} out of range [0, {total_verts-1}]")
    if not (0 <= b < total_verts):
        raise ValueError(f"MANO pair index B={b} out of range [0, {total_verts-1}]")
    if a == b:
        raise ValueError(f"MANO pair indices must be different, got {pair}")


def _validate_mediapipe_pair(pair):
    a, b = pair
    if a < 0 or b < 0:
        raise ValueError(f"MediaPipe pair indices must be >= 0, got {pair}")
    if a == b:
        raise ValueError(f"MediaPipe pair indices must be different, got {pair}")


def _pair_label(kind, pair):
    prefix = "v" if kind == "model" else "j"
    return f"{prefix}{pair[0]}-{prefix}{pair[1]}"


def _resolve_sources(config_overrides):
    explicit_sources = config_overrides.get("sources", None)
    if explicit_sources is not None:
        if isinstance(explicit_sources, dict):
            sources = {name: _normalize_optional_path(path) for name, path in explicit_sources.items()}
        else:
            sources = {}
            for item in explicit_sources:
                if not isinstance(item, dict):
                    raise ValueError("Explicit multi-point sources must be dict items with 'family' and 'path'.")
                family = str(item.get("family", "")).strip().lower()
                if not family:
                    raise ValueError("Explicit multi-point source is missing a family name.")
                sources[family] = _normalize_optional_path(item.get("path"))
        sources = {name: path for name, path in sources.items() if path is not None}
        if not sources:
            raise ValueError("No valid explicit multi-point sources were provided.")
        return sources

    sources = {
        "wilor": _normalize_optional_path(config_overrides.get("wilor_source", getattr(CONFIG, "WILOR_ROOT", None))),
        "hamba": _normalize_optional_path(config_overrides.get("hamba_source", getattr(CONFIG, "HAMBA_ROOT", None))),
        "dynhamr": _normalize_optional_path(config_overrides.get("dynhamr_source", getattr(CONFIG, "DYNHAMR_ROOT", None))),
        "mediapipe": _normalize_optional_path(config_overrides.get("mediapipe_source", getattr(CONFIG, "MEDIAPIPE_ROOT", None))),
    }
    return {name: path for name, path in sources.items() if path is not None}


def _resolve_pair_lists(config_overrides):
    mano_pairs = _dedupe_pairs(_default_mano_pairs() + list(EXTRA_MANO_PAIRS) + list(config_overrides.get("extra_mano_pairs", [])))
    mediapipe_pairs = _dedupe_pairs(
        _default_mediapipe_pairs() + list(EXTRA_MEDIAPIPE_PAIRS) + list(config_overrides.get("extra_mediapipe_pairs", []))
    )

    if not mano_pairs:
        raise ValueError("No MANO pairs resolved. Set MULTI_POINT_MANO_PAIRS or add EXTRA_MANO_PAIRS.")
    if not mediapipe_pairs:
        raise ValueError("No MediaPipe pairs resolved. Set MULTI_POINT_MEDIAPIPE_PAIRS or add EXTRA_MEDIAPIPE_PAIRS.")

    return mano_pairs, mediapipe_pairs


def _collect_model_frames(root_dir, j_reg, hand_idx, wrist_joint_idx, n_verts):
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
            wrist = joints[int(wrist_joint_idx)]
            frame_hands.append(
                {
                    "verts": verts_world - wrist,
                    "is_right": rec["right"] == 1,
                }
            )

        if frame_hands:
            frames.append(frame_hands)

    if not frames:
        raise RuntimeError(f"No frames found under model source '{root_dir}'.")

    return frames


def _analyze_model_pair(frames, hand_idx, region_a, region_b, source_path, pair_label):
    trajectory = []
    for frame_hands in frames:
        selected = [h["verts"] for h in frame_hands if int(h["is_right"]) == int(hand_idx)]
        if not selected:
            continue

        frame_diffs = []
        for verts in selected:
            centroid_a = verts[region_a].mean(axis=0)
            centroid_b = verts[region_b].mean(axis=0)
            frame_diffs.append(centroid_a - centroid_b)

        trajectory.append(np.mean(np.stack(frame_diffs, axis=0), axis=0))

    if not trajectory:
        raise RuntimeError(
            f"No valid model frames for HAND_IDX={hand_idx} under {source_path} for pair {pair_label}."
        )

    return _finish_analysis(trajectory)


def _analyze_mediapipe_pair(df, hand_idx, pair, source_path):
    hand_label = "Right" if int(hand_idx) == 1 else "Left"
    hand_df = df[df["hand_id"] == hand_label]
    if hand_df.empty:
        raise ValueError(f"No usable MediaPipe records in '{source_path}' for hand='{hand_label}'.")

    available_joint_ids = set(hand_df["joint_id"].astype(int).unique().tolist())
    missing = [joint_id for joint_id in pair if joint_id not in available_joint_ids]
    if missing:
        raise ValueError(
            f"MediaPipe pair indices {missing} not present in '{source_path}' for hand='{hand_label}'."
        )

    trajectory = []
    skipped_frames = 0
    point_a, point_b = pair

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
            f"No usable MediaPipe frames in '{source_path}' for hand='{hand_label}' with pair {pair}."
        )

    if skipped_frames:
        print(f"[info] {source_path}: skipped {skipped_frames} incomplete MediaPipe frames for pair {pair}")

    return _finish_analysis(trajectory)


def run_multi_point_analysis(config_overrides=None):
    overrides = config_overrides or {}

    sources = _resolve_sources(overrides)
    if not sources:
        raise ValueError("No multi-point sources resolved.")
    mano_pairs, mediapipe_pairs = _resolve_pair_lists(overrides)
    hand_idx = int(overrides.get("hand_idx", CONFIG.HAND_IDX))
    wrist_joint_idx = int(overrides.get("wrist_joint_idx", CONFIG.WRIST_JOINT_IDX))
    n_neighbors = int(overrides.get("n_neighbors", CONFIG.N_NEIGHBORS))

    if n_neighbors <= 0:
        raise ValueError(f"N_NEIGHBORS must be > 0, got {n_neighbors}")

    try:
        j_reg, faces = _load_mano_assets(str(CONFIG.MANO_RIGHT_PATH))
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Multi-point MANO analysis needs MANO pickle dependencies (commonly 'chumpy')."
        ) from exc

    n_verts = int(j_reg.shape[1])
    adjacency = _build_vertex_adjacency(n_verts, faces)

    mano_regions = {}
    for pair in mano_pairs:
        _validate_mano_pair(n_verts, pair)
        region_a = _build_region_indices(pair[0], adjacency, n_neighbors)
        region_b = _build_region_indices(pair[1], adjacency, n_neighbors)
        mano_regions[pair] = (region_a, region_b)

    for pair in mediapipe_pairs:
        _validate_mediapipe_pair(pair)

    entries = []

    model_families = [family for family in ("wilor", "hamba", "dynhamr") if family in sources]
    media_pipe_enabled = "mediapipe" in sources
    if not model_families and not media_pipe_enabled:
        raise ValueError("Multi-point analysis needs at least one supported source family.")

    model_frames_by_family = {
        family: _collect_model_frames(sources[family], j_reg, hand_idx, wrist_joint_idx, n_verts)
        for family in model_families
    }
    mediapipe_df = pd.read_csv(sources["mediapipe"]) if media_pipe_enabled else None

    for family, frames in model_frames_by_family.items():
        for pair in mano_pairs:
            region_a, region_b = mano_regions[pair]
            label = _pair_label("model", pair)
            entries.append(
                {
                    "source_family": family,
                    "kind": "model",
                    "slot": label,
                    "label": f"{family.upper()} {label}",
                    "source": sources[family],
                    "pair": pair,
                    "pair_label": label,
                    "result": _analyze_model_pair(frames, hand_idx, region_a, region_b, sources[family], label),
                }
            )

    if media_pipe_enabled:
        for pair in mediapipe_pairs:
            label = _pair_label("mediapipe", pair)
            entries.append(
                {
                    "source_family": "mediapipe",
                    "kind": "mediapipe",
                    "slot": label,
                    "label": f"MEDIAPIPE {label}",
                    "source": sources["mediapipe"],
                    "pair": pair,
                    "pair_label": label,
                    "result": _analyze_mediapipe_pair(mediapipe_df, hand_idx, pair, sources["mediapipe"]),
                }
            )

    return {
        "analysis": "multi_point_to_point",
        "sources": sources,
        "mano_pairs": mano_pairs,
        "mediapipe_pairs": mediapipe_pairs,
        "hand_idx": hand_idx,
        "wrist_joint_idx": wrist_joint_idx,
        "n_neighbors": n_neighbors,
        "entries": entries,
    }


def build_multi_point_figure(analysis_data, figsize_inches=(14, 9), dpi=100):
    fig, axes = plt.subplots(2, 1, figsize=figsize_inches, dpi=dpi, sharex=False)

    pair_order = []
    for entry in analysis_data["entries"]:
        if entry["pair_label"] not in pair_order:
            pair_order.append(entry["pair_label"])
    color_map = {pair_label: f"C{idx % 10}" for idx, pair_label in enumerate(pair_order)}

    for entry in analysis_data["entries"]:
        source_family = entry["source_family"]
        style = SOURCE_STYLE[source_family]
        color = color_map[entry["pair_label"]]
        source_label = source_family.upper()
        result = entry["result"]
        t = np.arange(len(result["magnitude"])) / FPS

        axes[0].plot(
            t,
            result["magnitude"],
            style,
            color=color,
            lw=1.5,
            label=f"{source_label} {entry['pair_label']}",
        )

        axes[1].semilogy(
            result["freqs"],
            result["psd"],
            style,
            color=color,
            lw=1.5,
            label=f"{source_label} {entry['pair_label']} ({result['dominant']:.2f} Hz)",
        )
        axes[1].axvline(result["dominant"], color=color, ls=":", alpha=0.35)

    axes[0].set_title("Filtered point-to-point displacement magnitude")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Displacement magnitude")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=2, fontsize="small", frameon=True)
    axes[0].grid(True)

    axes[1].set_title("Frequency spectrum of point-to-point motion")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power spectral density")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=2, fontsize="small", frameon=True)
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def main():
    analysis_data = run_multi_point_analysis()
    fig = build_multi_point_figure(analysis_data)
    plt.show()
    plt.close(fig)

    for entry in analysis_data["entries"]:
        print(
            f"{entry['source_family'].upper()} {entry['pair_label']}: "
            f"dominant={entry['result']['dominant']:.2f} Hz, "
            f"rms={entry['result']['rms']:.6f}"
        )


if __name__ == "__main__":
    main()

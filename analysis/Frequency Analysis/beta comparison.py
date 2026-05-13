import argparse
import importlib.util
from collections import deque
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from mano_pickle import load_mano_pickle
from npy_io import iter_model_frame_records


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


FPS = 30.0
LOWPASS_CUTOFF = 6.0
FILTER_ORDER = 3


@lru_cache(maxsize=1)
def _load_beta_average_module():
    module_path = PROJECT_ROOT / "3D Visualization " / "beta average.py"
    spec = importlib.util.spec_from_file_location("beta_average_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load beta-average helper module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_point_to_point_module():
    module_path = PROJECT_ROOT / "Frequency Analysis" / "Point to Point.py"
    spec = importlib.util.spec_from_file_location("point_to_point_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load point-to-point helper module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_mano_assets(mano_right_path):
    mano = load_mano_pickle(mano_right_path)
    j_reg = mano["J_regressor"]
    faces = np.asarray(mano["f"], dtype=np.int32)
    return j_reg, faces


def _resolve_mano_right_path(mano_model_path: str) -> Path:
    candidate = Path(mano_model_path).expanduser().resolve()
    candidates = []
    if candidate.is_dir():
        candidates.extend(
            [
                candidate / "MANO_RIGHT.pkl",
                candidate / "mano_data" / "MANO_RIGHT.pkl",
            ]
        )
    else:
        candidates.append(candidate)

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Could not find MANO_RIGHT.pkl under {mano_model_path}. "
        "Pass --mano_model_path pointing to the MANO asset directory or file."
    )


def _normalize_video_name(video_name: str) -> str:
    name = Path(str(video_name)).name
    if name.lower().endswith(".mp4"):
        return Path(name).stem
    return name


def _resolve_wilor_mesh_dir(wilor_root: str, video_name: str) -> Path:
    video_dir = Path(wilor_root).expanduser().resolve() / _normalize_video_name(video_name)
    mesh_dir = video_dir / "meshes"
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Could not find WiLoR mesh cache: {mesh_dir}")
    return mesh_dir


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
    return [sorted(list(neighbors)) for neighbors in adjacency]


def _select_graph_neighbors(seed, adjacency, n_neighbors):
    visited = {seed}
    queue = deque([(seed, 0)])
    ranked = []

    while queue:
        node, dist = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            next_dist = dist + 1
            ranked.append((next_dist, neighbor))
            queue.append((neighbor, next_dist))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [vertex_id for _, vertex_id in ranked[:n_neighbors]]


def _build_region_indices(seed, adjacency, n_neighbors):
    selected = _select_graph_neighbors(seed, adjacency, n_neighbors)
    if len(selected) < n_neighbors:
        print(
            f"[warn] seed={seed}: requested {n_neighbors} neighbors, got {len(selected)} available"
        )
    return np.asarray([seed] + selected, dtype=np.int32)


def _validate_config_indices(total_verts, vertex_a, vertex_b, n_neighbors):
    if not (0 <= vertex_a < total_verts):
        raise ValueError(f"MODEL_SPECIFIC_VERTEX_A={vertex_a} out of range [0, {total_verts - 1}]")
    if not (0 <= vertex_b < total_verts):
        raise ValueError(f"MODEL_SPECIFIC_VERTEX_B={vertex_b} out of range [0, {total_verts - 1}]")
    if vertex_a == vertex_b:
        raise ValueError("MODEL_SPECIFIC_VERTEX_A and MODEL_SPECIFIC_VERTEX_B must be different")
    if n_neighbors <= 0:
        raise ValueError(f"N_NEIGHBORS must be > 0, got {n_neighbors}")


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


def _extract_score(record):
    score_value = record.get("detection_confidence", 1.0)
    if score_value is None:
        return 1.0
    return float(score_value)


def _load_actual_model_frames(mesh_dir: Path):
    frames = []
    loaded_records = 0

    for frame_id, records in iter_model_frame_records(str(mesh_dir), pattern="*.npy"):
        frame_hands = []
        for record in records:
            frame_hands.append(
                {
                    "verts": np.asarray(record["verts_world"], dtype=np.float32),
                    "right": int(round(float(record["right"]))),
                    "score": _extract_score(record),
                }
            )
            loaded_records += 1

        if frame_hands:
            frames.append((int(frame_id), frame_hands))

    if not frames:
        raise RuntimeError(f"No WiLoR records found under {mesh_dir}")

    print(f"Loaded {len(frames)} raw-model frames ({loaded_records} hand records)")
    return frames


def _analyze_variant_frames(frames, hand_idx, region_a, region_b, label):
    trajectory = []
    used_frames = 0

    for _, frame_hands in frames:
        selected = [hand["verts"] for hand in frame_hands if int(hand["right"]) == int(hand_idx)]
        if not selected:
            continue

        frame_diffs = []
        for verts in selected:
            centroid_a = verts[region_a].mean(axis=0)
            centroid_b = verts[region_b].mean(axis=0)
            frame_diffs.append(centroid_a - centroid_b)

        trajectory.append(np.mean(np.stack(frame_diffs, axis=0), axis=0))
        used_frames += 1

    if not trajectory:
        raise RuntimeError(
            f"No usable frames for HAND_IDX={hand_idx} in variant '{label}'. "
            "Check hand side and input clip."
        )

    print(f"{label}: loaded {used_frames} usable frames")
    return _finish_analysis(trajectory)


def build_beta_comparison_figure(analysis_data, figsize_inches=(12, 10), dpi=100):
    point_to_point_module = _load_point_to_point_module()
    return point_to_point_module.build_point_to_point_figure(
        analysis_data,
        figsize_inches=figsize_inches,
        dpi=dpi,
    )


def run_beta_comparison_analysis(config_overrides=None):
    overrides = config_overrides or {}

    hand_idx = int(overrides.get("hand_idx", CONFIG.HAND_IDX))
    n_neighbors = int(overrides.get("n_neighbors", CONFIG.N_NEIGHBORS))
    vertex_a = int(overrides.get("vertex_a", CONFIG.MODEL_SPECIFIC_VERTEX_A))
    vertex_b = int(overrides.get("vertex_b", CONFIG.MODEL_SPECIFIC_VERTEX_B))
    video_name = str(overrides.get("video", "me 1.mp4"))
    wilor_root = str(overrides.get("wilor_root", Path(CONFIG.OUTPUTS_ROOT) / "wilor"))
    mano_model_path = str(overrides.get("mano_model_path", CONFIG.MANO_RIGHT_PATH))
    mano_right_path = _resolve_mano_right_path(mano_model_path)

    j_reg, faces = _load_mano_assets(str(mano_right_path))
    n_verts = int(j_reg.shape[1])
    _validate_config_indices(n_verts, vertex_a, vertex_b, n_neighbors)

    adjacency = _build_vertex_adjacency(n_verts, faces)
    region_a = _build_region_indices(vertex_a, adjacency, n_neighbors)
    region_b = _build_region_indices(vertex_b, adjacency, n_neighbors)

    mesh_dir = _resolve_wilor_mesh_dir(wilor_root, video_name)
    beta_average_module = _load_beta_average_module()

    actual_frames = _load_actual_model_frames(mesh_dir)
    beta_average_frames = beta_average_module.load_average_beta_frames(
        wilor_root=wilor_root,
        video_name=video_name,
        mano_model_path=str(mano_right_path),
        wrist_ground=False,
    )["frames"]

    entries = [
        {
            "label": "Actual model",
            "result": _analyze_variant_frames(actual_frames, hand_idx, region_a, region_b, "Actual model"),
        },
        {
            "label": "Beta average",
            "result": _analyze_variant_frames(
                beta_average_frames,
                hand_idx,
                region_a,
                region_b,
                "Beta average",
            ),
        },
    ]

    return {
        "analysis": "beta_comparison",
        "video": video_name,
        "mesh_dir": str(mesh_dir),
        "hand_idx": hand_idx,
        "vertex_a": vertex_a,
        "vertex_b": vertex_b,
        "n_neighbors": n_neighbors,
        "region_a": region_a,
        "region_b": region_b,
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run point-to-point frequency analysis on WiLoR for the raw model output, "
            "and the sequence-beta-averaged reconstruction."
        )
    )
    parser.add_argument(
        "--video",
        type=str,
        default="me 1.mp4",
        help="Video filename or stem. '.mp4' is stripped to match the WiLoR output folder name.",
    )
    parser.add_argument(
        "--wilor_root",
        type=str,
        default=str(Path(CONFIG.OUTPUTS_ROOT) / "wilor"),
        help="Root directory containing per-video WiLoR outputs.",
    )
    parser.add_argument(
        "--mano_model_path",
        type=str,
        default=str(CONFIG.MANO_RIGHT_PATH),
        help="Path to MANO assets or directly to MANO_RIGHT.pkl for beta-averaged reconstruction.",
    )
    parser.add_argument(
        "--hand_idx",
        type=int,
        default=int(CONFIG.HAND_IDX),
        help="1=right, 0=left.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=int(CONFIG.N_NEIGHBORS),
        help="Number of graph neighbors added around each seed vertex.",
    )
    parser.add_argument(
        "--vertex_a",
        type=int,
        default=int(CONFIG.MODEL_SPECIFIC_VERTEX_A),
        help="Seed vertex A for the point-to-point region.",
    )
    parser.add_argument(
        "--vertex_b",
        type=int,
        default=int(CONFIG.MODEL_SPECIFIC_VERTEX_B),
        help="Seed vertex B for the point-to-point region.",
    )
    parser.add_argument(
        "--save_png",
        type=str,
        default=None,
        help="Optional output .png path for the frequency figure.",
    )
    args = parser.parse_args()

    analysis_data = run_beta_comparison_analysis(
        {
            "video": args.video,
            "wilor_root": args.wilor_root,
            "mano_model_path": args.mano_model_path,
            "hand_idx": args.hand_idx,
            "n_neighbors": args.n_neighbors,
            "vertex_a": args.vertex_a,
            "vertex_b": args.vertex_b,
        }
    )

    print(f"Using WiLoR source: {analysis_data['mesh_dir']}")
    print(f"Using hand_idx={analysis_data['hand_idx']}")
    print(
        f"Region A indices: {analysis_data['region_a'].tolist()} | "
        f"Region B indices: {analysis_data['region_b'].tolist()}"
    )

    fig = build_beta_comparison_figure(analysis_data)

    if args.save_png:
        out_path = Path(args.save_png).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=fig.dpi, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()
    plt.close(fig)

    for entry in analysis_data["entries"]:
        label = entry["label"]
        result = entry["result"]
        print(f"{label} dominant frequency: {result['dominant']:.2f} Hz")
        print(f"{label} RMS amplitude: {result['rms']:.6f}")


if __name__ == "__main__":
    main()

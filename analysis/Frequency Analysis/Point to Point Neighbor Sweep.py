import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from analysis_metrics import finish_motion_analysis, subset_neighbor_pairs
from npy_io import iter_model_frame_records


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_POINT_COUNTS = tuple(range(10, 201, 10))


def _load_point_to_point_module():
    module_path = THIS_DIR / "Point to Point.py"
    spec = importlib.util.spec_from_file_location("point_to_point_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load point-to-point module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_point_counts(overrides):
    explicit = overrides.get("point_counts", None)
    if explicit is not None:
        point_counts = [int(value) for value in explicit]
    else:
        min_points = int(overrides.get("min_points", 10))
        max_points = int(overrides.get("max_points", 200))
        step_points = int(overrides.get("step_points", 10))
        if min_points <= 0 or max_points <= 0 or step_points <= 0:
            raise ValueError("min_points, max_points, and step_points must be positive integers.")
        if min_points > max_points:
            raise ValueError("min_points must be <= max_points.")
        point_counts = list(range(min_points, max_points + 1, step_points))

    resolved = []
    seen = set()
    for point_count in point_counts:
        value = int(point_count)
        if value <= 0:
            raise ValueError(f"Point counts must be positive integers, got {value}.")
        if value in seen:
            continue
        seen.add(value)
        resolved.append(value)

    if not resolved:
        raise ValueError("No point counts resolved for the neighbor sweep analysis.")
    return resolved


def _collect_model_frames(root_dir, j_reg, hand_idx, wrist_joint_idx, n_verts):
    frames = []
    for _, records in iter_model_frame_records(root_dir):
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

    usable_frames = 0
    for frame_hands in frames:
        if any(int(hand["is_right"]) == int(hand_idx) for hand in frame_hands):
            usable_frames += 1

    if usable_frames == 0:
        raise RuntimeError(
            f"No valid frames for HAND_IDX={hand_idx} under {root_dir}. "
            "Check hand side and input clip."
        )

    print(f"Loaded {len(frames)} raw frames and {usable_frames} usable handedness-matched frames for: {root_dir}")
    return frames


def _analyze_model_frames(
    frames,
    hand_idx,
    region_a,
    region_b,
    region_vertices,
    coherence_pairs,
    point_module,
    source_path,
    point_count,
):
    trajectory = []
    coherence_frames = []
    for frame_hands in frames:
        selected = [hand["verts"] for hand in frame_hands if int(hand["is_right"]) == int(hand_idx)]
        if not selected:
            continue

        frame_diffs = []
        for verts in selected:
            centroid_a = verts[region_a].mean(axis=0)
            centroid_b = verts[region_b].mean(axis=0)
            frame_diffs.append(centroid_a - centroid_b)

        trajectory.append(np.mean(np.stack(frame_diffs, axis=0), axis=0))
        coherence_frames.append(np.mean(np.stack([verts[region_vertices] for verts in selected], axis=0), axis=0))

    if not trajectory:
        raise RuntimeError(
            f"No usable trajectory remained for HAND_IDX={hand_idx} under {source_path} at point_count={point_count}."
        )

    return finish_motion_analysis(
        trajectory,
        fps=point_module.FPS,
        filter_kind="lowpass",
        filter_order=point_module.FILTER_ORDER,
        lowpass_cutoff_hz=point_module.LOWPASS_CUTOFF,
        psd_nperseg=256,
        coherence_positions=np.stack(coherence_frames, axis=0),
        coherence_pairs=coherence_pairs,
    )


def _legend_with_frequency_range(label, dominant_values):
    dominant_values = [float(value) for value in dominant_values]
    min_value = min(dominant_values)
    max_value = max(dominant_values)
    if abs(max_value - min_value) < 1e-9:
        return f"{label} ({min_value:.2f} Hz)"
    return f"{label} ({min_value:.2f}-{max_value:.2f} Hz)"


def run_point_to_point_neighbor_sweep_analysis(config_overrides=None):
    overrides = config_overrides or {}
    point_module = _load_point_to_point_module()
    entries = point_module._resolve_entries(overrides)
    if not entries:
        raise ValueError("No sources resolved for point-to-point neighbor sweep.")

    if any(entry["kind"] != "model" for entry in entries):
        raise ValueError("Point-to-point neighbor sweep supports only model sources, not MediaPipe CSV inputs.")

    hand_idx = int(overrides.get("hand_idx", CONFIG.HAND_IDX))
    wrist_joint_idx = int(overrides.get("wrist_joint_idx", CONFIG.WRIST_JOINT_IDX))
    vertex_a = int(overrides.get("vertex_a", CONFIG.MODEL_SPECIFIC_VERTEX_A))
    vertex_b = int(overrides.get("vertex_b", CONFIG.MODEL_SPECIFIC_VERTEX_B))
    point_counts = _resolve_point_counts(overrides)

    try:
        j_reg, faces = point_module._load_mano_assets(str(CONFIG.MANO_RIGHT_PATH))
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Neighbor sweep MANO analysis needs MANO pickle dependencies (commonly 'chumpy')."
        ) from exc

    n_verts = int(j_reg.shape[1])
    max_points = max(point_counts)
    max_neighbors = max_points - 1
    point_module._validate_config_indices(n_verts, vertex_a, vertex_b, max_neighbors)
    adjacency = point_module._build_vertex_adjacency(n_verts, faces)

    frames_by_source = {}
    for entry in entries:
        frames_by_source[entry["source"]] = _collect_model_frames(
            entry["source"],
            j_reg,
            hand_idx,
            wrist_joint_idx,
            n_verts,
        )

    series_by_source = {}
    for entry in entries:
        series_by_source[entry["source"]] = {
            "slot": entry.get("slot", ""),
            "label": entry["label"],
            "source": entry["source"],
            "kind": entry["kind"],
            "series": [],
        }

    for point_count in point_counts:
        n_neighbors = point_count - 1
        region_a = point_module._build_region_indices(vertex_a, adjacency, n_neighbors)
        region_b = point_module._build_region_indices(vertex_b, adjacency, n_neighbors)
        region_vertices = np.asarray(sorted(set(region_a.tolist()) | set(region_b.tolist())), dtype=np.int32)
        coherence_pairs = subset_neighbor_pairs(region_vertices.tolist(), adjacency)
        print(
            f"Neighbor sweep point_count={point_count}: region A size={len(region_a)}, region B size={len(region_b)}"
        )

        for entry in entries:
            result = _analyze_model_frames(
                frames_by_source[entry["source"]],
                hand_idx,
                region_a,
                region_b,
                region_vertices,
                coherence_pairs,
                point_module,
                entry["source"],
                point_count,
            )
            series_by_source[entry["source"]]["series"].append(
                {
                    "point_count": point_count,
                    "n_neighbors": n_neighbors,
                    "dominant": float(result["dominant"]),
                    "rms": float(result["rms"]),
                    "peak_ratio": float(result["peak_ratio"]),
                    "peak_sharpness": float(result["peak_sharpness"]),
                    "temporal_noise": float(result["temporal_noise"]),
                    "spatial_coherence": result["spatial_coherence"],
                    "num_samples": int(len(result["magnitude"])),
                }
            )

    resolved_entries = []
    for entry in entries:
        item = series_by_source[entry["source"]]
        item["series"].sort(key=lambda row: row["point_count"])
        resolved_entries.append(item)

    return {
        "analysis": "point_to_point_neighbor_sweep",
        "hand_idx": hand_idx,
        "wrist_joint_idx": wrist_joint_idx,
        "vertex_a": vertex_a,
        "vertex_b": vertex_b,
        "point_counts": point_counts,
        "entries": resolved_entries,
    }


def build_point_to_point_neighbor_sweep_figure(analysis_data, figsize_inches=(12, 9), dpi=100):
    fig, axes = plt.subplots(2, 1, figsize=figsize_inches, dpi=dpi, sharex=True)
    point_counts = analysis_data["point_counts"]
    point_module = _load_point_to_point_module()

    for index, entry in enumerate(analysis_data["entries"]):
        style = point_module.LINE_STYLES[index % len(point_module.LINE_STYLES)]
        color = f"C{index % 10}"
        dominant_values = [row["dominant"] for row in entry["series"]]
        rms_values = [row["rms"] for row in entry["series"]]
        label_with_range = _legend_with_frequency_range(entry["label"], dominant_values)

        axes[0].plot(
            point_counts,
            dominant_values,
            linestyle=style,
            color=color,
            marker="o",
            lw=1.5,
            label=label_with_range,
        )
        axes[1].plot(
            point_counts,
            rms_values,
            linestyle=style,
            color=color,
            marker="o",
            lw=1.5,
            label=label_with_range,
        )

    axes[0].set_title("Dominant frequency vs averaged region size")
    axes[0].set_ylabel("Dominant frequency (Hz)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("RMS amplitude vs averaged region size")
    axes[1].set_xlabel("Averaged points per region")
    axes[1].set_ylabel("RMS amplitude")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep point-to-point frequency analysis across increasing averaged region sizes "
            "from model sources."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Repeatable model source path. If omitted, the default point-to-point sources are used.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional repeatable label matching --source order.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Use the configured all-model point-to-point source set. MediaPipe inputs are not allowed here.",
    )
    parser.add_argument("--min-points", type=int, default=10, help="Minimum averaged points per region.")
    parser.add_argument("--max-points", type=int, default=200, help="Maximum averaged points per region.")
    parser.add_argument("--step-points", type=int, default=10, help="Step size for averaged points per region.")
    parser.add_argument(
        "--save_png",
        type=str,
        default=None,
        help="Optional output .png path for the neighbor-sweep figure.",
    )
    args = parser.parse_args()

    overrides = {
        "all_models": bool(args.all_models),
        "min_points": int(args.min_points),
        "max_points": int(args.max_points),
        "step_points": int(args.step_points),
    }
    if args.source:
        overrides["sources"] = list(args.source)
    if args.label:
        overrides["labels"] = list(args.label)

    analysis_data = run_point_to_point_neighbor_sweep_analysis(overrides)
    fig = build_point_to_point_neighbor_sweep_figure(analysis_data)

    if args.save_png:
        out_path = Path(args.save_png).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=fig.dpi, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()
    plt.close(fig)

    for entry in analysis_data["entries"]:
        print(entry["label"])
        for row in entry["series"]:
            print(
                f"  points={row['point_count']}: dominant={row['dominant']:.2f} Hz, "
                f"rms={row['rms']:.6f}, samples={row['num_samples']}"
            )


if __name__ == "__main__":
    main()

import argparse
import importlib.util
from collections import deque
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from analysis_metrics import finish_motion_analysis, subset_neighbor_pairs
from analysis_plotting import build_time_psd_metrics_figure
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
THIS_DIR = Path(__file__).resolve().parent
VARIANT_MODES = {"single_source_compare", "beta_only", "raw_plus_beta"}
VARIANT_LINESTYLES = {
    "raw": "-",
    "beta": "--",
}


@lru_cache(maxsize=1)
def _load_beta_average_module():
    module_path = THIS_DIR.parent / "3D Visualization " / "beta average.py"
    spec = importlib.util.spec_from_file_location("beta_average_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load beta-average helper module from: {module_path}")

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


def _resolve_model_source_path(source_path, wilor_root: str, video_name: str) -> Path:
    if source_path is not None:
        return Path(source_path).expanduser().resolve()
    return _resolve_wilor_mesh_dir(wilor_root, video_name)


def _source_label(source_path: Path) -> str:
    path = Path(source_path).expanduser().resolve()
    if path.name == "meshes":
        parent = path.parent
        if parent.parent.parent.name == "wilor_finetune":
            return f"{parent.parent.name}/{parent.name}"
        return parent.name
    return path.name


def _normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


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


def _hand_name_from_idx(hand_idx):
    return "right" if int(hand_idx) == 1 else "left"


def _resolve_source_entries(overrides):
    sources_override = overrides.get("sources", None)
    labels_override = overrides.get("labels", None)
    source_path_override = overrides.get("source_path", None)

    if sources_override is not None:
        raw_sources = list(sources_override)
    else:
        raw_sources = [source_path_override]

    normalized_sources = [_normalize_optional_path(source) for source in raw_sources]
    if not any(source is not None for source in normalized_sources):
        video_name = str(overrides.get("video", "me 1.mp4"))
        wilor_root = str(overrides.get("wilor_root", Path(CONFIG.OUTPUTS_ROOT) / "wilor"))
        normalized_sources = [str(_resolve_model_source_path(None, wilor_root, video_name))]

    if labels_override is not None:
        raw_labels = list(labels_override)
        if len(raw_labels) != len(normalized_sources):
            raise ValueError("labels length must match sources length when both are provided.")
    else:
        raw_labels = []
        for source in normalized_sources:
            raw_labels.append(_source_label(Path(source)))

    entries = []
    slot_names = ("A", "B", "C", "D", "E", "F")
    for index, source in enumerate(normalized_sources):
        if source is None:
            continue
        entries.append(
            {
                "slot": slot_names[index] if index < len(slot_names) else f"S{index + 1}",
                "source": str(Path(source).expanduser().resolve()),
                "label": raw_labels[index] if index < len(raw_labels) else _source_label(Path(source)),
            }
        )
    return entries


def _load_variant_frame_sets(source_path: str, mano_right_path: str, hand_idx: int):
    beta_average_module = _load_beta_average_module()
    actual_frames = _load_actual_model_frames(Path(source_path))
    beta_average_bundle = beta_average_module.load_average_beta_frames_for_source(
        source_path=source_path,
        mano_model_path=str(mano_right_path),
        wrist_ground=False,
        hand=_hand_name_from_idx(hand_idx),
    )
    return actual_frames, beta_average_bundle["frames"], beta_average_bundle


def _analyze_variant_frames(
    frames,
    hand_idx,
    region_a,
    region_b,
    label,
    region_vertices=None,
    coherence_pairs=None,
):
    trajectory = []
    coherence_frames = []
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
        if region_vertices is not None:
            coherence_frames.append(np.mean(np.stack([verts[region_vertices] for verts in selected], axis=0), axis=0))
        used_frames += 1

    if not trajectory:
        raise RuntimeError(
            f"No usable frames for HAND_IDX={hand_idx} in variant '{label}'. "
            "Check hand side and input clip."
        )

    print(f"{label}: loaded {used_frames} usable frames")
    return finish_motion_analysis(
        trajectory,
        fps=FPS,
        filter_kind="lowpass",
        filter_order=FILTER_ORDER,
        lowpass_cutoff_hz=LOWPASS_CUTOFF,
        psd_nperseg=256,
        coherence_positions=(
            np.stack(coherence_frames, axis=0)
            if region_vertices is not None and coherence_frames
            else None
        ),
        coherence_pairs=coherence_pairs,
    )


def _build_variant_entries(
    source_entries,
    hand_idx,
    region_a,
    region_b,
    region_vertices,
    coherence_pairs,
    mano_right_path,
    variant_mode,
):
    entries = []
    bundles = []
    for source_entry in source_entries:
        actual_frames, beta_frames, beta_bundle = _load_variant_frame_sets(
            source_entry["source"],
            mano_right_path=str(mano_right_path),
            hand_idx=hand_idx,
        )
        bundles.append({"source": source_entry["source"], "record_root": str(beta_bundle["record_root"])})

        if variant_mode == "single_source_compare":
            variants = [
                ("Actual model", "raw", actual_frames, "Actual model"),
                ("Beta average", "beta", beta_frames, "Beta average"),
            ]
        elif variant_mode == "beta_only":
            variants = [
                (f"{source_entry['label']} beta avg", "beta", beta_frames, f"{source_entry['label']} beta avg"),
            ]
        elif variant_mode == "raw_plus_beta":
            variants = [
                (f"{source_entry['label']} raw", "raw", actual_frames, f"{source_entry['label']} raw"),
                (f"{source_entry['label']} beta avg", "beta", beta_frames, f"{source_entry['label']} beta avg"),
            ]
        else:
            raise ValueError(f"Unsupported beta comparison variant mode: {variant_mode}")

        for label, variant, frames, trace_label in variants:
            slot = source_entry["slot"]
            if variant_mode == "single_source_compare":
                entry_slot = label
            elif variant_mode == "beta_only":
                entry_slot = slot
            else:
                entry_slot = f"{slot}:{variant}"
            entries.append(
                {
                    "slot": entry_slot,
                    "label": label,
                    "plot_label": trace_label,
                    "variant": variant,
                    "source_slot": slot,
                    "source": source_entry["source"],
                    "kind": "model",
                    "result": _analyze_variant_frames(
                        frames,
                        hand_idx,
                        region_a,
                        region_b,
                        label,
                        region_vertices=region_vertices,
                        coherence_pairs=coherence_pairs,
                    ),
                }
            )

    return entries, bundles


def build_beta_comparison_figure(analysis_data, figsize_inches=(12, 10), dpi=100):
    source_slots = []
    for entry in analysis_data["entries"]:
        slot = entry.get("source_slot", entry.get("slot", ""))
        if slot not in source_slots:
            source_slots.append(slot)
    color_map = {slot: f"C{index % 10}" for index, slot in enumerate(source_slots)}

    title_time = "Raw vs beta-average comparison"
    title_psd = "Frequency spectrum of beta comparison"
    if analysis_data.get("variant_mode") == "beta_only":
        title_time = "Beta-average comparison across model sources"
        title_psd = "Frequency spectrum of beta-average model comparison"
    elif analysis_data.get("variant_mode") == "raw_plus_beta":
        title_time = "Raw and beta-average comparison across model sources"
        title_psd = "Frequency spectrum of raw and beta-average model comparison"

    return build_time_psd_metrics_figure(
        analysis_data["entries"],
        fps=FPS,
        title_time=title_time,
        title_psd=title_psd,
        figsize_inches=figsize_inches,
        dpi=dpi,
        style_resolver=lambda _index, entry: {
            "color": color_map[entry.get("source_slot", entry.get("slot", ""))],
            "linestyle": VARIANT_LINESTYLES.get(entry.get("variant", "raw"), "-"),
            "linewidth": 1.5,
        },
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
    variant_mode = str(overrides.get("variant_mode", "single_source_compare")).strip().lower()
    if variant_mode not in VARIANT_MODES:
        valid = ", ".join(sorted(VARIANT_MODES))
        raise ValueError(f"Unsupported variant_mode '{variant_mode}'. Valid options: {valid}")
    mano_right_path = _resolve_mano_right_path(mano_model_path)

    j_reg, faces = _load_mano_assets(str(mano_right_path))
    n_verts = int(j_reg.shape[1])
    _validate_config_indices(n_verts, vertex_a, vertex_b, n_neighbors)

    adjacency = _build_vertex_adjacency(n_verts, faces)
    region_a = _build_region_indices(vertex_a, adjacency, n_neighbors)
    region_b = _build_region_indices(vertex_b, adjacency, n_neighbors)
    region_vertices = np.asarray(sorted(set(region_a.tolist()) | set(region_b.tolist())), dtype=np.int32)
    coherence_pairs = subset_neighbor_pairs(region_vertices.tolist(), adjacency)
    source_entries = _resolve_source_entries(overrides)
    if variant_mode == "single_source_compare" and len(source_entries) != 1:
        raise ValueError("single_source_compare mode requires exactly one source.")

    entries, bundles = _build_variant_entries(
        source_entries,
        hand_idx=hand_idx,
        region_a=region_a,
        region_b=region_b,
        region_vertices=region_vertices,
        coherence_pairs=coherence_pairs,
        mano_right_path=mano_right_path,
        variant_mode=variant_mode,
    )
    primary_source = Path(source_entries[0]["source"]).expanduser().resolve()

    return {
        "analysis": str(overrides.get("analysis_id", "beta_comparison")),
        "video": video_name,
        "variant_mode": variant_mode,
        "source_path": str(primary_source),
        "source_paths": [entry["source"] for entry in source_entries],
        "source_label": _source_label(primary_source),
        "source_labels": [entry["label"] for entry in source_entries],
        "mesh_dir": str(primary_source),
        "hand_idx": hand_idx,
        "vertex_a": vertex_a,
        "vertex_b": vertex_b,
        "n_neighbors": n_neighbors,
        "region_a": region_a,
        "region_b": region_b,
        "record_root": bundles[0]["record_root"],
        "record_roots": bundles,
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run point-to-point frequency analysis on a compatible saved model source for the raw "
            "model output and the sequence-beta-averaged reconstruction."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Direct path to a compatible model source. This can be a mesh cache directory or a stride clip root.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="clip_2.mp4",
        help="Video filename or stem used with --wilor_root when --source is not provided.",
    )
    parser.add_argument(
        "--wilor_root",
        type=str,
        default=str(Path(CONFIG.OUTPUTS_ROOT) / "wilor"),
        help="Root directory containing per-video WiLoR outputs when --source is not provided.",
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
            "source_path": args.source,
            "video": args.video,
            "wilor_root": args.wilor_root,
            "mano_model_path": args.mano_model_path,
            "hand_idx": args.hand_idx,
            "n_neighbors": args.n_neighbors,
            "vertex_a": args.vertex_a,
            "vertex_b": args.vertex_b,
        }
    )

    print(f"Using source: {analysis_data['source_path']}")
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

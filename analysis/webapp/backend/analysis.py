from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_metrics import finish_motion_analysis, subset_neighbor_pairs
from analysis_plotting import build_time_psd_metrics_figure
from .loaders import extract_beta_average_records, figure_to_svg, load_source_frames
from .mano import load_mano_assets
from .settings import AppSettings, hand_label_to_value, hand_value_to_label, parse_number_list, parse_pair_text


def _candidate_hand_values(hand: str) -> List[int]:
    normalized = str(hand).strip().lower()
    if normalized == "auto":
        return [1, 0, -1]
    return [hand_label_to_value(normalized)]


def _run_with_hand_fallback(hand: str, fn):
    errors = []
    for hand_value in _candidate_hand_values(hand):
        try:
            return hand_value, fn(hand_value)
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(errors[-1])
    raise RuntimeError("No usable hands found for the selected sources.")


def _build_vertex_adjacency(total_verts: int, faces: np.ndarray) -> List[List[int]]:
    adjacency = [set() for _ in range(total_verts)]
    for tri in faces:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        adjacency[a].add(b)
        adjacency[a].add(c)
        adjacency[b].add(a)
        adjacency[b].add(c)
        adjacency[c].add(a)
        adjacency[c].add(b)
    return [sorted(list(neighbors)) for neighbors in adjacency]


def _select_graph_neighbors(seed: int, adjacency: List[List[int]], n_neighbors: int) -> List[int]:
    visited = {seed}
    queue = deque([(seed, 0)])
    ranked = []
    while queue:
        node, distance = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            ranked.append((distance + 1, neighbor))
            queue.append((neighbor, distance + 1))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [vertex_id for _, vertex_id in ranked[:n_neighbors]]


def _build_region_indices(seed: int, adjacency: List[List[int]], n_neighbors: int) -> np.ndarray:
    return np.asarray([seed] + _select_graph_neighbors(seed, adjacency, n_neighbors), dtype=np.int32)


def _full_mesh_neighbor_pairs(settings: AppSettings) -> list[tuple[int, int]]:
    mano = load_mano_assets(str(settings.mano_right_path))
    adjacency = _build_vertex_adjacency(int(mano["j_regressor"].shape[1]), mano["faces_right"])
    return subset_neighbor_pairs(range(int(mano["j_regressor"].shape[1])), adjacency)


def _build_region_metadata(
    n_neighbors: int,
    vertex_a: int,
    vertex_b: int,
    settings: AppSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    mano = load_mano_assets(str(settings.mano_right_path))
    adjacency = _build_vertex_adjacency(int(mano["j_regressor"].shape[1]), mano["faces_right"])
    region_a = _build_region_indices(vertex_a, adjacency, n_neighbors)
    region_b = _build_region_indices(vertex_b, adjacency, n_neighbors)
    region_vertices = np.asarray(sorted(set(region_a.tolist()) | set(region_b.tolist())), dtype=np.int32)
    coherence_pairs = subset_neighbor_pairs(region_vertices.tolist(), adjacency)
    return region_a, region_b, region_vertices, coherence_pairs


def _selected_hands(frame: dict, hand_value: int) -> List[dict]:
    return [hand for hand in frame["hands"] if int(hand["right"]) == int(hand_value)]


def _analyze_centroid(source: dict, frames: List[dict], hand_value: int, wrist_joint_idx: int, fps: float) -> dict:
    centroids = []
    coherence_frames = []
    for frame in frames:
        hands = _selected_hands(frame, hand_value)
        if not hands:
            continue
        if source["family"] == "mediapipe":
            merged = np.concatenate([hand["points"] for hand in hands], axis=0)
            centroids.append(merged.mean(axis=0))
            continue
        centered = []
        for hand in hands:
            wrist = np.asarray(hand["joints_world"][int(wrist_joint_idx)], dtype=np.float32)
            centered.append(hand["verts_world"] - wrist[None, :])
        merged = np.concatenate(centered, axis=0)
        centroids.append(merged.mean(axis=0))
        coherence_frames.append(np.mean(np.stack(centered, axis=0), axis=0))
    if not centroids:
        raise RuntimeError(f"No usable centroid frames for source {source['id']}.")
    return finish_motion_analysis(
        centroids,
        fps=fps,
        filter_kind="bandpass",
        filter_order=3,
        band_low_hz=2.0,
        band_high_hz=12.0,
        psd_nperseg=512,
        coherence_positions=np.stack(coherence_frames, axis=0) if coherence_frames else None,
        coherence_pairs=None if source["family"] == "mediapipe" else _full_mesh_neighbor_pairs(_analyze_centroid.settings),
    )


_analyze_centroid.settings = None


def _analyze_point_to_point(source: dict, frames: List[dict], hand_value: int, wrist_joint_idx: int, fps: float, vertex_a: int, vertex_b: int, n_neighbors: int, mediapipe_point_a: int, mediapipe_point_b: int) -> dict:
    trajectory = []
    if source["family"] == "mediapipe":
        for frame in frames:
            hands = _selected_hands(frame, hand_value)
            if not hands:
                continue
            frame_diffs = []
            for hand in hands:
                points = np.asarray(hand["points"], dtype=np.float32)
                if max(mediapipe_point_a, mediapipe_point_b) >= len(points):
                    continue
                frame_diffs.append(points[mediapipe_point_a] - points[mediapipe_point_b])
            if frame_diffs:
                trajectory.append(np.mean(np.stack(frame_diffs, axis=0), axis=0))
        if not trajectory:
            raise RuntimeError(f"No usable MediaPipe point-to-point frames for source {source['id']}.")
        return finish_motion_analysis(
            trajectory,
            fps=fps,
            filter_kind="lowpass",
            filter_order=3,
            lowpass_cutoff_hz=6.0,
            psd_nperseg=256,
        )

    region_a, region_b, region_vertices, coherence_pairs = _build_region_metadata(
        n_neighbors,
        vertex_a,
        vertex_b,
        settings=_analyze_point_to_point.settings,
    )
    coherence_frames = []
    for frame in frames:
        hands = _selected_hands(frame, hand_value)
        if not hands:
            continue
        frame_diffs = []
        selected_regions = []
        for hand in hands:
            wrist = np.asarray(hand["joints_world"][int(wrist_joint_idx)], dtype=np.float32)
            centered = np.asarray(hand["verts_world"], dtype=np.float32) - wrist[None, :]
            frame_diffs.append(centered[region_a].mean(axis=0) - centered[region_b].mean(axis=0))
            selected_regions.append(centered[region_vertices])
        trajectory.append(np.mean(np.stack(frame_diffs, axis=0), axis=0))
        coherence_frames.append(np.mean(np.stack(selected_regions, axis=0), axis=0))
    if not trajectory:
        raise RuntimeError(f"No usable point-to-point model frames for source {source['id']}.")
    return finish_motion_analysis(
        trajectory,
        fps=fps,
        filter_kind="lowpass",
        filter_order=3,
        lowpass_cutoff_hz=6.0,
        psd_nperseg=256,
        coherence_positions=np.stack(coherence_frames, axis=0),
        coherence_pairs=coherence_pairs,
    )


_analyze_point_to_point.settings = None


def _analyze_multi_point(source: dict, frames: List[dict], hand_value: int, wrist_joint_idx: int, fps: float, n_neighbors: int, mano_pairs: Sequence[Tuple[int, int]], mediapipe_pairs: Sequence[Tuple[int, int]]) -> List[dict]:
    entries = []
    if source["family"] == "mediapipe":
        for pair in mediapipe_pairs:
            result = _analyze_point_to_point(
                source=source,
                frames=frames,
                hand_value=hand_value,
                wrist_joint_idx=wrist_joint_idx,
                fps=fps,
                vertex_a=0,
                vertex_b=1,
                n_neighbors=n_neighbors,
                mediapipe_point_a=int(pair[0]),
                mediapipe_point_b=int(pair[1]),
            )
            entries.append({"pair_label": f"j{pair[0]}-j{pair[1]}", "result": result})
        return entries

    for pair in mano_pairs:
        result = _analyze_point_to_point(
            source=source,
            frames=frames,
            hand_value=hand_value,
            wrist_joint_idx=wrist_joint_idx,
            fps=fps,
            vertex_a=int(pair[0]),
            vertex_b=int(pair[1]),
            n_neighbors=n_neighbors,
            mediapipe_point_a=0,
            mediapipe_point_b=1,
        )
        entries.append({"pair_label": f"v{pair[0]}-v{pair[1]}", "result": result})
    return entries


def _camera_space_frames(source: dict, settings: AppSettings, frame_start: Optional[int], frame_end: Optional[int], wrist_joint_idx: int) -> List[dict]:
    return load_source_frames(
        source,
        settings=settings,
        frame_start=frame_start,
        frame_end=frame_end,
        include_camera_space=True,
        wrist_joint_idx=wrist_joint_idx,
    )


def _analyze_camera_space(source: dict, frames: List[dict], hand_value: int, fps: float, n_neighbors: int, mano_pairs: Sequence[Tuple[int, int]], settings: AppSettings) -> List[dict]:
    region_map = [_build_region_metadata(n_neighbors, pair[0], pair[1], settings) for pair in mano_pairs]
    entries = []
    for pair, (region_a, region_b, region_vertices, coherence_pairs) in zip(mano_pairs, region_map):
        trajectory = []
        coherence_frames = []
        for frame in frames:
            hands = _selected_hands(frame, hand_value)
            if not hands:
                continue
            frame_diffs = []
            selected_regions = []
            for hand in hands:
                verts = np.asarray(hand["verts_camera_centered"], dtype=np.float32)
                frame_diffs.append(verts[region_a].mean(axis=0) - verts[region_b].mean(axis=0))
                selected_regions.append(verts[region_vertices])
            trajectory.append(np.mean(np.stack(frame_diffs, axis=0), axis=0))
            coherence_frames.append(np.mean(np.stack(selected_regions, axis=0), axis=0))
        result = finish_motion_analysis(
            trajectory,
            fps=fps,
            filter_kind="lowpass",
            filter_order=3,
            lowpass_cutoff_hz=6.0,
            psd_nperseg=256,
            coherence_positions=np.stack(coherence_frames, axis=0),
            coherence_pairs=coherence_pairs,
        )
        entries.append({"pair_label": f"v{pair[0]}-v{pair[1]}", "result": result})
    return entries


def _beta_variant_entries(source: dict, settings: AppSettings, hand_value: int, wrist_joint_idx: int, vertex_a: int, vertex_b: int, n_neighbors: int, fps: float, frame_start: Optional[int], frame_end: Optional[int]) -> List[dict]:
    region_a, region_b, region_vertices, coherence_pairs = _build_region_metadata(n_neighbors, vertex_a, vertex_b, settings)
    actual_frames = load_source_frames(source, settings=settings, frame_start=frame_start, frame_end=frame_end)
    beta_bundle = extract_beta_average_records(source, settings=settings, hand_filter=hand_value)
    beta_frames = [
        frame
        for frame in beta_bundle["frames"]
        if (frame_start is None or frame["frame_id"] >= frame_start) and (frame_end is None or frame["frame_id"] <= frame_end)
    ]
    entries = []
    for label, frames in [("Actual model", actual_frames), ("Beta average", beta_frames)]:
        trajectory = []
        coherence_frames = []
        for frame in frames:
            hands = _selected_hands(frame, hand_value)
            if not hands:
                continue
            frame_diffs = []
            selected_regions = []
            for hand in hands:
                verts_world = np.asarray(hand["verts_world"], dtype=np.float32)
                frame_diffs.append(verts_world[region_a].mean(axis=0) - verts_world[region_b].mean(axis=0))
                selected_regions.append(verts_world[region_vertices])
            trajectory.append(np.mean(np.stack(frame_diffs, axis=0), axis=0))
            coherence_frames.append(np.mean(np.stack(selected_regions, axis=0), axis=0))
        entries.append(
            {
                "label": label,
                "result": finish_motion_analysis(
                    trajectory,
                    fps=fps,
                    filter_kind="lowpass",
                    filter_order=3,
                    lowpass_cutoff_hz=6.0,
                    psd_nperseg=256,
                    coherence_positions=np.stack(coherence_frames, axis=0),
                    coherence_pairs=coherence_pairs,
                ),
            }
        )
    return entries


def _serialize_result(result: dict, fps: float) -> dict:
    t = np.arange(len(result["magnitude"]), dtype=np.float32) / float(fps)
    return {
        "dominant_hz": float(result["dominant"]),
        "peak_ratio": float(result["peak_ratio"]),
        "peak_sharpness": float(result["peak_sharpness"]),
        "temporal_noise": float(result["temporal_noise"]),
        "spatial_coherence": None if result["spatial_coherence"] is None else float(result["spatial_coherence"]),
        "rms_amplitude": float(result["rms"]),
        "sample_count": int(len(result["magnitude"])),
        "plots": {
            "time_s": t.tolist(),
            "magnitude": np.asarray(result["magnitude"], dtype=np.float32).tolist(),
            "freqs_hz": np.asarray(result["freqs"], dtype=np.float32).tolist(),
            "psd": np.asarray(result["psd"], dtype=np.float32).tolist(),
            "filtered_xyz": np.asarray(result["filtered"], dtype=np.float32).tolist(),
        },
    }


def _compare_figure(title: str, entries: List[dict], fps: float, style_resolver=None):
    return build_time_psd_metrics_figure(
        entries,
        fps=fps,
        title_time=title,
        title_psd="Power spectral density",
        figsize_inches=(12, 10),
        dpi=100,
        style_resolver=style_resolver,
    )


def _multi_pair_figure(title: str, entries: List[dict], fps: float):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), dpi=100)
    for index, entry in enumerate(entries):
        result = entry["result"]
        t = np.arange(len(result["magnitude"])) / float(fps)
        color = f"C{index % 10}"
        label = f"{entry['label']} ({result['dominant']:.2f} Hz)"
        axes[0].plot(t, result["magnitude"], color=color, lw=1.5, label=label)
        axes[1].semilogy(result["freqs"], result["psd"], color=color, lw=1.5, label=label)
        axes[1].axvline(result["dominant"], color=color, ls=":", alpha=0.35)
    axes[0].set_title(title)
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    axes[1].set_title("Power spectral density")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].grid(True)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    fig.tight_layout()
    return fig


def _sweep_figure(entries: List[dict]):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=100, sharex=True)
    for index, entry in enumerate(entries):
        color = f"C{index % 10}"
        counts = [row["point_count"] for row in entry["series"]]
        dominant = [row["dominant_hz"] for row in entry["series"]]
        rms = [row["rms_amplitude"] for row in entry["series"]]
        axes[0].plot(counts, dominant, color=color, marker="o", lw=1.5, label=entry["label"])
        axes[1].plot(counts, rms, color=color, marker="o", lw=1.5, label=entry["label"])
    axes[0].set_title("Dominant frequency vs region size")
    axes[0].set_ylabel("Dominant frequency (Hz)")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].set_title("RMS amplitude vs region size")
    axes[1].set_xlabel("Point count")
    axes[1].set_ylabel("RMS amplitude")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    return fig


class AnalysisService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        _analyze_centroid.settings = settings
        _analyze_point_to_point.settings = settings

    def run(self, mode_id: str, sources: List[dict], frame_ranges: Dict[str, Tuple[Optional[int], Optional[int]]], params: dict) -> Tuple[dict, str]:
        requested_hand = str(params.get("hand", self.settings.default_hand)).strip().lower()
        fps = float(params.get("fps", self.settings.default_fps))
        wrist_joint_idx = int(params.get("wrist_joint_idx", self.settings.default_wrist_joint_index))

        if mode_id == "centroid_compare":
            entries = []
            for source in sources:
                frame_start, frame_end = frame_ranges[source["id"]]
                frames = load_source_frames(source, settings=self.settings, frame_start=frame_start, frame_end=frame_end)
                hand_value, result = _run_with_hand_fallback(
                    requested_hand,
                    lambda candidate: _analyze_centroid(source, frames, candidate, wrist_joint_idx, fps),
                )
                entries.append(
                    {
                        "label": f"{source['clip_id']} {source['family']}",
                        "source_id": source["id"],
                        "hand_used": hand_value_to_label(hand_value),
                        "result": result,
                    }
                )
            svg = figure_to_svg(_compare_figure("Hand displacement over time", entries, fps))
            return (
                {
                    "title": "Centroid Compare",
                    "entries": [
                        {
                            "label": entry["label"],
                            "source_id": entry["source_id"],
                            "hand_used": entry["hand_used"],
                            **_serialize_result(entry["result"], fps),
                        }
                        for entry in entries
                    ],
                    "metrics": {"hand_request": requested_hand, "fps": fps},
                },
                svg,
            )

        if mode_id == "point_to_point":
            vertex_a = int(params["vertex_a"])
            vertex_b = int(params["vertex_b"])
            n_neighbors = int(params.get("n_neighbors", self.settings.default_neighbor_count))
            mediapipe_point_a = int(params["mediapipe_point_a"])
            mediapipe_point_b = int(params["mediapipe_point_b"])
            entries = []
            for source in sources:
                frame_start, frame_end = frame_ranges[source["id"]]
                frames = load_source_frames(source, settings=self.settings, frame_start=frame_start, frame_end=frame_end)
                hand_value, result = _run_with_hand_fallback(
                    requested_hand,
                    lambda candidate: _analyze_point_to_point(
                        source=source,
                        frames=frames,
                        hand_value=candidate,
                        wrist_joint_idx=wrist_joint_idx,
                        fps=fps,
                        vertex_a=vertex_a,
                        vertex_b=vertex_b,
                        n_neighbors=n_neighbors,
                        mediapipe_point_a=mediapipe_point_a,
                        mediapipe_point_b=mediapipe_point_b,
                    ),
                )
                entries.append(
                    {
                        "label": f"{source['clip_id']} {source['family']}",
                        "source_id": source["id"],
                        "hand_used": hand_value_to_label(hand_value),
                        "result": result,
                    }
                )
            svg = figure_to_svg(_compare_figure("Filtered region-difference displacement", entries, fps))
            return (
                {
                    "title": "Point to Point",
                    "entries": [
                        {
                            "label": entry["label"],
                            "source_id": entry["source_id"],
                            "hand_used": entry["hand_used"],
                            **_serialize_result(entry["result"], fps),
                        }
                        for entry in entries
                    ],
                    "metrics": {
                        "hand_request": requested_hand,
                        "fps": fps,
                        "vertex_a": vertex_a,
                        "vertex_b": vertex_b,
                        "n_neighbors": n_neighbors,
                        "mediapipe_point_a": mediapipe_point_a,
                        "mediapipe_point_b": mediapipe_point_b,
                    },
                },
                svg,
            )

        if mode_id == "multi_point":
            n_neighbors = int(params.get("n_neighbors", self.settings.default_neighbor_count))
            mano_pairs = parse_pair_text(params.get("mano_pairs_text")) or ((4, 8),)
            mediapipe_pairs = parse_pair_text(params.get("mediapipe_pairs_text")) or ((4, 8),)
            entries = []
            for source in sources:
                frame_start, frame_end = frame_ranges[source["id"]]
                frames = load_source_frames(source, settings=self.settings, frame_start=frame_start, frame_end=frame_end)
                hand_value, pair_entries = _run_with_hand_fallback(
                    requested_hand,
                    lambda candidate: _analyze_multi_point(
                        source,
                        frames,
                        candidate,
                        wrist_joint_idx,
                        fps,
                        n_neighbors,
                        mano_pairs,
                        mediapipe_pairs,
                    ),
                )
                for pair_entry in pair_entries:
                    entries.append(
                        {
                            "label": f"{source['clip_id']} {source['family']} {pair_entry['pair_label']}",
                            "source_id": source["id"],
                            "pair_label": pair_entry["pair_label"],
                            "hand_used": hand_value_to_label(hand_value),
                            "result": pair_entry["result"],
                        }
                    )
            svg = figure_to_svg(_multi_pair_figure("Filtered point-to-point displacement magnitude", entries, fps))
            return (
                {
                    "title": "Multi Pair",
                    "entries": [
                        {
                            "label": entry["label"],
                            "source_id": entry["source_id"],
                            "pair_label": entry["pair_label"],
                            "hand_used": entry["hand_used"],
                            **_serialize_result(entry["result"], fps),
                        }
                        for entry in entries
                    ],
                    "metrics": {"hand_request": requested_hand, "fps": fps, "n_neighbors": n_neighbors},
                },
                svg,
            )

        if mode_id == "neighbor_sweep":
            vertex_a = int(params["vertex_a"])
            vertex_b = int(params["vertex_b"])
            point_counts = parse_number_list(params.get("point_counts_text")) or (10, 20, 30, 40, 50)
            entries = []
            for source in sources:
                series = []
                frame_start, frame_end = frame_ranges[source["id"]]
                frames = load_source_frames(source, settings=self.settings, frame_start=frame_start, frame_end=frame_end)
                hand_value, _ = _run_with_hand_fallback(
                    requested_hand,
                    lambda candidate: _analyze_point_to_point(
                        source=source,
                        frames=frames,
                        hand_value=candidate,
                        wrist_joint_idx=wrist_joint_idx,
                        fps=fps,
                        vertex_a=vertex_a,
                        vertex_b=vertex_b,
                        n_neighbors=max(1, int(point_counts[0]) - 1),
                        mediapipe_point_a=0,
                        mediapipe_point_b=1,
                    ),
                )
                for point_count in point_counts:
                    result = _analyze_point_to_point(
                        source=source,
                        frames=frames,
                        hand_value=hand_value,
                        wrist_joint_idx=wrist_joint_idx,
                        fps=fps,
                        vertex_a=vertex_a,
                        vertex_b=vertex_b,
                        n_neighbors=max(1, int(point_count) - 1),
                        mediapipe_point_a=0,
                        mediapipe_point_b=1,
                    )
                    series.append(
                        {
                            "point_count": int(point_count),
                            "dominant_hz": float(result["dominant"]),
                            "peak_ratio": float(result["peak_ratio"]),
                            "peak_sharpness": float(result["peak_sharpness"]),
                            "temporal_noise": float(result["temporal_noise"]),
                            "spatial_coherence": None if result["spatial_coherence"] is None else float(result["spatial_coherence"]),
                            "rms_amplitude": float(result["rms"]),
                            "sample_count": int(len(result["magnitude"])),
                        }
                    )
                entries.append(
                    {
                        "label": f"{source['clip_id']} {source['family']}",
                        "source_id": source["id"],
                        "hand_used": hand_value_to_label(hand_value),
                        "series": series,
                    }
                )
            svg = figure_to_svg(_sweep_figure(entries))
            return (
                {
                    "title": "Neighbor Sweep",
                    "entries": entries,
                    "metrics": {"hand_request": requested_hand, "fps": fps, "vertex_a": vertex_a, "vertex_b": vertex_b},
                },
                svg,
            )

        if mode_id == "camera_space_frequency":
            if len(sources) != 1:
                raise ValueError("Camera-space frequency analysis requires exactly one source.")
            source = sources[0]
            frame_start, frame_end = frame_ranges[source["id"]]
            frames = _camera_space_frames(source, self.settings, frame_start, frame_end, wrist_joint_idx)
            mano_pairs = parse_pair_text(params.get("mano_pairs_text")) or ((4, 8),)
            hand_value, entries = _run_with_hand_fallback(
                requested_hand,
                lambda candidate: _analyze_camera_space(
                    source,
                    frames,
                    candidate,
                    fps,
                    int(params.get("n_neighbors", self.settings.default_neighbor_count)),
                    mano_pairs,
                    self.settings,
                ),
            )
            serialized = [
                {
                    "label": entry["pair_label"],
                    "pair_label": entry["pair_label"],
                    "hand_used": hand_value_to_label(hand_value),
                    **_serialize_result(entry["result"], fps),
                }
                for entry in entries
            ]
            svg = figure_to_svg(_multi_pair_figure("Camera-space point-to-point motion", [{"label": entry["pair_label"], "result": entry["result"]} for entry in entries], fps))
            return (
                {
                    "title": "Camera-Space Frequency",
                    "entries": serialized,
                    "metrics": {"source_id": source["id"], "hand_request": requested_hand, "hand_used": hand_value_to_label(hand_value), "fps": fps},
                },
                svg,
            )

        if mode_id == "beta_comparison":
            if len(sources) != 1:
                raise ValueError("Beta comparison requires exactly one source.")
            source = sources[0]
            frame_start, frame_end = frame_ranges[source["id"]]
            hand_value, entries = _run_with_hand_fallback(
                requested_hand,
                lambda candidate: _beta_variant_entries(
                    source=source,
                    settings=self.settings,
                    hand_value=candidate,
                    wrist_joint_idx=wrist_joint_idx,
                    vertex_a=int(params["vertex_a"]),
                    vertex_b=int(params["vertex_b"]),
                    n_neighbors=int(params.get("n_neighbors", self.settings.default_neighbor_count)),
                    fps=fps,
                    frame_start=frame_start,
                        frame_end=frame_end,
                    ),
                )
            style_map = {"Actual model": "-", "Beta average": "--"}
            svg = figure_to_svg(
                _compare_figure(
                    "Raw vs beta-average comparison",
                    entries,
                    fps,
                    style_resolver=lambda index, entry: {
                        "color": "C0",
                        "linestyle": style_map.get(entry["label"], "-" if index == 0 else "--"),
                        "linewidth": 1.5,
                    },
                )
            )
            return (
                {
                    "title": "Beta Comparison",
                    "entries": [{"label": entry["label"], "hand_used": hand_value_to_label(hand_value), **_serialize_result(entry["result"], fps)} for entry in entries],
                    "metrics": {"source_id": source["id"], "hand_request": requested_hand, "hand_used": hand_value_to_label(hand_value), "fps": fps},
                },
                svg,
            )

        if mode_id == "beta_multi_point":
            if len(sources) != 1:
                raise ValueError("Beta multi-point comparison requires exactly one source.")
            source = sources[0]
            frame_start, frame_end = frame_ranges[source["id"]]
            pairs = parse_pair_text(params.get("mano_pairs_text")) or ((4, 8), (9, 13))
            entries = []
            hand_value = None
            for pair in pairs:
                hand_value, pair_entries = _run_with_hand_fallback(
                    requested_hand,
                    lambda candidate: _beta_variant_entries(
                        source=source,
                        settings=self.settings,
                        hand_value=candidate,
                        wrist_joint_idx=wrist_joint_idx,
                        vertex_a=int(pair[0]),
                        vertex_b=int(pair[1]),
                        n_neighbors=int(params.get("n_neighbors", self.settings.default_neighbor_count)),
                        fps=fps,
                        frame_start=frame_start,
                        frame_end=frame_end,
                    ),
                )
                for entry in pair_entries:
                    entries.append(
                        {
                            "label": f"{entry['label']} v{pair[0]}-v{pair[1]}",
                            "pair_label": f"v{pair[0]}-v{pair[1]}",
                            "hand_used": hand_value_to_label(hand_value),
                            "result": entry["result"],
                        }
                    )
            svg = figure_to_svg(_multi_pair_figure("Raw vs beta-average multi-pair comparison", entries, fps))
            return (
                {
                    "title": "Beta Multi Pair",
                    "entries": [{"label": entry["label"], "pair_label": entry["pair_label"], "hand_used": entry["hand_used"], **_serialize_result(entry["result"], fps)} for entry in entries],
                    "metrics": {"source_id": source["id"], "hand_request": requested_hand, "hand_used": hand_value_to_label(hand_value if hand_value is not None else -1), "fps": fps},
                },
                svg,
            )

        raise ValueError(f"Unsupported analysis mode: {mode_id}")

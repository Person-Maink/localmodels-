from __future__ import annotations

from dataclasses import asdict, dataclass
import fnmatch
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work
import FILENAME as CONFIG
from analysis_metrics import dominant_frequency_metrics
from npy_io import iter_model_frame_records


ANALYSIS_ROOT = Path(__file__).resolve().parent
ANGLE_ORDER = ("0", "45", "90", "135", "180", "BE")
FAMILY_CHOICES = ("wilor", "stride")
CUSTOM_METRICS_FILENAME = "metrics.csv"
CUSTOM_MANIFEST_FILENAME = "manifest.json"
GRAPH1_OUTPUT_FOLDER = "dataset_point_to_point"
CUSTOM_OUTPUT_FOLDER = "custom"
DEFAULT_METRIC_COLUMNS = (
    "dominant_hz",
    "peak_freq_error_hz",
    "peak_ratio",
    "peak_sharpness",
    "temporal_noise",
    "spatial_coherence",
    "rms_amplitude",
    "palm_orientation_mean_deg",
    "palm_orientation_var_deg",
    "frequency_stability_hz_std",
)
METRIC_DISPLAY_NAMES = {
    "dominant_hz": "Dominant Frequency (Hz)",
    "peak_freq_error_hz": "Peak Frequency Error (Hz)",
    "peak_ratio": "Peak Ratio",
    "peak_sharpness": "Peak Sharpness",
    "temporal_noise": "Temporal Noise",
    "spatial_coherence": "Spatial Coherence",
    "rms_amplitude": "RMS Amplitude",
    "palm_orientation_mean_deg": "Palm Orientation Mean (deg)",
    "palm_orientation_var_deg": "Palm Orientation Variance (deg^2)",
    "frequency_stability_hz_std": "Frequency Stability SD (Hz)",
}
NUMERIC_COLUMNS = {
    "expected_hz",
    "dominant_hz",
    "fft_peak_hz",
    "peak_freq_error_hz",
    "peak_ratio",
    "peak_sharpness",
    "temporal_noise",
    "spatial_coherence",
    "rms_amplitude",
    "palm_orientation_mean_deg",
    "palm_orientation_var_deg",
    "palm_orientation_usable_frames",
    "frequency_stability_hz_std",
    "frequency_stability_window_count",
    "num_samples",
}


@dataclass(frozen=True)
class DatasetClipInfo:
    clip_id: str
    person: str
    angle: str
    cohort: str
    activity_family: str
    activity_state: str
    speed_label: str | None
    expected_hz: float | None
    condition_without_person: str
    activity_condition: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def default_dataset_root() -> Path:
    return Path(CONFIG.ANALYSIS_ROOT).expanduser().resolve().parent / "data" / "new_dataset" / "processed"


def default_custom_output_dir(family: str) -> Path:
    _validate_family(family)
    return Path(CONFIG.ANALYSIS_OUTPUT_DIR).expanduser().resolve() / CUSTOM_OUTPUT_FOLDER / family


def default_graph1_output_dir(family: str) -> Path:
    _validate_family(family)
    return Path(CONFIG.ANALYSIS_OUTPUT_DIR).expanduser().resolve() / GRAPH1_OUTPUT_FOLDER / family


def metrics_cache_path(output_dir: Path | str) -> Path:
    return Path(output_dir).expanduser().resolve() / CUSTOM_METRICS_FILENAME


def manifest_cache_path(output_dir: Path | str) -> Path:
    return Path(output_dir).expanduser().resolve() / CUSTOM_MANIFEST_FILENAME


def parse_dataset_clip_name(clip_id: str) -> DatasetClipInfo:
    parts = str(clip_id).strip().split("_")
    if len(parts) < 4:
        raise ValueError(f"Malformed dataset clip id: {clip_id}")

    person = parts[-1]
    angle = parts[-2]
    if angle not in ANGLE_ORDER:
        raise ValueError(f"Unsupported angle '{angle}' in clip id: {clip_id}")

    condition_without_person = "_".join(parts[:-1])
    if parts[0] == "clean":
        if len(parts) != 5:
            raise ValueError(f"Malformed clean clip id: {clip_id}")
        speed_label = parts[1]
        activity_family = parts[2]
        if speed_label not in {"fast", "slow"}:
            raise ValueError(f"Unsupported clean speed '{speed_label}' in clip id: {clip_id}")
        if activity_family not in {"finger", "wrist"}:
            raise ValueError(f"Unsupported clean activity '{activity_family}' in clip id: {clip_id}")
        expected_hz = 5.0 if speed_label == "fast" else 4.0
        return DatasetClipInfo(
            clip_id=clip_id,
            person=person,
            angle=angle,
            cohort="clean",
            activity_family=activity_family,
            activity_state="clean",
            speed_label=speed_label,
            expected_hz=expected_hz,
            condition_without_person=condition_without_person,
            activity_condition=f"{speed_label}_{activity_family}",
        )

    if len(parts) >= 6 and parts[0] == "tremor" and parts[1] == "simulation":
        body = parts[:-2]
        activity_state = body[-1]
        activity_family = "_".join(body[2:-1])
        if activity_state not in {"normal", "tremor"}:
            raise ValueError(f"Unsupported tremor-simulation state '{activity_state}' in clip id: {clip_id}")
        if activity_family not in {"big_spiral", "small_spiral", "line"}:
            raise ValueError(f"Unsupported tremor-simulation activity '{activity_family}' in clip id: {clip_id}")
        return DatasetClipInfo(
            clip_id=clip_id,
            person=person,
            angle=angle,
            cohort="tremor_simulation",
            activity_family=activity_family,
            activity_state=activity_state,
            speed_label=None,
            expected_hz=None,
            condition_without_person=condition_without_person,
            activity_condition=f"{activity_family}_{activity_state}",
        )

    if len(parts) == 5 and parts[0] == "edge" and parts[1] == "cases":
        activity_family = parts[2]
        if activity_family not in {"finger", "wrist", "still", "leaving"}:
            raise ValueError(f"Unsupported edge-case activity '{activity_family}' in clip id: {clip_id}")
        return DatasetClipInfo(
            clip_id=clip_id,
            person=person,
            angle=angle,
            cohort="edge_cases",
            activity_family=activity_family,
            activity_state="edge_case",
            speed_label=None,
            expected_hz=None,
            condition_without_person=condition_without_person,
            activity_condition=activity_family,
        )

    raise ValueError(f"Unsupported dataset clip id format: {clip_id}")


def parse_dataset_video_path(video_path: Path | str) -> DatasetClipInfo:
    return parse_dataset_clip_name(Path(video_path).stem)


def select_dataset_clips(
    dataset_root: Path | str,
    clip_patterns: list[str] | tuple[str, ...] | None = None,
    people: list[str] | tuple[str, ...] | None = None,
    angles: list[str] | tuple[str, ...] | None = None,
) -> list[DatasetClipInfo]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    normalized_patterns = _normalize_optional_tokens(clip_patterns)
    normalized_people = set(_normalize_optional_tokens(people))
    normalized_angles = set(_normalize_optional_tokens(angles))

    rows: list[DatasetClipInfo] = []
    for video_path in sorted(dataset_root.glob("*.mp4"), key=lambda path: path.name):
        clip = parse_dataset_video_path(video_path)
        if normalized_patterns and not any(fnmatch.fnmatchcase(clip.clip_id, pattern) for pattern in normalized_patterns):
            continue
        if normalized_people and clip.person not in normalized_people:
            continue
        if normalized_angles and clip.angle not in normalized_angles:
            continue
        rows.append(clip)

    if not rows:
        raise RuntimeError("No dataset clips matched the requested filters.")
    return rows


def discover_family_sources(family: str, family_output_root: Path | str | None = None) -> dict[str, str]:
    """Discover model outputs, optionally below an explicit family output root.

    ``family_output_root`` is useful for isolated hyperparameter sweeps whose
    directory is not the standard ``outputs/<family>`` location.
    """
    _validate_family(family)
    root = (
        Path(family_output_root).expanduser().resolve()
        if family_output_root is not None
        else Path(CONFIG.OUTPUTS_ROOT).expanduser().resolve() / family
    )
    sources: dict[str, str] = {}
    if not root.is_dir():
        return sources

    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if family == "wilor":
            mesh_root = child / "meshes"
            if not mesh_root.is_dir():
                continue
            sources[child.name] = str(mesh_root.resolve())
            continue
        if family == "stride":
            if not (child / "meshes").is_dir():
                continue
            if not (child / "refined_sequence.npz").is_file():
                continue
            sources[child.name] = str(child.resolve())
            continue
    return sources


def validate_clip_inventory(
    clips: list[DatasetClipInfo],
    family: str,
    allow_missing: bool = False,
    family_output_root: Path | str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    available_sources = discover_family_sources(family, family_output_root=family_output_root)
    selected_rows: list[dict[str, Any]] = []
    missing_clip_ids: list[str] = []

    for clip in clips:
        source_path = available_sources.get(clip.clip_id)
        if source_path is None:
            missing_clip_ids.append(clip.clip_id)
            continue
        row = clip.to_row()
        row["family"] = family
        row["source_path"] = source_path
        selected_rows.append(row)

    if missing_clip_ids and not allow_missing:
        preview = "\n".join(f"- {clip_id}" for clip_id in missing_clip_ids[:40])
        remainder = len(missing_clip_ids) - min(len(missing_clip_ids), 40)
        if remainder > 0:
            preview = f"{preview}\n- ... ({remainder} more)"
        raise RuntimeError(
            "The selected dataset clips are not fully covered by the chosen family outputs. "
            f"Family='{family}' missing {len(missing_clip_ids)} clip(s):\n{preview}"
        )
    if not selected_rows:
        raise RuntimeError(f"No clips remain for family '{family}' after inventory validation.")
    return selected_rows, missing_clip_ids


def filter_metrics_frame(
    metrics_df: pd.DataFrame,
    clip_patterns: list[str] | tuple[str, ...] | None = None,
    people: list[str] | tuple[str, ...] | None = None,
    angles: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    filtered = metrics_df.copy()
    normalized_patterns = _normalize_optional_tokens(clip_patterns)
    normalized_people = set(_normalize_optional_tokens(people))
    normalized_angles = set(_normalize_optional_tokens(angles))

    if normalized_patterns:
        filtered = filtered[
            filtered["clip_id"].map(lambda clip_id: any(fnmatch.fnmatchcase(str(clip_id), pattern) for pattern in normalized_patterns))
        ]
    if normalized_people:
        filtered = filtered[filtered["person"].isin(normalized_people)]
    if normalized_angles:
        filtered = filtered[filtered["angle"].isin(normalized_angles)]
    return filtered.reset_index(drop=True)


def write_metrics_cache(output_dir: Path | str, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = ensure_dir(output_dir)
    metrics_path = metrics_cache_path(output_dir)
    manifest_path = manifest_cache_path(output_dir)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Cannot write an empty metrics cache.")
    frame = _sort_metrics_frame(frame)
    frame.to_csv(metrics_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics_path, manifest_path


def load_metrics_cache(output_dir: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir = Path(output_dir).expanduser().resolve()
    metrics_path = metrics_cache_path(output_dir)
    manifest_path = manifest_cache_path(output_dir)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Could not find metrics cache: {metrics_path}")

    frame = pd.read_csv(metrics_path)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "angle" in frame.columns:
        frame["angle"] = frame["angle"].astype(str)
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _sort_metrics_frame(frame), manifest


def ensure_dir(path: Path | str) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def metric_display_name(metric_name: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())


def angle_positions() -> dict[str, int]:
    return {angle: index for index, angle in enumerate(ANGLE_ORDER)}


@lru_cache(maxsize=1)
def load_point_to_point_module():
    return _load_script_module(
        "dataset_tremor_point_to_point_module",
        ANALYSIS_ROOT / "Frequency Analysis" / "Point to Point.py",
    )


@lru_cache(maxsize=1)
def load_metric_analysis_module():
    return _load_script_module(
        "dataset_tremor_metric_analysis_module",
        ANALYSIS_ROOT / "Frequency Analysis" / "metric analysis.py",
    )


def run_point_to_point_for_sources(sources: list[str], labels: list[str] | None = None) -> dict[str, Any]:
    if not sources:
        raise ValueError("At least one source path is required for point-to-point analysis.")
    module = load_point_to_point_module()
    resolved_labels = labels or [Path(source).name for source in sources]
    per_source_hand_idx = [resolve_hand_idx_for_source(source_path) for source_path in sources]
    unique_hand_idx = sorted(set(per_source_hand_idx))
    if len(unique_hand_idx) == 1:
        return module.run_point_to_point_analysis(
            {
                "sources": list(sources),
                "labels": list(resolved_labels),
                "hand_idx": int(unique_hand_idx[0]),
            }
        )

    # Overlay batches can legitimately mix left/right detections across participants.
    # In that case, analyze each source with its own inferred hand and merge the entries.
    merged_entries: list[dict[str, Any]] = []
    template: dict[str, Any] | None = None
    for source_path, label, hand_idx in zip(sources, resolved_labels, per_source_hand_idx):
        single_analysis = module.run_point_to_point_analysis(
            {
                "sources": [str(source_path)],
                "labels": [str(label)],
                "hand_idx": int(hand_idx),
            }
        )
        if template is None:
            template = {key: value for key, value in single_analysis.items() if key != "entries"}
        merged_entries.extend(single_analysis["entries"])

    if template is None:
        raise RuntimeError("Point-to-point overlay analysis did not produce any entries.")

    return {
        **template,
        "hand_idx": None,
        "hand_idx_per_source": per_source_hand_idx,
        "entries": merged_entries,
    }


def run_point_to_point_for_clip(source_path: str, label: str | None = None) -> tuple[dict[str, Any], dict[str, Any], float]:
    analysis_data = run_point_to_point_for_sources([str(source_path)], labels=[label or Path(source_path).name])
    entry = analysis_data["entries"][0]
    fps = float(getattr(load_point_to_point_module(), "FPS", 30.0))
    return analysis_data, entry["result"], fps


def compute_frequency_stability_hz_std(
    magnitude: np.ndarray,
    fps: float,
    window_seconds: float = 2.0,
    overlap_fraction: float = 0.5,
) -> tuple[float | None, int]:
    values = np.asarray(magnitude, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return None, 0

    window_size = int(round(float(window_seconds) * float(fps)))
    if window_size <= 0 or values.size < window_size:
        return None, 0
    step = max(1, int(round(window_size * (1.0 - float(overlap_fraction)))))

    peaks: list[float] = []
    for start in range(0, values.size - window_size + 1, step):
        window = values[start : start + window_size]
        peak_hz, _, _ = dominant_frequency_metrics(window, fps=float(fps))
        if np.isfinite(peak_hz) and peak_hz > 0.0:
            peaks.append(float(peak_hz))

    if len(peaks) < 2:
        return None, len(peaks)
    return float(np.std(np.asarray(peaks, dtype=np.float32), ddof=0)), len(peaks)


def compute_palm_orientation_metrics(source_path: str, hand_value: int | None = None) -> dict[str, Any]:
    module = load_metric_analysis_module()
    hand_value = int(resolve_hand_idx_for_source(source_path) if hand_value is None else hand_value)
    region_a, region_b = module._resolve_facing_region_metadata()
    required_vertex = max(int(np.max(region_a)), int(np.max(region_b)))
    angle_values: list[float] = []

    for _, records in iter_model_frame_records(source_path, pattern="*.npy"):
        frame_directions = []
        for record in records:
            if int(record.get("right", -1)) != hand_value:
                continue
            verts = module._camera_relative_verts(record)
            if verts.ndim != 2 or verts.shape[0] <= required_vertex:
                continue
            direction = verts[region_a].mean(axis=0) - verts[region_b].mean(axis=0)
            normalized = module._normalize_vector(direction)
            if normalized is not None:
                frame_directions.append(normalized)
        if not frame_directions:
            continue
        mean_direction = module._normalize_vector(np.mean(np.stack(frame_directions, axis=0), axis=0))
        if mean_direction is None:
            continue
        cosine = float(np.clip(np.dot(mean_direction, module.CAMERA_FACING_AXIS), -1.0, 1.0))
        angle_values.append(float(np.degrees(np.arccos(cosine))))

    if not angle_values:
        return {
            "palm_orientation_mean_deg": None,
            "palm_orientation_var_deg": None,
            "palm_orientation_usable_frames": 0,
        }

    angle_array = np.asarray(angle_values, dtype=np.float32)
    return {
        "palm_orientation_mean_deg": float(np.mean(angle_array)),
        "palm_orientation_var_deg": float(np.var(angle_array, ddof=0)),
        "palm_orientation_usable_frames": int(angle_array.size),
    }


def resolve_hand_idx_for_source(source_path: str, max_frames: int = 25) -> int:
    counts = {0: 0, 1: 0}
    observed_frames = 0
    for _, records in iter_model_frame_records(source_path, pattern="*.npy"):
        observed_frames += 1
        for record in records:
            hand_value = int(record.get("right", -1))
            if hand_value in counts:
                counts[hand_value] += 1
        if observed_frames >= int(max_frames):
            break

    if counts[1] > counts[0]:
        return 1
    if counts[0] > counts[1]:
        return 0
    return int(CONFIG.HAND_IDX)


def resolve_hand_idx_for_sources(sources: list[str]) -> int:
    resolved = {resolve_hand_idx_for_source(source_path) for source_path in sources}
    if not resolved:
        return int(CONFIG.HAND_IDX)
    if len(resolved) > 1:
        joined = ", ".join(sorted(map(str, resolved)))
        raise RuntimeError(f"Mixed hand-side detections across the requested sources: {joined}")
    return int(next(iter(resolved)))


def _load_script_module(module_name: str, file_path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_optional_tokens(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if not values:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _sort_metrics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = [column for column in ("cohort", "activity_family", "activity_state", "speed_label", "angle", "person", "clip_id") if column in frame.columns]
    if "angle" in frame.columns:
        angle_position = angle_positions()
        frame = frame.assign(_angle_sort=frame["angle"].map(lambda angle: angle_position.get(str(angle), len(angle_position))))
        sort_columns = [column for column in sort_columns if column != "angle"] + ["_angle_sort", "angle"]
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "_angle_sort" in frame.columns:
        frame = frame.drop(columns=["_angle_sort"])
    return frame


def _validate_family(family: str) -> None:
    if family not in FAMILY_CHOICES:
        valid_text = ", ".join(FAMILY_CHOICES)
        raise ValueError(f"Unsupported family '{family}'. Valid choices: {valid_text}")

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean

import numpy as np

from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work
import FILENAME as CONFIG
from mano_pickle import load_mano_pickle
from npy_io import iter_model_frame_records


# NumPy legacy aliases for old pickle compatibility.
np.bool = np.bool_
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.unicode = str
np.nan = float("nan")
np.inf = float("inf")


METRIC_FIELDS = (
    ("dominant_hz", "welch_peak_hz"),
    ("fft_peak_hz", "fft_peak_hz"),
    ("peak_ratio", "peak_ratio"),
    ("peak_sharpness", "peak_sharpness"),
    ("temporal_noise", "temporal_noise"),
    ("spatial_coherence", "spatial_coherence"),
    ("rms_amplitude", "rms_amplitude"),
)
FACING_FIELDS = (
    "facing_dir_x",
    "facing_dir_y",
    "facing_dir_z",
    "facing_angle_deg",
    "facing_consistency",
)
SUMMARY_COLUMNS = (
    "model",
    "sources",
    "clips",
    "analyses",
    "rows",
    "welch_peak_hz",
    "fft_peak_hz",
    "peak_ratio",
    "peak_sharpness",
    "temporal_noise",
    "spatial_coherence",
    "rms_amplitude",
    "facing_dir_x",
    "facing_dir_y",
    "facing_dir_z",
    "facing_angle_deg",
    "facing_consistency",
)
BY_ANALYSIS_COLUMNS = (
    "model",
    "analysis",
    "sources",
    "clips",
    "rows",
    "welch_peak_hz",
    "fft_peak_hz",
    "peak_ratio",
    "peak_sharpness",
    "temporal_noise",
    "spatial_coherence",
    "rms_amplitude",
)
ME_VS_OTHER_COLUMNS = (
    "model",
    "me_sources",
    "me_clips",
    "me_analyses",
    "me_rows",
    "me_welch_peak_hz",
    "me_fft_peak_hz",
    "me_peak_ratio",
    "me_peak_sharpness",
    "me_temporal_noise",
    "me_spatial_coherence",
    "me_rms_amplitude",
    "me_facing_dir_x",
    "me_facing_dir_y",
    "me_facing_dir_z",
    "me_facing_angle_deg",
    "me_facing_consistency",
    "other_sources",
    "other_clips",
    "other_analyses",
    "other_rows",
    "other_welch_peak_hz",
    "other_fft_peak_hz",
    "other_peak_ratio",
    "other_peak_sharpness",
    "other_temporal_noise",
    "other_spatial_coherence",
    "other_rms_amplitude",
    "other_facing_dir_x",
    "other_facing_dir_y",
    "other_facing_dir_z",
    "other_facing_angle_deg",
    "other_facing_consistency",
)
FACING_AUDIT_COLUMNS = (
    "model",
    "clip_id",
    "source",
    "hand_used",
    "usable_frames",
    "facing_dir_x",
    "facing_dir_y",
    "facing_dir_z",
    "facing_angle_deg",
    "facing_consistency",
)
FINETUNE_DISPLAY_NAMES = {
    "lora_finetuning": "WiLoR Finetune LoRA",
    "main_learnable_finetnuing": "WiLoR Finetune Main Learnable",
    "main_static_finetuning": "WiLoR Finetune Main Static",
}
FAMILY_DISPLAY_NAMES = {
    "hamba": "Hamba",
    "wilor": "WiLoR",
    "dynhamr": "DynHAMR",
    "stride": "Stride",
    "mediapipe": "MediaPipe",
}
MODEL_SORT_ORDER = {
    "hamba": 0,
    "wilor": 1,
    "wilor_finetune:lora_finetuning": 2,
    "wilor_finetune:main_learnable_finetnuing": 3,
    "wilor_finetune:main_static_finetuning": 4,
    "dynhamr": 5,
    "stride": 6,
    "mediapipe": 7,
}
CAMERA_FACING_AXIS = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
VECTOR_EPSILON = 1e-8


@dataclass(frozen=True)
class MetricRecord:
    model_key: str
    model_label: str
    analysis: str
    source: str
    clip_id: str
    values: dict[str, float | None]


@dataclass(frozen=True)
class FacingSourceRecord:
    model_key: str
    model_label: str
    clip_id: str
    source: str
    hand_used: str
    usable_frames: int
    facing_dir_x: float | None
    facing_dir_y: float | None
    facing_dir_z: float | None
    facing_angle_deg: float | None
    facing_consistency: float


def _default_metrics_csv() -> Path:
    output_dir = Path(getattr(CONFIG, "ANALYSIS_OUTPUT_DIR", Path(CONFIG.OUTPUTS_ROOT) / "analysis_images"))
    return output_dir / "metrics.csv"


def _default_output_dir(metrics_csv: Path) -> Path:
    return metrics_csv.resolve().parent / "metric_analysis"


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read a Run All metrics.csv export, summarize the saved metrics by model family / "
            "finetune variant, and save readable tables."
        )
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=_default_metrics_csv(),
        help="Path to the saved Run All metrics.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the summary exports will be written.",
    )
    parser.add_argument(
        "--include-mediapipe",
        action="store_true",
        help="Include MediaPipe rows. By default the summary keeps only kind=model rows.",
    )
    parser.add_argument(
        "--raw-row-weighting",
        action="store_true",
        help=(
            "Average directly over all metric rows. The default instead balances each analysis type "
            "equally per model before building the main summary."
        ),
    )
    return parser.parse_args()


def parse_model_descriptor(source: str) -> tuple[str, str, str]:
    path = Path(str(source))
    parts = path.parts
    if "outputs" not in parts:
        raise ValueError(f"Could not resolve model family from source path: {source}")

    outputs_index = parts.index("outputs")
    if outputs_index + 1 >= len(parts):
        raise ValueError(f"Malformed outputs path: {source}")

    family = parts[outputs_index + 1]
    if family == "wilor_finetune":
        experiment = parts[outputs_index + 2] if outputs_index + 2 < len(parts) else "unknown_experiment"
        clip_id = parts[outputs_index + 3] if outputs_index + 3 < len(parts) else path.stem
        key = f"{family}:{experiment}"
        label = FINETUNE_DISPLAY_NAMES.get(experiment, f"WiLoR Finetune {experiment.replace('_', ' ').title()}")
        return key, label, clip_id

    clip_id = parts[outputs_index + 2] if outputs_index + 2 < len(parts) else path.stem
    key = family
    label = FAMILY_DISPLAY_NAMES.get(family, family.replace("_", " ").title())
    return key, label, clip_id


def load_metric_records(metrics_csv: Path, include_mediapipe: bool = False) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "success":
                continue
            kind = row.get("kind", "")
            if not include_mediapipe and kind != "model":
                continue
            source = (row.get("source") or "").strip()
            if not source:
                continue

            try:
                model_key, model_label, clip_id = parse_model_descriptor(source)
            except ValueError:
                continue

            values: dict[str, float | None] = {}
            for csv_name, public_name in METRIC_FIELDS:
                values[public_name] = _parse_optional_float(row.get(csv_name))

            records.append(
                MetricRecord(
                    model_key=model_key,
                    model_label=model_label,
                    analysis=row.get("analysis", "").strip() or "unknown",
                    source=source,
                    clip_id=clip_id,
                    values=values,
                )
            )

    return records


@lru_cache(maxsize=1)
def _load_facing_mano_faces(mano_right_path: str) -> np.ndarray:
    mano = load_mano_pickle(Path(mano_right_path).expanduser().resolve())
    return np.asarray(mano["f"], dtype=np.int32)


def _build_vertex_adjacency(total_verts: int, tri_faces: np.ndarray) -> list[list[int]]:
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


def _select_graph_neighbors(seed: int, adjacency: list[list[int]], n_neighbors: int) -> list[int]:
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


def _build_region_indices(seed: int, adjacency: list[list[int]], n_neighbors: int) -> np.ndarray:
    return np.asarray([seed] + _select_graph_neighbors(seed, adjacency, n_neighbors), dtype=np.int32)


def _resolve_facing_region_metadata() -> tuple[np.ndarray, np.ndarray]:
    faces = _load_facing_mano_faces(str(CONFIG.MANO_RIGHT_PATH))
    total_verts = int(np.max(faces)) + 1
    adjacency = _build_vertex_adjacency(total_verts, faces)
    seed_a = int(getattr(CONFIG, "FACING_MODEL_VERTEX_A", 333))
    seed_b = int(getattr(CONFIG, "FACING_MODEL_VERTEX_B", 173))
    n_neighbors = int(CONFIG.N_NEIGHBORS)
    if not (0 <= seed_a < total_verts):
        raise ValueError(f"FACING_MODEL_VERTEX_A={seed_a} out of range [0, {total_verts - 1}]")
    if not (0 <= seed_b < total_verts):
        raise ValueError(f"FACING_MODEL_VERTEX_B={seed_b} out of range [0, {total_verts - 1}]")
    if seed_a == seed_b:
        raise ValueError("FACING_MODEL_VERTEX_A and FACING_MODEL_VERTEX_B must be different")
    if n_neighbors <= 0:
        raise ValueError(f"N_NEIGHBORS must be > 0, got {n_neighbors}")
    return (
        _build_region_indices(seed_a, adjacency, n_neighbors),
        _build_region_indices(seed_b, adjacency, n_neighbors),
    )


def _camera_relative_verts(record: dict) -> np.ndarray:
    cam_t = np.asarray(record.get("cam_t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    verts = record.get("verts")
    if verts is not None:
        return np.asarray(verts, dtype=np.float32) + cam_t[None, :]
    return np.asarray(record["verts_world"], dtype=np.float32)


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    values = np.asarray(vector, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(values)):
        return None
    norm = float(np.linalg.norm(values))
    if norm <= VECTOR_EPSILON:
        return None
    return values / norm


def _summarize_unit_directions(unit_directions: list[np.ndarray]) -> dict:
    if not unit_directions:
        return {
            "facing_dir_x": None,
            "facing_dir_y": None,
            "facing_dir_z": None,
            "facing_angle_deg": None,
            "facing_consistency": 0.0,
        }

    mean_vector = np.mean(np.stack(unit_directions, axis=0), axis=0)
    consistency = float(np.linalg.norm(mean_vector))
    if not np.isfinite(consistency) or consistency <= VECTOR_EPSILON:
        return {
            "facing_dir_x": None,
            "facing_dir_y": None,
            "facing_dir_z": None,
            "facing_angle_deg": None,
            "facing_consistency": 0.0,
        }

    mean_direction = mean_vector / consistency
    cosine = float(np.clip(np.dot(mean_direction, CAMERA_FACING_AXIS), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosine)))
    return {
        "facing_dir_x": float(mean_direction[0]),
        "facing_dir_y": float(mean_direction[1]),
        "facing_dir_z": float(mean_direction[2]),
        "facing_angle_deg": angle_deg,
        "facing_consistency": consistency,
    }


def _summarize_source_facing_from_frame_records(
    frame_records,
    region_a: np.ndarray,
    region_b: np.ndarray,
    hand_value: int,
) -> dict:
    unit_directions: list[np.ndarray] = []
    required_vertex = max(int(np.max(region_a)), int(np.max(region_b)))
    for _, records in frame_records:
        frame_vectors = []
        for record in records:
            if int(record.get("right", -1)) != int(hand_value):
                continue
            verts = _camera_relative_verts(record)
            if verts.ndim != 2 or verts.shape[0] <= required_vertex:
                continue
            direction = verts[region_a].mean(axis=0) - verts[region_b].mean(axis=0)
            normalized = _normalize_vector(direction)
            if normalized is not None:
                frame_vectors.append(normalized)
        if not frame_vectors:
            continue
        frame_direction = _normalize_vector(np.mean(np.stack(frame_vectors, axis=0), axis=0))
        if frame_direction is not None:
            unit_directions.append(frame_direction)

    summary = _summarize_unit_directions(unit_directions)
    summary["usable_frames"] = int(len(unit_directions))
    return summary


def _hand_value_to_label(hand_value: int) -> str:
    if int(hand_value) == 1:
        return "right"
    if int(hand_value) == 0:
        return "left"
    return "unknown"


def compute_facing_source_records(records: list[MetricRecord]) -> list[FacingSourceRecord]:
    region_a, region_b = _resolve_facing_region_metadata()
    source_index: dict[str, tuple[str, str, str]] = {}
    for record in records:
        if record.model_key == "mediapipe":
            continue
        source_index.setdefault(record.source, (record.model_key, record.model_label, record.clip_id))

    hand_value = int(CONFIG.HAND_IDX)
    facing_rows = []
    for source_path, (model_key, model_label, clip_id) in sorted(source_index.items()):
        try:
            summary = _summarize_source_facing_from_frame_records(
                iter_model_frame_records(source_path, pattern="*.npy"),
                region_a=region_a,
                region_b=region_b,
                hand_value=hand_value,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[metric analysis] Skipping facing metric for {source_path}: {exc}")
            continue
        if int(summary["usable_frames"]) <= 0:
            continue
        facing_rows.append(
            FacingSourceRecord(
                model_key=model_key,
                model_label=model_label,
                clip_id=clip_id,
                source=source_path,
                hand_used=_hand_value_to_label(hand_value),
                usable_frames=int(summary["usable_frames"]),
                facing_dir_x=summary["facing_dir_x"],
                facing_dir_y=summary["facing_dir_y"],
                facing_dir_z=summary["facing_dir_z"],
                facing_angle_deg=summary["facing_angle_deg"],
                facing_consistency=float(summary["facing_consistency"]),
            )
        )
    return facing_rows


def _aggregate_facing_source_records(records: list[FacingSourceRecord]) -> dict:
    unit_directions = []
    for record in records:
        if (
            record.facing_dir_x is None
            or record.facing_dir_y is None
            or record.facing_dir_z is None
        ):
            continue
        normalized = _normalize_vector(np.asarray([record.facing_dir_x, record.facing_dir_y, record.facing_dir_z], dtype=np.float32))
        if normalized is not None:
            unit_directions.append(normalized)
    return _summarize_unit_directions(unit_directions)


def _attach_facing_metrics(rows: list[dict], facing_source_records: list[FacingSourceRecord]) -> None:
    grouped: dict[str, list[FacingSourceRecord]] = defaultdict(list)
    for record in facing_source_records:
        grouped[record.model_key].append(record)

    for row in rows:
        row.update(_aggregate_facing_source_records(grouped.get(row["model_key"], [])))


def summarize_by_model_and_analysis(records: list[MetricRecord]) -> list[dict]:
    grouped: dict[tuple[str, str], list[MetricRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.model_key, record.analysis)].append(record)

    rows = []
    for (model_key, analysis), group in grouped.items():
        model_label = group[0].model_label
        row = {
            "model_key": model_key,
            "model": model_label,
            "analysis": analysis,
            "sources": len({record.source for record in group}),
            "clips": len({record.clip_id for record in group}),
            "rows": len(group),
        }
        for _, public_name in METRIC_FIELDS:
            values = [record.values[public_name] for record in group if record.values[public_name] is not None]
            row[public_name] = fmean(values) if values else None
        rows.append(row)

    rows.sort(key=lambda row: (_model_sort_key(row["model_key"]), row["analysis"], row["model"]))
    return rows


def summarize_by_model(
    records: list[MetricRecord],
    balance_analyses: bool = True,
) -> list[dict]:
    per_analysis_rows = summarize_by_model_and_analysis(records)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in per_analysis_rows:
        grouped[row["model_key"]].append(row)

    rows = []
    for model_key, analysis_rows in grouped.items():
        model_label = analysis_rows[0]["model"]
        all_records = [record for record in records if record.model_key == model_key]
        row = {
            "model_key": model_key,
            "model": model_label,
            "sources": len({record.source for record in all_records}),
            "clips": len({record.clip_id for record in all_records}),
            "analyses": len({record.analysis for record in all_records}),
            "rows": len(all_records),
        }
        for _, public_name in METRIC_FIELDS:
            if balance_analyses:
                values = [analysis_row[public_name] for analysis_row in analysis_rows if analysis_row[public_name] is not None]
            else:
                values = [record.values[public_name] for record in all_records if record.values[public_name] is not None]
            row[public_name] = fmean(values) if values else None
        rows.append(row)

    rows.sort(key=lambda row: (_model_sort_key(row["model_key"]), row["model"]))
    return rows


def summarize_me_vs_other(
    records: list[MetricRecord],
    facing_source_records: list[FacingSourceRecord],
    balance_analyses: bool = True,
) -> list[dict]:
    me_records = [record for record in records if is_me_clip(record.clip_id)]
    other_records = [record for record in records if not is_me_clip(record.clip_id)]

    me_summary_rows = summarize_by_model(me_records, balance_analyses=balance_analyses)
    other_summary_rows = summarize_by_model(other_records, balance_analyses=balance_analyses)
    _attach_facing_metrics(
        me_summary_rows,
        [record for record in facing_source_records if is_me_clip(record.clip_id)],
    )
    _attach_facing_metrics(
        other_summary_rows,
        [record for record in facing_source_records if not is_me_clip(record.clip_id)],
    )

    me_summary = {row["model_key"]: row for row in me_summary_rows}
    other_summary = {row["model_key"]: row for row in other_summary_rows}

    all_model_keys = sorted(set(me_summary) | set(other_summary), key=_model_sort_key)
    rows = []
    for model_key in all_model_keys:
        me_row = me_summary.get(model_key)
        other_row = other_summary.get(model_key)
        model_label = (me_row or {}).get("model") or (other_row or {}).get("model") or model_key
        row = {"model": model_label, "model_key": model_key}
        _merge_cohort_fields(row, "me", me_row)
        _merge_cohort_fields(row, "other", other_row)
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_csv_value(row.get(column)) for column in columns})


def render_markdown_table(rows: list[dict], columns: tuple[str, ...]) -> str:
    headers = [column.replace("_", " ").title() for column in columns]
    formatted_rows = []
    for row in rows:
        formatted_rows.append([_format_markdown_value(row.get(column)) for column in columns])

    widths = [len(header) for header in headers]
    for values in formatted_rows:
        for index, value in enumerate(values):
            widths[index] = max(widths[index], len(value))

    def fmt_line(values):
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    lines = [fmt_line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    for values in formatted_rows:
        lines.append(fmt_line(values))
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_table(rows, columns) + "\n", encoding="utf-8")


def write_facing_audit_csv(path: Path, rows: list[FacingSourceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FACING_AUDIT_COLUMNS))
        writer.writeheader()
        for row in rows:
            payload = {
                "model": row.model_label,
                "clip_id": row.clip_id,
                "source": row.source,
                "hand_used": row.hand_used,
                "usable_frames": row.usable_frames,
                "facing_dir_x": row.facing_dir_x,
                "facing_dir_y": row.facing_dir_y,
                "facing_dir_z": row.facing_dir_z,
                "facing_angle_deg": row.facing_angle_deg,
                "facing_consistency": row.facing_consistency,
            }
            writer.writerow({column: _format_csv_value(payload.get(column)) for column in FACING_AUDIT_COLUMNS})


def _model_sort_key(model_key: str) -> tuple[int, str]:
    return (MODEL_SORT_ORDER.get(model_key, 1000), model_key)


def is_me_clip(clip_id: str) -> bool:
    text = str(clip_id).strip().lower()
    return bool(re.match(r"^me(?:[ _-]?\d+)(?:\b|_)", text))


def _merge_cohort_fields(target: dict, prefix: str, row: dict | None) -> None:
    target[f"{prefix}_sources"] = None if row is None else row.get("sources")
    target[f"{prefix}_clips"] = None if row is None else row.get("clips")
    target[f"{prefix}_analyses"] = None if row is None else row.get("analyses")
    target[f"{prefix}_rows"] = None if row is None else row.get("rows")
    for _, public_name in METRIC_FIELDS:
        target[f"{prefix}_{public_name}"] = None if row is None else row.get(public_name)
    for field_name in FACING_FIELDS:
        target[f"{prefix}_{field_name}"] = None if row is None else row.get(field_name)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12f}"
    return value


def _format_markdown_value(value):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1.0:
            return f"{value:.4f}"
        return f"{value:.3e}"
    return str(value)


def main():
    args = _parse_args()
    metrics_csv = args.metrics_csv.expanduser().resolve()
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Could not find metrics CSV: {metrics_csv}")

    output_dir = (args.output_dir or _default_output_dir(metrics_csv)).expanduser().resolve()
    records = load_metric_records(metrics_csv, include_mediapipe=bool(args.include_mediapipe))
    if not records:
        raise RuntimeError("No metric rows matched the requested filters.")

    summary_rows = summarize_by_model(records, balance_analyses=not args.raw_row_weighting)
    by_analysis_rows = summarize_by_model_and_analysis(records)
    facing_source_records = compute_facing_source_records(records)
    _attach_facing_metrics(summary_rows, facing_source_records)
    me_vs_other_rows = summarize_me_vs_other(
        records,
        facing_source_records=facing_source_records,
        balance_analyses=not args.raw_row_weighting,
    )

    summary_csv = output_dir / "model_summary.csv"
    summary_md = output_dir / "model_summary.md"
    by_analysis_csv = output_dir / "model_summary_by_analysis.csv"
    me_vs_other_csv = output_dir / "model_summary_me_vs_other.csv"
    me_vs_other_md = output_dir / "model_summary_me_vs_other.md"
    facing_audit_csv = output_dir / "model_facing_by_source.csv"

    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_markdown(summary_md, summary_rows, SUMMARY_COLUMNS)
    write_csv(by_analysis_csv, by_analysis_rows, BY_ANALYSIS_COLUMNS)
    write_csv(me_vs_other_csv, me_vs_other_rows, ME_VS_OTHER_COLUMNS)
    write_markdown(me_vs_other_md, me_vs_other_rows, ME_VS_OTHER_COLUMNS)
    write_facing_audit_csv(facing_audit_csv, facing_source_records)

    weighting_text = "raw metric rows" if args.raw_row_weighting else "analysis-balanced means"
    print(f"Saved model summary CSV: {summary_csv}")
    print(f"Saved model summary Markdown: {summary_md}")
    print(f"Saved per-analysis CSV: {by_analysis_csv}")
    print(f"Saved me-vs-other CSV: {me_vs_other_csv}")
    print(f"Saved me-vs-other Markdown: {me_vs_other_md}")
    print(f"Saved facing audit CSV: {facing_audit_csv}")
    print(f"Main table weighting: {weighting_text}")
    print()
    print(render_markdown_table(summary_rows, SUMMARY_COLUMNS))
    print()
    print("Me-vs-other split:")
    print(render_markdown_table(me_vs_other_rows, ME_VS_OTHER_COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

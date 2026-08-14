from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work
import FILENAME as CONFIG
from npy_io import iter_model_frame_records


DEFAULT_WRIST_VERTEX = 0
DEFAULT_ANCHOR_VERTICES = (0, 5, 9, 13, 17)
DEFAULT_MIN_MATCHED_FRAMES = 8


@dataclass(frozen=True)
class MatchedSequences:
    verts_a: np.ndarray
    verts_b: np.ndarray
    frame_ids: np.ndarray
    fps: float | None
    source_a: str
    source_b: str
    source_a_frames: int
    source_b_frames: int


def load_matched_sequences(
    source_a: str | Path,
    source_b: str | Path,
    hand_side: int | None = None,
    min_frames: int = DEFAULT_MIN_MATCHED_FRAMES,
    fps: float | None = None,
) -> MatchedSequences:
    side = int(CONFIG.HAND_IDX if hand_side is None else hand_side)
    frames_a = _load_source_frame_map(source_a, side)
    frames_b = _load_source_frame_map(source_b, side)
    matched_ids = sorted(set(frames_a) & set(frames_b))

    if len(matched_ids) < int(min_frames):
        raise ValueError(
            "Too few matched frames for fusion analysis: "
            f"{len(matched_ids)} matched, need at least {min_frames}. "
            f"source_a_frames={len(frames_a)}, source_b_frames={len(frames_b)}"
        )

    verts_a = np.stack([frames_a[frame_id] for frame_id in matched_ids], axis=0).astype(np.float32, copy=False)
    verts_b = np.stack([frames_b[frame_id] for frame_id in matched_ids], axis=0).astype(np.float32, copy=False)

    if verts_a.shape != verts_b.shape:
        raise ValueError(f"Matched vertex shapes differ: source_a={verts_a.shape}, source_b={verts_b.shape}")
    if verts_a.ndim != 3 or verts_a.shape[2] != 3:
        raise ValueError(f"Expected [T, V, 3] vertices, got {verts_a.shape}")

    return MatchedSequences(
        verts_a=verts_a,
        verts_b=verts_b,
        frame_ids=np.asarray(matched_ids, dtype=np.int32),
        fps=None if fps is None else float(fps),
        source_a=str(Path(source_a)),
        source_b=str(Path(source_b)),
        source_a_frames=len(frames_a),
        source_b_frames=len(frames_b),
    )


def wrist_center(
    verts: np.ndarray,
    wrist_vertex: int | None = None,
    wrist_joint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = _as_vertices(verts)
    if wrist_joint is not None:
        wrist = np.asarray(wrist_joint, dtype=np.float32)
        if wrist.shape == (3,):
            wrist = np.broadcast_to(wrist.reshape(1, 3), (values.shape[0], 3)).copy()
        if wrist.shape != (values.shape[0], 3):
            raise ValueError(f"wrist_joint must have shape [3] or [T, 3], got {wrist.shape}")
    else:
        vertex = DEFAULT_WRIST_VERTEX if wrist_vertex is None else int(wrist_vertex)
        if vertex < 0 or vertex >= values.shape[1]:
            raise ValueError(f"wrist_vertex {vertex} is outside vertex range 0..{values.shape[1] - 1}")
        wrist = values[:, vertex, :]

    return values - wrist[:, None, :], wrist.astype(np.float32, copy=False)


def wrist_center_with_regressor(
    verts: np.ndarray,
    j_reg: np.ndarray,
    wrist_joint_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = _as_vertices(verts)
    regressor = np.asarray(j_reg, dtype=np.float32)
    if regressor.ndim != 2 or regressor.shape[1] != values.shape[1]:
        raise ValueError(
            f"J_regressor shape {regressor.shape} is incompatible with vertices {values.shape}"
        )
    joints = np.einsum("jv,tvc->tjc", regressor, values, optimize=True)
    wrist = joints[:, int(wrist_joint_idx), :]
    return values - wrist[:, None, :], wrist.astype(np.float32, copy=False)


def align_sequence_to_reference(
    moving: np.ndarray,
    reference: np.ndarray,
    anchor_vertices: Sequence[int],
    allow_scale: bool = False,
) -> np.ndarray:
    moving_values = _as_vertices(moving)
    reference_values = _as_vertices(reference)
    if moving_values.shape != reference_values.shape:
        raise ValueError(f"moving/reference shapes differ: {moving_values.shape} vs {reference_values.shape}")

    anchors = np.asarray([int(vertex_id) for vertex_id in anchor_vertices], dtype=np.int32)
    if anchors.size < 3:
        raise ValueError("At least three anchor vertices are required for rigid alignment.")
    if np.any(anchors < 0) or np.any(anchors >= moving_values.shape[1]):
        raise ValueError(f"anchor vertices must be in 0..{moving_values.shape[1] - 1}: {anchors.tolist()}")

    aligned = np.empty_like(moving_values, dtype=np.float32)
    for index in range(moving_values.shape[0]):
        aligned[index] = _kabsch_align_frame(
            moving_values[index],
            reference_values[index],
            anchors,
            allow_scale=bool(allow_scale),
        )
    return aligned


def raw_average_fusion(
    wilor_centered: np.ndarray,
    stride_centered: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return float(alpha) * np.asarray(wilor_centered, dtype=np.float32) + (
        1.0 - float(alpha)
    ) * np.asarray(stride_centered, dtype=np.float32)


def tremor_residual_fusion(
    wilor_centered: np.ndarray,
    stride_centered: np.ndarray,
    fps: float,
    band: tuple[float, float] = (4.0, 12.0),
    residual_weight: float = 1.0,
    filter_order: int = 4,
) -> np.ndarray:
    wilor_values = _as_vertices(wilor_centered)
    stride_values = _as_vertices(stride_centered)
    if wilor_values.shape != stride_values.shape:
        raise ValueError(f"WiLoR/STRIDE shapes differ: {wilor_values.shape} vs {stride_values.shape}")

    residual = wilor_values - stride_values
    filtered = _bandpass_sos_filter(
        residual,
        fps=float(fps),
        band=band,
        filter_order=int(filter_order),
    )
    return stride_values + float(residual_weight) * filtered


def _load_source_frame_map(source_path: str | Path, hand_side: int) -> dict[int, np.ndarray]:
    frame_map: dict[int, np.ndarray] = {}
    total_records = 0
    matched_records = 0
    for discovered_frame_id, records in iter_model_frame_records(source_path, pattern="*.npy"):
        candidates = []
        for rec in records:
            total_records += 1
            right = int(rec.get("right", -1))
            if right != int(hand_side):
                continue
            frame_id = rec.get("frame_id", None)
            if frame_id is None:
                frame_id = discovered_frame_id
            if int(frame_id) < 0:
                frame_id = discovered_frame_id
            verts = _record_vertices(rec)
            candidates.append((int(frame_id), verts, str(rec.get("path", source_path))))
            matched_records += 1

        if not candidates:
            continue

        frame_id, verts, _path = candidates[0]
        if len(candidates) > 1:
            print(
                f"[fusion_utils] Warning: frame {frame_id} has {len(candidates)} records for hand_side={hand_side}; "
                "using the first record."
            )
        if frame_id in frame_map:
            print(f"[fusion_utils] Warning: duplicate frame_id={frame_id}; keeping the first record.")
            continue
        frame_map[frame_id] = verts

    if not frame_map:
        raise ValueError(
            f"No usable records for hand_side={hand_side} under '{source_path}'. "
            f"Loaded records={total_records}, matched records={matched_records}."
        )
    return frame_map


def _record_vertices(record: dict) -> np.ndarray:
    if "verts_world" in record and record["verts_world"] is not None:
        verts = np.asarray(record["verts_world"], dtype=np.float32)
    elif "verts" in record and record["verts"] is not None:
        verts = np.asarray(record["verts"], dtype=np.float32)
    else:
        raise ValueError(f"Record is missing verts_world/verts: {record.get('path', '<unknown>')}")

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Invalid vertex shape in {record.get('path', '<unknown>')}: {verts.shape}")
    return verts


def _as_vertices(verts: np.ndarray) -> np.ndarray:
    values = np.asarray(verts, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"Expected vertices with shape [T, V, 3], got {values.shape}")
    return values


def _kabsch_align_frame(
    moving: np.ndarray,
    reference: np.ndarray,
    anchors: np.ndarray,
    allow_scale: bool,
) -> np.ndarray:
    moving_anchor = moving[anchors]
    reference_anchor = reference[anchors]
    moving_centroid = moving_anchor.mean(axis=0)
    reference_centroid = reference_anchor.mean(axis=0)
    moving_centered = moving_anchor - moving_centroid
    reference_centered = reference_anchor - reference_centroid

    covariance = moving_centered.T @ reference_centered
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T

    scale = 1.0
    if allow_scale:
        denom = float(np.sum(moving_centered * moving_centered))
        if denom > 1e-12:
            scale = float(np.sum(singular_values) / denom)

    return ((moving - moving_centroid) @ rotation.T * scale + reference_centroid).astype(np.float32)


def _bandpass_sos_filter(
    values: np.ndarray,
    fps: float,
    band: tuple[float, float],
    filter_order: int,
) -> np.ndarray:
    low_hz, high_hz = float(band[0]), float(band[1])
    nyquist = 0.5 * float(fps)
    if nyquist <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if low_hz <= 0.0 or high_hz <= low_hz:
        raise ValueError(f"Invalid band: {band}")
    if low_hz >= nyquist:
        print(
            f"[fusion_utils] Warning: band low {low_hz:g} Hz is above Nyquist {nyquist:g} Hz; "
            "returning zero residual."
        )
        return np.zeros_like(values, dtype=np.float32)

    clipped_high = min(high_hz, nyquist * 0.999)
    sos = butter(
        int(filter_order),
        [low_hz / nyquist, clipped_high / nyquist],
        btype="bandpass",
        output="sos",
    )
    try:
        return np.asarray(sosfiltfilt(sos, values, axis=0), dtype=np.float32)
    except ValueError:
        print(
            "[fusion_utils] Warning: sequence too short for zero-phase residual filtering; "
            "using causal SOS filtering."
        )
        return np.asarray(sosfilt(sos, values, axis=0), dtype=np.float32)

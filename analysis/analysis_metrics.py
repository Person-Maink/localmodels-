from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, welch


TREMOR_BAND_LOW_HZ = 4.0
TREMOR_BAND_HIGH_HZ = 12.0
PEAK_SHARPNESS_RADIUS_BINS = 2


def lowpass_filter(
    signal: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    return _run_filter(signal, fps=fps, order=order, mode="lowpass", cutoff_hz=cutoff_hz)


def bandpass_filter(
    signal: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    return _run_filter(
        signal,
        fps=fps,
        order=order,
        mode="bandpass",
        low_hz=low_hz,
        high_hz=high_hz,
    )


def finish_motion_analysis(
    trajectory: Sequence[np.ndarray],
    fps: float,
    filter_kind: str,
    filter_order: int,
    lowpass_cutoff_hz: float | None = None,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    psd_nperseg: int = 256,
    coherence_positions: np.ndarray | None = None,
    coherence_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict:
    if len(trajectory) == 0:
        raise RuntimeError("No trajectory frames were available for analysis.")

    trajectory_array = np.stack(trajectory, axis=0).astype(np.float32, copy=False)
    if filter_kind == "bandpass":
        if band_low_hz is None or band_high_hz is None:
            raise ValueError("band_low_hz and band_high_hz are required for bandpass analysis.")
        filtered = bandpass_filter(
            trajectory_array,
            fps=fps,
            low_hz=band_low_hz,
            high_hz=band_high_hz,
            order=filter_order,
        )
    elif filter_kind == "lowpass":
        if lowpass_cutoff_hz is None:
            raise ValueError("lowpass_cutoff_hz is required for lowpass analysis.")
        filtered = lowpass_filter(
            trajectory_array,
            fps=fps,
            cutoff_hz=lowpass_cutoff_hz,
            order=filter_order,
        )
    else:
        raise ValueError(f"Unsupported filter kind: {filter_kind}")

    magnitude = np.linalg.norm(filtered, axis=1).astype(np.float32, copy=False)
    magnitude = magnitude - magnitude.mean()
    freqs, psd = welch(magnitude, fs=float(fps), nperseg=min(int(psd_nperseg), len(magnitude)))

    dominant_hz, peak_ratio, peak_sharpness = dominant_frequency_metrics(magnitude, fps=fps)
    rms_amplitude = float(np.sqrt(np.mean(np.square(magnitude, dtype=np.float64))))
    temporal_noise = frame_to_frame_variance(trajectory_array)
    spatial_coherence = spatial_coherence_from_positions(coherence_positions, coherence_pairs)

    return {
        "trajectory": trajectory_array,
        "filtered": np.asarray(filtered, dtype=np.float32),
        "magnitude": magnitude,
        "freqs": np.asarray(freqs, dtype=np.float32),
        "psd": np.asarray(psd, dtype=np.float32),
        "dominant": float(dominant_hz),
        "rms": float(rms_amplitude),
        "peak_ratio": float(peak_ratio),
        "peak_sharpness": float(peak_sharpness),
        "temporal_noise": float(temporal_noise),
        "spatial_coherence": None if spatial_coherence is None else float(spatial_coherence),
    }


def dominant_frequency_metrics(
    signal: np.ndarray,
    fps: float,
    band_low_hz: float = TREMOR_BAND_LOW_HZ,
    band_high_hz: float = TREMOR_BAND_HIGH_HZ,
    sharpness_radius_bins: int = PEAK_SHARPNESS_RADIUS_BINS,
) -> tuple[float, float, float]:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0, 0.0

    spectrum = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(values.size, d=1.0 / float(fps))

    band_mask = (freqs >= float(band_low_hz)) & (freqs <= float(band_high_hz))
    band_indices = np.flatnonzero(band_mask)
    if band_indices.size == 0:
        nonzero = np.flatnonzero(freqs > 0.0)
        if nonzero.size == 0:
            return 0.0, 0.0, 0.0
        band_indices = nonzero

    band_spectrum = spectrum[band_indices]
    peak_offset = int(np.argmax(band_spectrum))
    peak_index = int(band_indices[peak_offset])
    peak_value = float(spectrum[peak_index])

    band_sum = float(np.sum(band_spectrum))
    peak_ratio = peak_value / band_sum if band_sum > 0.0 else 0.0

    left = max(0, peak_offset - int(sharpness_radius_bins))
    right = min(len(band_spectrum), peak_offset + int(sharpness_radius_bins) + 1)
    neighborhood = band_spectrum[left:right]
    neighborhood_mean = float(np.mean(neighborhood)) if neighborhood.size else 0.0
    peak_sharpness = peak_value / neighborhood_mean if neighborhood_mean > 0.0 else 0.0

    return float(freqs[peak_index]), float(peak_ratio), float(peak_sharpness)


def frame_to_frame_variance(trajectory: np.ndarray) -> float:
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    velocities = np.linalg.norm(np.diff(values, axis=0), axis=1)
    return float(np.var(velocities, dtype=np.float64))


def spatial_coherence_from_positions(
    positions: np.ndarray | None,
    neighbor_pairs: Iterable[tuple[int, int]] | None,
) -> float | None:
    if positions is None or neighbor_pairs is None:
        return None

    values = np.asarray(positions, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 2:
        return None

    motion = np.linalg.norm(np.diff(values, axis=0), axis=2)
    correlations = []
    for a_index, b_index in neighbor_pairs:
        if not (0 <= int(a_index) < motion.shape[1] and 0 <= int(b_index) < motion.shape[1]):
            continue
        signal_a = motion[:, int(a_index)]
        signal_b = motion[:, int(b_index)]
        if signal_a.size < 2 or signal_b.size < 2:
            continue
        if np.allclose(signal_a, signal_a[0]) or np.allclose(signal_b, signal_b[0]):
            continue
        corr = np.corrcoef(signal_a, signal_b)[0, 1]
        if np.isfinite(corr):
            correlations.append(float(corr))

    if not correlations:
        return None
    return float(np.mean(correlations))


def subset_neighbor_pairs(
    selected_vertices: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    ordered_vertices = [int(vertex_id) for vertex_id in selected_vertices]
    index_by_vertex = {vertex_id: offset for offset, vertex_id in enumerate(ordered_vertices)}
    pairs = set()
    for vertex_id in ordered_vertices:
        a_index = index_by_vertex[vertex_id]
        for neighbor_id in adjacency[int(vertex_id)]:
            neighbor_index = index_by_vertex.get(int(neighbor_id))
            if neighbor_index is None or neighbor_index == a_index:
                continue
            pairs.add(tuple(sorted((a_index, neighbor_index))))
    return sorted(pairs)


def _run_filter(
    signal: np.ndarray,
    fps: float,
    order: int,
    mode: str,
    cutoff_hz: float | None = None,
    low_hz: float | None = None,
    high_hz: float | None = None,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if values.shape[0] <= _minimum_filter_length(order):
        return values.copy()

    nyquist = 0.5 * float(fps)
    if mode == "lowpass":
        if cutoff_hz is None:
            raise ValueError("cutoff_hz is required for lowpass filtering.")
        b, a = butter(int(order), float(cutoff_hz) / nyquist, btype="low")
        return _safe_filtfilt(b, a, values)

    if mode == "bandpass":
        if low_hz is None or high_hz is None:
            raise ValueError("low_hz and high_hz are required for bandpass filtering.")
        b_high, a_high = butter(int(order), float(low_hz) / nyquist, btype="high")
        high_passed = _safe_filtfilt(b_high, a_high, values)
        b_low, a_low = butter(int(order), float(high_hz) / nyquist, btype="low")
        return _safe_filtfilt(b_low, a_low, high_passed)

    raise ValueError(f"Unsupported filter mode: {mode}")


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, signal: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(filtfilt(b, a, signal, axis=0), dtype=np.float32)
    except ValueError:
        return np.asarray(signal, dtype=np.float32).copy()


def _minimum_filter_length(order: int) -> int:
    return max(8, 3 * max(int(order), 1) + 3)

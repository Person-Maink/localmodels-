from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG


FPS = 30.0
FILTER_ORDER = 3
HIGHPASS_CUTOFF = 2.0
LOWPASS_CUTOFF = 12.0


def _normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _infer_label(path_text, fallback):
    if path_text is None:
        return fallback

    p = Path(path_text)
    if p.name:
        return p.name
    return fallback


def _bandpass_filter(signal):
    nyq = 0.5 * FPS

    b_high, a_high = butter(FILTER_ORDER, HIGHPASS_CUTOFF / nyq, btype="high")
    signal = filtfilt(b_high, a_high, signal, axis=0)

    b_low, a_low = butter(FILTER_ORDER, LOWPASS_CUTOFF / nyq, btype="low")
    signal = filtfilt(b_low, a_low, signal, axis=0)

    return signal


def _analyze_mediapipe(csv_path, hand_idx):
    df = pd.read_csv(csv_path)
    centroids = []

    hand_label = "Right" if int(hand_idx) == 1 else "Left"

    for frame_id in sorted(df.frame_id.unique()):
        frame = df[(df.frame_id == frame_id) & (df.hand_id == hand_label)]
        if frame.empty:
            continue

        points = frame.sort_values("joint_id")[["x", "y", "z"]].to_numpy()
        centroids.append(points.mean(axis=0))

    if not centroids:
        raise ValueError(f"No usable MediaPipe records in '{csv_path}' for hand='{hand_label}'.")

    centroids = np.stack(centroids, axis=0)
    filtered = _bandpass_filter(centroids)
    magnitude = np.linalg.norm(filtered, axis=1)
    magnitude -= magnitude.mean()

    freqs, psd = welch(magnitude, fs=FPS, nperseg=min(512, len(magnitude)))
    dominant = float(freqs[np.argmax(psd)])
    rms = float(np.sqrt(np.mean(magnitude ** 2)))

    return {
        "centroids": centroids,
        "filtered": filtered,
        "magnitude": magnitude,
        "freqs": freqs,
        "psd": psd,
        "dominant": dominant,
        "rms": rms,
    }


def _plot_results(named_results):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    for i, (label, result) in enumerate(named_results):
        style = "-" if i == 0 else "--"
        color = f"C{i}"
        t = np.arange(len(result["magnitude"])) / FPS

        axes[0].plot(
            t,
            result["magnitude"],
            style,
            color=color,
            lw=1.5,
            label=f"{label} ({result['dominant']:.2f} Hz)",
        )

        axes[1].semilogy(
            result["freqs"],
            result["psd"],
            style,
            color=color,
            lw=1.5,
            label=label,
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

    axes[0].set_title("Hand displacement over time")
    axes[0].set_ylabel("Displacement magnitude")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Frequency spectrum (PSD)")
    axes[1].set_ylabel("Power")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_title("Per-axis filtered displacement")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Displacement")
    axes[2].legend(ncol=3)
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    csv_a = _normalize_optional_path(getattr(CONFIG, "MEDIAPIPE_ROOT", None))
    csv_b = _normalize_optional_path(getattr(CONFIG, "MEDIAPIPE_COMP", None))

    if csv_a is None and csv_b is None:
        raise ValueError("Both MediaPipe inputs are None. Set MEDIAPIPE_ROOT and/or MEDIAPIPE_COMP.")

    hand_idx = int(CONFIG.HAND_IDX)

    named_results = []

    if csv_a is not None:
        label_a = getattr(CONFIG, "MEDIAPIPE_LABEL", None) or _infer_label(csv_a, "MediaPipe A")
        named_results.append((label_a, _analyze_mediapipe(csv_a, hand_idx)))

    if csv_b is not None:
        label_b = getattr(CONFIG, "MEDIAPIPE_COMP_LABEL", None) or _infer_label(csv_b, "MediaPipe B")
        named_results.append((label_b, _analyze_mediapipe(csv_b, hand_idx)))

    _plot_results(named_results)

    for label, result in named_results:
        print(f"{label} dominant frequency: {result['dominant']:.2f} Hz")
        print(f"{label} RMS amplitude: {result['rms']:.6f}")


if __name__ == "__main__":
    main()

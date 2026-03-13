import argparse
import pickle
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
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


FPS = 30.0
FILTER_ORDER = 3
HIGHPASS_CUTOFF = 2.0
LOWPASS_CUTOFF = 12.0
DEFAULT_SLOT_NAMES = ("A", "B", "C")


@lru_cache(maxsize=1)
def _load_j_regressor(mano_right_path):
    with open(mano_right_path, "rb") as f:
        mano = pickle.load(f, encoding="latin1")
    return mano["J_regressor"]


def _normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _default_source_a():
    if hasattr(CONFIG, "ANALYSIS_SOURCE_A"):
        return getattr(CONFIG, "ANALYSIS_SOURCE_A")
    return getattr(CONFIG, "MODEL_ROOT", None)


def _default_source_b():
    if hasattr(CONFIG, "ANALYSIS_SOURCE_B"):
        return getattr(CONFIG, "ANALYSIS_SOURCE_B")
    return getattr(CONFIG, "MODEL_COMP", None)


def _default_all_model_sources():
    return [
        getattr(CONFIG, "WILOR_ROOT", None),
        getattr(CONFIG, "HAMBA_ROOT", None),
        getattr(CONFIG, "MEDIAPIPE_ROOT", None),
    ]


def _default_all_model_labels():
    return ["WILOR", "HAMBA", "MEDIAPIPE"]


def _source_kind(path_text):
    path = Path(path_text)
    if path.suffix.lower() == ".csv":
        return "mediapipe"
    return "model"


def _infer_label(path_text, fallback):
    if path_text is None:
        return fallback

    p = Path(path_text)
    if p.suffix.lower() == ".csv":
        return p.stem
    if p.name.lower() in {"meshes", "mesh", "npy"} and p.parent.name:
        return p.parent.name
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


def _finish_analysis(centroids):
    centroids = np.stack(centroids, axis=0)
    filtered = _bandpass_filter(centroids)
    magnitude = np.linalg.norm(filtered, axis=1)
    magnitude -= magnitude.mean()

    freqs, psd = welch(magnitude, fs=FPS, nperseg=min(512, len(magnitude)))
    dominant = float(freqs[np.argmax(psd)])
    rms = float(np.sqrt(np.mean(magnitude**2)))

    return {
        "centroids": centroids,
        "filtered": filtered,
        "magnitude": magnitude,
        "freqs": freqs,
        "psd": psd,
        "dominant": dominant,
        "rms": rms,
    }


def _analyze_model(root_dir, hand_idx, wrist_joint_idx, j_reg):
    centroids = []
    total_records = 0
    matched_records = 0

    for folder in list_frame_folders(root_dir):
        records = load_frame_records(folder, pattern="*.npy")
        total_records += len(records)

        centered_meshes = []
        for rec in records:
            if int(rec.get("right", -1)) != int(hand_idx):
                continue

            verts_world = rec["verts_world"]
            if verts_world.shape[0] != j_reg.shape[1]:
                raise ValueError(
                    f"Vertex count mismatch in {rec['path']}: got {verts_world.shape[0]}, expected {j_reg.shape[1]}"
                )

            joints = j_reg @ verts_world
            wrist = joints[int(wrist_joint_idx)]
            centered_meshes.append(verts_world - wrist)
            matched_records += 1

        if not centered_meshes:
            continue

        merged = np.concatenate(centered_meshes, axis=0)
        centroids.append(merged.mean(axis=0))

    if not centroids:
        raise ValueError(
            f"No usable model records in '{root_dir}' for HAND_IDX={hand_idx}. "
            f"Loaded {total_records} records, matched {matched_records}."
        )

    return _finish_analysis(centroids)


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

    return _finish_analysis(centroids)


def _analyze_source(source_path, hand_idx, wrist_joint_idx, j_reg):
    if _source_kind(source_path) == "mediapipe":
        return _analyze_mediapipe(source_path, hand_idx)
    return _analyze_model(source_path, hand_idx, wrist_joint_idx, j_reg)


def _resolve_entries(overrides):
    all_models = bool(overrides.get("all_models", False))
    sources_override = overrides.get("sources", None)
    labels_override = overrides.get("labels", None)

    if sources_override is not None:
        raw_sources = list(sources_override)
    elif all_models:
        raw_sources = _default_all_model_sources()
    else:
        raw_sources = [
            overrides.get("source_a", _default_source_a()),
            overrides.get("source_b", _default_source_b()),
        ]

    sources = [_normalize_optional_path(source) for source in raw_sources]
    sources = [source for source in sources if source is not None]
    if not sources:
        raise ValueError(
            "No analysis sources resolved. Set ANALYSIS_SOURCE_A/B in FILENAME.py, "
            "pass explicit sources, or use --all-models with configured WILOR/HAMBA/MEDIAPIPE roots."
        )

    if labels_override is not None:
        if len(labels_override) != len(raw_sources):
            raise ValueError("labels length must match sources length when both are provided.")
        raw_labels = list(labels_override)
    elif all_models and sources_override is None:
        raw_labels = _default_all_model_labels()
    else:
        raw_labels = []
        if len(raw_sources) >= 1:
            raw_labels.append(
                overrides.get("label_a") or getattr(CONFIG, "ANALYSIS_LABEL_A", None) or _infer_label(raw_sources[0], "Source A")
            )
        if len(raw_sources) >= 2:
            raw_labels.append(
                overrides.get("label_b") or getattr(CONFIG, "ANALYSIS_LABEL_B", None) or _infer_label(raw_sources[1], "Source B")
            )
        for idx in range(2, len(raw_sources)):
            raw_labels.append(_infer_label(raw_sources[idx], f"Source {idx + 1}"))

    entries = []
    for idx, source in enumerate(raw_sources):
        normalized = _normalize_optional_path(source)
        if normalized is None:
            continue
        label = raw_labels[idx] if idx < len(raw_labels) else None
        entries.append(
            {
                "slot": DEFAULT_SLOT_NAMES[idx] if idx < len(DEFAULT_SLOT_NAMES) else f"S{idx + 1}",
                "source": normalized,
                "kind": _source_kind(normalized),
                "label": label or _infer_label(normalized, f"Source {idx + 1}"),
            }
        )
    return entries


def run_compare_analysis(config_overrides=None):
    overrides = config_overrides or {}
    entries = _resolve_entries(overrides)
    hand_idx = int(overrides.get("hand_idx", CONFIG.HAND_IDX))
    wrist_joint_idx = int(overrides.get("wrist_joint_idx", CONFIG.WRIST_JOINT_IDX))

    j_reg = None
    if any(entry["kind"] == "model" for entry in entries):
        try:
            j_reg = _load_j_regressor(str(CONFIG.MANO_RIGHT_PATH))
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Model analysis needs MANO pickle dependencies (missing module while loading MANO_RIGHT_PATH). "
                "Install the missing module (commonly 'chumpy') or switch sources to MediaPipe CSV inputs."
            ) from exc

    resolved_entries = []
    for entry in entries:
        resolved_entries.append(
            {
                **entry,
                "result": _analyze_source(entry["source"], hand_idx, wrist_joint_idx, j_reg),
            }
        )

    return {
        "analysis": "compare",
        "hand_idx": hand_idx,
        "wrist_joint_idx": wrist_joint_idx,
        "entries": resolved_entries,
    }


def build_compare_figure(analysis_data, figsize_inches=(13, 11), dpi=100):
    fig, axes = plt.subplots(3, 1, figsize=figsize_inches, dpi=dpi)
    axis_labels = ["x", "y", "z"]

    for i, entry in enumerate(analysis_data["entries"]):
        label = entry["label"]
        result = entry["result"]

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

        for axis_i, axis_name in enumerate(axis_labels):
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

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare frequency analyses across configured sources.")
    parser.add_argument("--all-models", action="store_true", help="Compare WiLoR, Hamba, and MediaPipe together.")
    args = parser.parse_args()

    analysis_data = run_compare_analysis({"all_models": args.all_models})
    fig = build_compare_figure(analysis_data)
    plt.show()
    plt.close(fig)

    for entry in analysis_data["entries"]:
        label = entry["label"]
        result = entry["result"]
        print(f"{label} dominant frequency: {result['dominant']:.2f} Hz")
        print(f"{label} RMS amplitude: {result['rms']:.6f}")


if __name__ == "__main__":
    main()

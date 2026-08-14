from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work
import FILENAME as CONFIG
from analysis_metrics import dominant_frequency_metrics, finish_motion_analysis
from fusion_utils import (
    DEFAULT_ANCHOR_VERTICES,
    align_sequence_to_reference,
    load_matched_sequences,
    raw_average_fusion,
    tremor_residual_fusion,
    wrist_center,
    wrist_center_with_regressor,
)
from mano_pickle import load_mano_pickle


FPS = 30.0
FILTER_ORDER = 4
PSD_NPERSEG = 512
LINE_STYLES = ("-", "--", "-.", ":", (0, (6, 2)), (0, (3, 1, 1, 1)))


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def parse_int_list(text: str | None) -> list[int]:
    if text is None or str(text).strip() == "":
        return list(DEFAULT_ANCHOR_VERTICES)
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if len(values) < 3:
        raise argparse.ArgumentTypeError("Expected at least three comma-separated anchor vertices.")
    return values


def default_output_dir() -> Path:
    return Path(PROJECT_ROOT) / "analysis_images" / "fusion"


def load_mano_regressor():
    try:
        mano = load_mano_pickle(str(CONFIG.MANO_RIGHT_PATH))
        return np.asarray(mano["J_regressor"], dtype=np.float32)
    except Exception as exc:
        print(
            "[fusion] Warning: could not load MANO J_regressor; "
            f"falling back to wrist vertex 0. Details: {exc}"
        )
        return None


def center_sequence(verts: np.ndarray, j_reg: np.ndarray | None, wrist_joint_idx: int):
    if j_reg is None:
        return wrist_center(verts)
    return wrist_center_with_regressor(verts, j_reg, wrist_joint_idx)


def build_variants(
    wilor_centered: np.ndarray,
    stride_centered: np.ndarray,
    fps: float,
    alpha_values: list[float],
    residual_weights: list[float],
    band: tuple[float, float],
) -> list[dict]:
    variants = [
        {"variant_name": "wilor_wrist_centered", "verts": wilor_centered, "alpha": None, "residual_weight": None},
        {"variant_name": "stride_wrist_centered", "verts": stride_centered, "alpha": None, "residual_weight": None},
    ]
    for alpha in alpha_values:
        variants.append(
            {
                "variant_name": f"raw_avg_alpha_{_format_param(alpha)}",
                "verts": raw_average_fusion(wilor_centered, stride_centered, alpha),
                "alpha": float(alpha),
                "residual_weight": None,
            }
        )
    for weight in residual_weights:
        variants.append(
            {
                "variant_name": f"residual_fusion_lambda_{_format_param(weight)}",
                "verts": tremor_residual_fusion(
                    wilor_centered,
                    stride_centered,
                    fps=fps,
                    band=band,
                    residual_weight=weight,
                    filter_order=FILTER_ORDER,
                ),
                "alpha": None,
                "residual_weight": float(weight),
            }
        )
    return variants


def analyze_variant(
    variant: dict,
    fps: float,
    band: tuple[float, float],
) -> dict:
    verts = np.asarray(variant["verts"], dtype=np.float32)
    trajectory = verts.mean(axis=1)
    result = finish_motion_analysis(
        list(trajectory),
        fps=float(fps),
        filter_kind="bandpass",
        filter_order=FILTER_ORDER,
        band_low_hz=float(band[0]),
        band_high_hz=float(band[1]),
        psd_nperseg=PSD_NPERSEG,
    )

    scalar = np.linalg.norm(trajectory - trajectory.mean(axis=0, keepdims=True), axis=1)
    scalar = scalar - scalar.mean()
    freqs, psd = welch(scalar, fs=float(fps), nperseg=min(PSD_NPERSEG, len(scalar)))
    band_mask = (freqs >= float(band[0])) & (freqs <= float(band[1]))
    high_mask = freqs > float(band[1])
    peak_power = 0.0
    if np.any(band_mask):
        peak_power = float(np.max(psd[band_mask]))
    band_energy = _integrate_psd(freqs[band_mask], psd[band_mask])
    total_psd_energy = _integrate_psd(freqs, psd)
    high_freq_energy = _integrate_psd(freqs[high_mask], psd[high_mask])
    peak_ratio = peak_power / band_energy if band_energy > 0.0 else 0.0
    dominant_hz, _, _ = dominant_frequency_metrics(
        scalar,
        fps=float(fps),
        band_low_hz=float(band[0]),
        band_high_hz=float(band[1]),
    )

    result.update(
        {
            "scalar": scalar.astype(np.float32),
            "freqs": np.asarray(freqs, dtype=np.float32),
            "psd": np.asarray(psd, dtype=np.float32),
            "dominant": float(dominant_hz),
            "peak_power": float(peak_power),
            "peak_ratio": float(peak_ratio),
            "rms": float(np.sqrt(np.mean(np.square(scalar, dtype=np.float64)))),
            "high_freq_energy": float(high_freq_energy),
            "total_psd_energy": float(total_psd_energy),
        }
    )
    return result


def run_fusion_analysis(config_overrides: dict | None = None) -> dict:
    cfg = dict(config_overrides or {})
    clip = str(cfg["clip"])
    source_a = str(cfg["source_a"])
    source_b = str(cfg["source_b"])
    fps = float(cfg.get("fps") or FPS)
    hand_side = int(cfg.get("hand_side", CONFIG.HAND_IDX))
    band = (float(cfg.get("band_low", 4.0)), float(cfg.get("band_high", 12.0)))
    alpha_values = list(cfg.get("alpha_values", [0.0, 0.25, 0.5, 0.75, 1.0]))
    residual_weights = list(cfg.get("residual_weights", [0.5, 1.0, 1.5]))
    align = bool(cfg.get("align", False))
    anchor_vertices = list(cfg.get("anchor_vertices", DEFAULT_ANCHOR_VERTICES))
    output_dir = Path(cfg.get("output_dir") or default_output_dir()).expanduser().resolve()

    matched = load_matched_sequences(
        source_a,
        source_b,
        hand_side=hand_side,
        fps=fps,
    )

    if cfg.get("dry_run", False):
        return {
            "dry_run": True,
            "clip": clip,
            "source_a": source_a,
            "source_b": source_b,
            "source_a_frames": matched.source_a_frames,
            "source_b_frames": matched.source_b_frames,
            "matched_frame_count": int(len(matched.frame_ids)),
            "vertex_shape": list(matched.verts_a.shape[1:]),
            "fps": fps,
            "output_paths": output_paths(output_dir, clip),
        }

    j_reg = load_mano_regressor()
    wrist_joint_idx = int(cfg.get("wrist_joint_idx", CONFIG.WRIST_JOINT_IDX))
    wilor_centered, wilor_wrist = center_sequence(matched.verts_a, j_reg, wrist_joint_idx)
    stride_centered, stride_wrist = center_sequence(matched.verts_b, j_reg, wrist_joint_idx)

    if align:
        stride_centered = align_sequence_to_reference(
            moving=stride_centered,
            reference=wilor_centered,
            anchor_vertices=anchor_vertices,
            allow_scale=False,
        )

    variants = build_variants(
        wilor_centered=wilor_centered,
        stride_centered=stride_centered,
        fps=fps,
        alpha_values=alpha_values,
        residual_weights=residual_weights,
        band=band,
    )

    analyzed = []
    for variant in variants:
        result = analyze_variant(variant, fps=fps, band=band)
        analyzed.append({**variant, "result": result})

    rows = metric_rows(
        clip=clip,
        source_a=source_a,
        source_b=source_b,
        aligned=align,
        band=band,
        fps=fps,
        num_frames=len(matched.frame_ids),
        analyzed=analyzed,
    )
    paths = output_paths(output_dir, clip)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_csv(paths["metrics_csv"], rows)
    summary = build_summary(
        clip=clip,
        source_a=source_a,
        source_b=source_b,
        matched=matched,
        config={
            "fps": fps,
            "hand_side": hand_side,
            "band_low": band[0],
            "band_high": band[1],
            "alpha_values": alpha_values,
            "residual_weights": residual_weights,
            "align": align,
            "anchor_vertices": anchor_vertices,
            "point_mode": cfg.get("point_mode", "centroid"),
            "wrist_joint_idx": wrist_joint_idx,
        },
        rows=rows,
        freqs=analyzed[0]["result"]["freqs"],
    )
    save_json(paths["summary_json"], summary)
    save_psd_plot(paths["psd_svg"], analyzed, band, clip)
    save_timeseries_plot(paths["timeseries_svg"], analyzed, fps, clip)
    save_bar_plot(paths["bars_svg"], rows, clip)
    return {
        "dry_run": False,
        "clip": clip,
        "rows": rows,
        "summary": summary,
        "output_paths": paths,
    }


def metric_rows(
    clip: str,
    source_a: str,
    source_b: str,
    aligned: bool,
    band: tuple[float, float],
    fps: float,
    num_frames: int,
    analyzed: list[dict],
) -> list[dict]:
    rows = []
    for item in analyzed:
        result = item["result"]
        rows.append(
            {
                "clip": clip,
                "variant_name": item["variant_name"],
                "source_a": source_a,
                "source_b": source_b,
                "alpha": "" if item.get("alpha") is None else float(item["alpha"]),
                "residual_weight": "" if item.get("residual_weight") is None else float(item["residual_weight"]),
                "aligned": bool(aligned),
                "band_low": float(band[0]),
                "band_high": float(band[1]),
                "dominant_hz": float(result["dominant"]),
                "peak_power": float(result["peak_power"]),
                "peak_ratio": float(result["peak_ratio"]),
                "rms_amplitude": float(result["rms"]),
                "high_freq_energy": float(result["high_freq_energy"]),
                "total_psd_energy": float(result["total_psd_energy"]),
                "num_frames": int(num_frames),
                "fps": float(fps),
            }
        )
    return rows


def build_summary(
    clip: str,
    source_a: str,
    source_b: str,
    matched,
    config: dict,
    rows: list[dict],
    freqs: np.ndarray,
) -> dict:
    best_peak = max(rows, key=lambda row: float(row["peak_ratio"])) if rows else None
    wilor_row = next((row for row in rows if row["variant_name"] == "wilor_wrist_centered"), None)
    best_reduction = None
    if wilor_row is not None:
        tolerance = _frequency_bin_width(freqs)
        wilor_dominant = float(wilor_row["dominant_hz"])
        candidates = [
            row
            for row in rows
            if row["variant_name"] != "stride_wrist_centered"
            and abs(float(row["dominant_hz"]) - wilor_dominant) <= tolerance
        ]
        if candidates:
            best_reduction = min(candidates, key=lambda row: float(row["high_freq_energy"]))

    return {
        "clip": clip,
        "config": config,
        "matched_frame_count": int(len(matched.frame_ids)),
        "source_paths": {"source_a": source_a, "source_b": source_b},
        "source_frame_counts": {
            "source_a": int(matched.source_a_frames),
            "source_b": int(matched.source_b_frames),
        },
        "best_variant_by_peak_ratio": best_peak,
        "best_variant_by_high_freq_reduction_preserving_dominant_hz": best_reduction,
        "all_metrics": rows,
    }


def output_paths(output_dir: Path, clip: str) -> dict:
    safe_clip = _safe_filename(clip)
    return {
        "metrics_csv": str(output_dir / f"{safe_clip}_fusion_metrics.csv"),
        "summary_json": str(output_dir / f"{safe_clip}_fusion_summary.json"),
        "psd_svg": str(output_dir / f"{safe_clip}_fusion_psd.svg"),
        "timeseries_svg": str(output_dir / f"{safe_clip}_fusion_timeseries.svg"),
        "bars_svg": str(output_dir / f"{safe_clip}_fusion_bars.svg"),
    }


def save_metrics_csv(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "clip",
        "variant_name",
        "source_a",
        "source_b",
        "alpha",
        "residual_weight",
        "aligned",
        "band_low",
        "band_high",
        "dominant_hz",
        "peak_power",
        "peak_ratio",
        "rms_amplitude",
        "high_freq_energy",
        "total_psd_energy",
        "num_frames",
        "fps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)


def save_psd_plot(path: str, analyzed: list[dict], band: tuple[float, float], clip: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for index, item in enumerate(analyzed):
        result = item["result"]
        ax.semilogy(
            result["freqs"],
            result["psd"],
            label=f"{item['variant_name']} ({result['dominant']:.2f} Hz)",
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            linewidth=1.4,
        )
    ax.axvspan(float(band[0]), float(band[1]), color="gold", alpha=0.12)
    ax.set_title(f"{clip} fusion PSD overlay")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_timeseries_plot(path: str, analyzed: list[dict], fps: float, clip: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for index, item in enumerate(analyzed):
        scalar = item["result"]["scalar"]
        t = np.arange(len(scalar), dtype=np.float32) / float(fps)
        ax.plot(
            t,
            scalar,
            label=item["variant_name"],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            linewidth=1.2,
        )
    ax.set_title(f"{clip} centroid displacement time series")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Centered centroid displacement")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(path: str, rows: list[dict], clip: str) -> None:
    labels = [row["variant_name"] for row in rows]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.42
    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2.0, [float(row["peak_ratio"]) for row in rows], width=width, label="Peak ratio", color="C0")
    ax2.bar(
        x + width / 2.0,
        [float(row["high_freq_energy"]) for row in rows],
        width=width,
        label="High-frequency energy",
        color="C3",
        alpha=0.75,
    )
    ax1.set_title(f"{clip} fusion metric comparison")
    ax1.set_ylabel("Peak ratio")
    ax2.set_ylabel("High-frequency energy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.grid(True, axis="y")
    bars_1, labels_1 = ax1.get_legend_handles_labels()
    bars_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(bars_1 + bars_2, labels_1 + labels_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _integrate_psd(freqs: np.ndarray, psd: np.ndarray) -> float:
    if len(freqs) == 0:
        return 0.0
    if len(freqs) == 1:
        return float(psd[0])
    return float(np.trapz(psd, freqs))


def _frequency_bin_width(freqs: np.ndarray) -> float:
    values = np.asarray(freqs, dtype=np.float32)
    if values.size < 2:
        return 0.0
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _format_param(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _safe_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    return safe.strip("_") or "clip"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare and fuse WiLoR and STRIDE hand sequences.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--wilor-source", "--source-a", dest="source_a", required=True)
    parser.add_argument("--stride-source", "--source-b", dest="source_b", required=True)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--hand-side", type=int, default=int(CONFIG.HAND_IDX))
    parser.add_argument("--alpha-values", type=parse_float_list, default=parse_float_list("0,0.25,0.5,0.75,1.0"))
    parser.add_argument("--residual-weights", type=parse_float_list, default=parse_float_list("0.5,1.0,1.5"))
    parser.add_argument("--band-low", type=float, default=4.0)
    parser.add_argument("--band-high", type=float, default=12.0)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--anchor-vertices", type=parse_int_list, default=list(DEFAULT_ANCHOR_VERTICES))
    parser.add_argument("--point-mode", choices=("centroid",), default="centroid")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_fusion_analysis(vars(args))
    if result.get("dry_run"):
        print(json.dumps(_jsonable(result), indent=2))
        return

    print(f"Saved fusion metrics: {result['output_paths']['metrics_csv']}")
    print(f"Saved fusion summary: {result['output_paths']['summary_json']}")
    print(f"Saved fusion plots: {result['output_paths']['psd_svg']}")
    print(f"Saved fusion plots: {result['output_paths']['timeseries_svg']}")
    print(f"Saved fusion plots: {result['output_paths']['bars_svg']}")


if __name__ == "__main__":
    main()

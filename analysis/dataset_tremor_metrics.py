from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "analysis_matplotlib_cache"))
from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work
from dataset_tremor_common import (
    ANGLE_ORDER,
    FAMILY_CHOICES,
    compute_frequency_stability_hz_std,
    compute_palm_orientation_metrics,
    default_custom_output_dir,
    default_dataset_root,
    run_point_to_point_for_clip,
    select_dataset_clips,
    validate_clip_inventory,
    write_metrics_cache,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a clip-level point-to-point metric cache for the new tremor dataset "
            "and save it under analysis_images/custom/<family>/."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help="Path to the processed new_dataset video folder.",
    )
    parser.add_argument(
        "--family",
        choices=FAMILY_CHOICES,
        default="wilor",
        help="Processed output family to analyze. Families are kept separate by design.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where metrics.csv and manifest.json will be written.",
    )
    parser.add_argument(
        "--family-output-root",
        type=Path,
        default=None,
        help=(
            "Override the standard outputs/<family> directory. For an isolated "
            "STRIDE sweep config, pass outputs/stride_search/<stage>/<config-id>."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow the selected inventory to proceed even when the chosen family is missing some clips.",
    )
    parser.add_argument(
        "--clips",
        nargs="*",
        default=None,
        help="Optional fnmatch-style clip-id filters, for example 'clean_*' or 'tremor_simulation_line_*'.",
    )
    parser.add_argument(
        "--people",
        nargs="*",
        default=None,
        help="Optional participant filters.",
    )
    parser.add_argument(
        "--angles",
        nargs="*",
        default=None,
        choices=ANGLE_ORDER,
        help="Optional camera-angle filters.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (args.output_dir or default_custom_output_dir(args.family)).expanduser().resolve()

    selected_clips = select_dataset_clips(
        dataset_root,
        clip_patterns=args.clips,
        people=args.people,
        angles=args.angles,
    )
    clip_rows, missing_clip_ids = validate_clip_inventory(
        selected_clips,
        family=args.family,
        allow_missing=bool(args.allow_missing),
        family_output_root=args.family_output_root,
    )

    started_at = time.time()
    metric_rows = []
    total = len(clip_rows)
    for index, clip_row in enumerate(clip_rows, start=1):
        clip_id = clip_row["clip_id"]
        source_path = clip_row["source_path"]
        print(f"[dataset metrics] {index}/{total} {clip_id}")
        _, result, fps = run_point_to_point_for_clip(source_path, label=clip_id)
        orientation = compute_palm_orientation_metrics(source_path)
        frequency_stability_hz_std, frequency_stability_window_count = compute_frequency_stability_hz_std(
            result["magnitude"],
            fps=float(fps),
        )

        expected_hz = clip_row["expected_hz"]
        dominant_hz = float(result["dominant"])
        peak_freq_error_hz = None
        if expected_hz is not None:
            peak_freq_error_hz = abs(dominant_hz - float(expected_hz))

        metric_rows.append(
            {
                **clip_row,
                "dominant_hz": dominant_hz,
                "fft_peak_hz": float(result.get("fft_peak_hz", 0.0)),
                "peak_freq_error_hz": peak_freq_error_hz,
                "peak_ratio": float(result["peak_ratio"]),
                "peak_sharpness": float(result["peak_sharpness"]),
                "temporal_noise": float(result["temporal_noise"]),
                "spatial_coherence": None if result.get("spatial_coherence") is None else float(result["spatial_coherence"]),
                "rms_amplitude": float(result["rms"]),
                "palm_orientation_mean_deg": orientation["palm_orientation_mean_deg"],
                "palm_orientation_var_deg": orientation["palm_orientation_var_deg"],
                "palm_orientation_usable_frames": int(orientation["palm_orientation_usable_frames"]),
                "frequency_stability_hz_std": frequency_stability_hz_std,
                "frequency_stability_window_count": int(frequency_stability_window_count),
                "num_samples": int(len(result["magnitude"])),
            }
        )

    elapsed_seconds = time.time() - started_at
    manifest = {
        "dataset_root": str(dataset_root),
        "family": args.family,
        "family_output_root": None if args.family_output_root is None else str(args.family_output_root.expanduser().resolve()),
        "output_dir": str(output_dir),
        "allow_missing": bool(args.allow_missing),
        "filters": {
            "clips": args.clips or [],
            "people": args.people or [],
            "angles": args.angles or [],
        },
        "selected_clip_count": len(clip_rows),
        "missing_clip_count": len(missing_clip_ids),
        "missing_clip_ids": missing_clip_ids,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
    }
    metrics_path, manifest_path = write_metrics_cache(output_dir, metric_rows, manifest)
    print(f"Saved dataset metrics CSV: {metrics_path}")
    print(f"Saved dataset metrics manifest: {manifest_path}")
    print(
        f"Dataset metric build complete: clips={len(clip_rows)}, "
        f"missing={len(missing_clip_ids)}, elapsed={elapsed_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()

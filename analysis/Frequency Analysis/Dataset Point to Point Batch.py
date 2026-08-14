from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "analysis_matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work
from dataset_tremor_common import (
    ANGLE_ORDER,
    FAMILY_CHOICES,
    default_custom_output_dir,
    default_dataset_root,
    default_graph1_output_dir,
    ensure_dir,
    filter_metrics_frame,
    load_metrics_cache,
    run_point_to_point_for_sources,
)

GRAPH1_FIGSIZE_INCHES = (13.2, 10)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one standard point-to-point overlay figure per dataset condition "
            "using the cached dataset_tremor_metrics output."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help="Path to the processed new_dataset video folder. Used only for filter consistency.",
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=FAMILY_CHOICES,
        help="Processed output family to plot.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory containing the dataset metrics cache. Defaults to analysis_images/custom/<family>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the overlay figures will be written.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Accepted for interface symmetry with the cache builder. The overlay script only reads the cache.",
    )
    parser.add_argument(
        "--clips",
        nargs="*",
        default=None,
        help="Optional fnmatch-style clip-id filters.",
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
    cache_dir = (args.cache_dir or default_custom_output_dir(args.family)).expanduser().resolve()
    metrics_df, manifest = load_metrics_cache(cache_dir)
    filtered = filter_metrics_frame(metrics_df, clip_patterns=args.clips, people=args.people, angles=args.angles)
    if filtered.empty:
        raise RuntimeError("No cached metric rows matched the requested graph-1 filters.")

    output_dir = ensure_dir(args.output_dir or default_graph1_output_dir(args.family))
    point_module = None
    saved_count = 0
    for condition_key, group in filtered.groupby("condition_without_person", sort=True):
        ordered = group.sort_values("person", kind="stable")
        sources = ordered["source_path"].astype(str).tolist()
        labels = ordered["person"].astype(str).str.title().tolist()
        analysis_data = run_point_to_point_for_sources(sources, labels=labels)
        if point_module is None:
            from dataset_tremor_common import load_point_to_point_module

            point_module = load_point_to_point_module()
        fig = point_module.build_point_to_point_figure(analysis_data, figsize_inches=GRAPH1_FIGSIZE_INCHES)
        fig_path = output_dir / f"{condition_key}.svg"
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved_count += 1
        print(f"[dataset graph1] Saved {fig_path}")

    print(
        f"Saved dataset point-to-point overlays: {saved_count} figure(s) "
        f"from cache family='{args.family}' at {output_dir}"
    )
    if manifest.get("missing_clip_count"):
        print(
            "[dataset graph1] Note: the cache was built with missing clips allowed; "
            "some condition groups may contain fewer than five participants."
        )


if __name__ == "__main__":
    main()

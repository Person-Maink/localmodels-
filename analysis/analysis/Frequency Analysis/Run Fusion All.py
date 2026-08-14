from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work
import FILENAME as CONFIG


THIS_DIR = Path(__file__).resolve().parent


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_ALL = _load_module("freq_run_all_for_fusion", THIS_DIR / "Run All.py")
FUSION = _load_module("freq_wilor_stride_fusion", THIS_DIR / "WiLoR STRIDE Fusion.py")


def default_output_dir() -> Path:
    return Path(PROJECT_ROOT) / "analysis_images" / "fusion"


def build_pairs(outputs_root: Path) -> list[tuple[object, object]]:
    pools = {
        "wilor_all": RUN_ALL._discover_model_family(outputs_root, "wilor"),
        "stride_all": RUN_ALL._discover_stride(outputs_root),
    }
    wilor_by_match = {}
    for item in pools["wilor_all"]:
        wilor_by_match.setdefault(item.match_id, []).append(item)

    stride_by_match = {}
    for item in pools["stride_all"]:
        stride_by_match.setdefault(item.match_id, []).append(item)

    pairs = []
    for match_id in sorted(set(wilor_by_match) & set(stride_by_match)):
        for wilor_item in sorted(wilor_by_match[match_id], key=lambda item: (item.display_id, item.path)):
            for stride_item in sorted(stride_by_match[match_id], key=lambda item: (item.display_id, item.path)):
                pairs.append((wilor_item, stride_item))
    return pairs


def save_combined_csv(path: Path, rows: list[dict]) -> None:
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
        "status",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_batch(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.expanduser().resolve()
    pairs = build_pairs(args.outputs_root.expanduser().resolve())
    rows = []
    planned = []

    for wilor_item, stride_item in pairs:
        clip = wilor_item.match_id
        planned.append(
            {
                "clip": clip,
                "wilor": wilor_item.path,
                "stride": stride_item.path,
            }
        )
        if args.dry_run:
            continue

        try:
            result = FUSION.run_fusion_analysis(
                {
                    "clip": clip,
                    "source_a": wilor_item.path,
                    "source_b": stride_item.path,
                    "output_dir": output_dir,
                    "fps": args.fps,
                    "hand_side": args.hand_side,
                    "alpha_values": FUSION.parse_float_list(args.alpha_values),
                    "residual_weights": FUSION.parse_float_list(args.residual_weights),
                    "band_low": args.band_low,
                    "band_high": args.band_high,
                    "align": args.align,
                    "anchor_vertices": FUSION.parse_int_list(args.anchor_vertices),
                    "point_mode": "centroid",
                }
            )
            for row in result["rows"]:
                rows.append({**row, "status": "ok", "error": ""})
            print(f"[fusion batch] OK: {clip}")
        except Exception as exc:
            error = str(exc)
            print(f"[fusion batch] Warning: failed {clip}: {error}")
            rows.append(
                {
                    "clip": clip,
                    "source_a": wilor_item.path,
                    "source_b": stride_item.path,
                    "status": "failed",
                    "error": error,
                }
            )

    combined_path = output_dir / "all_fusion_metrics.csv"
    if not args.dry_run:
        save_combined_csv(combined_path, rows)

    return {
        "planned_pairs": planned,
        "pair_count": len(planned),
        "combined_csv": str(combined_path),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WiLoR/STRIDE fusion analysis for all matched clips.")
    parser.add_argument("--outputs-root", type=Path, default=Path(CONFIG.OUTPUTS_ROOT))
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hand-side", type=int, default=int(CONFIG.HAND_IDX))
    parser.add_argument("--alpha-values", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--residual-weights", default="0.5,1.0,1.5")
    parser.add_argument("--band-low", type=float, default=4.0)
    parser.add_argument("--band-high", type=float, default=12.0)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--anchor-vertices", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_batch(args)
    print(f"Matched WiLoR/STRIDE pairs: {result['pair_count']}")
    if args.dry_run:
        for item in result["planned_pairs"][:20]:
            print(f"  {item['clip']}: {item['wilor']} :: {item['stride']}")
        if result["pair_count"] > 20:
            print(f"  ... {result['pair_count'] - 20} more")
    else:
        print(f"Saved combined fusion metrics: {result['combined_csv']}")


if __name__ == "__main__":
    main()

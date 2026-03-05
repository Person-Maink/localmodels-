import argparse
import csv
import hashlib
import importlib.util
import json
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# Headless backend for batch image export.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG

THIS_DIR = Path(__file__).resolve().parent
COMP_SUFFIXES = ("_amplified_modified", "_amplified", "_modified")

SCENARIOS = (
    ("hamba_vs_hamba_comp", "hamba_all", "hamba_comp"),
    ("wilor_vs_wilor_comp", "wilor_all", "wilor_comp"),
    ("mediapipe_vs_mediapipe_comp", "mediapipe_all", "mediapipe_comp"),
    ("wilor_vs_hamba", "wilor_all", "hamba_all"),
    ("wilor_vs_mediapipe", "wilor_all", "mediapipe_all"),
    ("mediapipe_vs_hamba", "mediapipe_all", "hamba_all"),
)
SAME_CLIP_COMP_SCENARIOS = {
    "hamba_vs_hamba_comp",
    "wilor_vs_wilor_comp",
    "mediapipe_vs_mediapipe_comp",
}
CROSS_MODEL_SAME_CLIP_SCENARIOS = {
    "wilor_vs_hamba",
    "wilor_vs_mediapipe",
    "mediapipe_vs_hamba",
}


@dataclass(frozen=True)
class SourceItem:
    family: str
    kind: str
    clip_id: str
    path: str
    is_comp: bool


@dataclass(frozen=True)
class PairItem:
    scenario_id: str
    source_a: SourceItem
    source_b: SourceItem
    pair_id: str


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _int_config(name, fallback):
    try:
        return int(getattr(CONFIG, name, fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _default_output_dir():
    raw = getattr(CONFIG, "ANALYSIS_OUTPUT_DIR", None)
    if raw:
        return Path(str(raw))

    return Path(CONFIG.OUTPUTS_ROOT) / "analysis_images"


def _parse_args():
    parser = argparse.ArgumentParser(description="Run exhaustive frequency analyses and save figure/metric artifacts.")
    parser.add_argument("--width-px", type=int, default=_int_config("ANALYSIS_IMAGE_WIDTH_PX", 1920))
    parser.add_argument("--height-px", type=int, default=_int_config("ANALYSIS_IMAGE_HEIGHT_PX", 1080))
    parser.add_argument("--dpi", type=int, default=_int_config("ANALYSIS_IMAGE_DPI", 100))
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--dry-run", action="store_true", help="Build and report scenario/pair matrix without running analyses.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Run only the first N pairs after deterministic ordering.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Repeatable scenario filter. Options: " + ", ".join(scenario_id for scenario_id, _, _ in SCENARIOS),
    )
    return parser.parse_args()


def _is_comp_clip(name):
    return any(name.endswith(suffix) for suffix in COMP_SUFFIXES)


def _discover_model_family(outputs_root, family):
    family_root = Path(outputs_root) / family
    items = []
    if not family_root.exists():
        return items

    for mesh_dir in sorted(family_root.glob("*/meshes"), key=lambda p: p.parent.name):
        if not mesh_dir.is_dir():
            continue

        clip_id = mesh_dir.parent.name
        items.append(
            SourceItem(
                family=family,
                kind="model",
                clip_id=clip_id,
                path=str(mesh_dir.resolve()),
                is_comp=_is_comp_clip(clip_id),
            )
        )

    return items


def _discover_mediapipe(outputs_root):
    keypoints_dir = Path(outputs_root) / "mediapipe" / "keypoints"
    items = []
    if not keypoints_dir.exists():
        return items

    for csv_path in sorted(keypoints_dir.glob("*_keypoints.csv"), key=lambda p: p.name):
        if not csv_path.is_file() or csv_path.name.startswith(".~lock."):
            continue

        clip_id = csv_path.name[: -len("_keypoints.csv")]
        items.append(
            SourceItem(
                family="mediapipe",
                kind="mediapipe",
                clip_id=clip_id,
                path=str(csv_path.resolve()),
                is_comp=_is_comp_clip(clip_id),
            )
        )

    return items


def _slug(text):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "item"


def _canonical_clip_id(clip_id):
    base = clip_id
    changed = True
    while changed:
        changed = False
        for suffix in COMP_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                changed = True
                break
    return base


def _build_pair_id(scenario_id, source_a, source_b):
    payload = f"{scenario_id}|{source_a.path}|{source_b.path}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{scenario_id}__{_slug(source_a.clip_id)}__{_slug(source_b.clip_id)}__{digest}"


def _build_source_pools(outputs_root):
    hamba_all = _discover_model_family(outputs_root, "hamba")
    wilor_all = _discover_model_family(outputs_root, "wilor")
    mediapipe_all = _discover_mediapipe(outputs_root)

    pools = {
        "hamba_all": hamba_all,
        "hamba_comp": [item for item in hamba_all if item.is_comp],
        "wilor_all": wilor_all,
        "wilor_comp": [item for item in wilor_all if item.is_comp],
        "mediapipe_all": mediapipe_all,
        "mediapipe_comp": [item for item in mediapipe_all if item.is_comp],
    }

    discovery = {
        "hamba": {"all": len(pools["hamba_all"]), "comp": len(pools["hamba_comp"])},
        "wilor": {"all": len(pools["wilor_all"]), "comp": len(pools["wilor_comp"])},
        "mediapipe": {"all": len(pools["mediapipe_all"]), "comp": len(pools["mediapipe_comp"])},
    }

    return pools, discovery


def _build_scenarios_and_pairs(pools, requested_scenarios=None):
    allowed = {scenario_id for scenario_id, _, _ in SCENARIOS}

    if requested_scenarios:
        unknown = sorted(set(requested_scenarios) - allowed)
        if unknown:
            raise ValueError(
                "Unknown --scenario value(s): "
                + ", ".join(unknown)
                + ". Valid options: "
                + ", ".join(sorted(allowed))
            )
        enabled = set(requested_scenarios)
    else:
        enabled = allowed

    scenario_rows = []
    pairs = []

    for scenario_id, a_pool_name, b_pool_name in SCENARIOS:
        a_items = pools[a_pool_name]
        b_items = pools[b_pool_name]
        if scenario_id in SAME_CLIP_COMP_SCENARIOS:
            total_pairs = 0
            b_by_canonical = {}
            for item in b_items:
                b_by_canonical.setdefault(_canonical_clip_id(item.clip_id), []).append(item)

            for item in a_items:
                if item.is_comp:
                    continue
                total_pairs += len(b_by_canonical.get(_canonical_clip_id(item.clip_id), []))
        elif scenario_id in CROSS_MODEL_SAME_CLIP_SCENARIOS:
            total_pairs = 0
            b_by_canonical = {}
            for item in b_items:
                b_by_canonical.setdefault(_canonical_clip_id(item.clip_id), []).append(item)

            for item in a_items:
                total_pairs += len(b_by_canonical.get(_canonical_clip_id(item.clip_id), []))
        else:
            total_pairs = len(a_items) * len(b_items)

        scenario_row = {
            "scenario_id": scenario_id,
            "source_a_pool": a_pool_name,
            "source_b_pool": b_pool_name,
            "source_a_count": len(a_items),
            "source_b_count": len(b_items),
            "pair_count_total": total_pairs,
            "enabled": scenario_id in enabled,
        }
        scenario_rows.append(scenario_row)

        if scenario_id not in enabled:
            continue

        if scenario_id in SAME_CLIP_COMP_SCENARIOS:
            b_by_canonical = {}
            for item in b_items:
                b_by_canonical.setdefault(_canonical_clip_id(item.clip_id), []).append(item)

            for source_a in a_items:
                if source_a.is_comp:
                    continue
                canonical = _canonical_clip_id(source_a.clip_id)
                matches = b_by_canonical.get(canonical, [])
                for source_b in matches:
                    pairs.append(
                        PairItem(
                            scenario_id=scenario_id,
                            source_a=source_a,
                            source_b=source_b,
                            pair_id=_build_pair_id(scenario_id, source_a, source_b),
                        )
                    )
        elif scenario_id in CROSS_MODEL_SAME_CLIP_SCENARIOS:
            b_by_canonical = {}
            for item in b_items:
                b_by_canonical.setdefault(_canonical_clip_id(item.clip_id), []).append(item)

            for source_a in a_items:
                canonical = _canonical_clip_id(source_a.clip_id)
                matches = b_by_canonical.get(canonical, [])
                for source_b in matches:
                    pairs.append(
                        PairItem(
                            scenario_id=scenario_id,
                            source_a=source_a,
                            source_b=source_b,
                            pair_id=_build_pair_id(scenario_id, source_a, source_b),
                        )
                    )
        else:
            for source_a in a_items:
                for source_b in b_items:
                    pairs.append(
                        PairItem(
                            scenario_id=scenario_id,
                            source_a=source_a,
                            source_b=source_b,
                            pair_id=_build_pair_id(scenario_id, source_a, source_b),
                        )
                    )

    return scenario_rows, pairs


def _entry_label(source):
    return f"{source.family.upper()}:{source.clip_id}"


def _make_row(pair, analysis_name, entry):
    result = entry["result"]
    return {
        "scenario": pair.scenario_id,
        "pair_id": pair.pair_id,
        "analysis": analysis_name,
        "status": "success",
        "slot": entry.get("slot", ""),
        "label": entry["label"],
        "source": entry.get("source", ""),
        "kind": entry.get("kind", "model"),
        "dominant_hz": f"{result['dominant']:.8f}",
        "rms_amplitude": f"{result['rms']:.12f}",
        "num_samples": int(len(result["magnitude"])),
    }


def _save_metrics_csv(path, rows):
    fieldnames = [
        "scenario",
        "pair_id",
        "analysis",
        "status",
        "slot",
        "label",
        "source",
        "kind",
        "dominant_hz",
        "rms_amplitude",
        "num_samples",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_entries(entries):
    summarized = []
    for entry in entries:
        result = entry["result"]
        summarized.append(
            {
                "slot": entry.get("slot", ""),
                "label": entry["label"],
                "source": entry.get("source", ""),
                "kind": entry.get("kind", "model"),
                "dominant_hz": float(result["dominant"]),
                "rms_amplitude": float(result["rms"]),
                "num_samples": int(len(result["magnitude"])),
            }
        )
    return summarized


def _print_discovery(discovery, scenario_rows, selected_pairs):
    print("Discovery counts:")
    for family in ("hamba", "wilor", "mediapipe"):
        counts = discovery[family]
        print(f"  {family}: all={counts['all']}, comp={counts['comp']}")

    print("Scenario matrix:")
    for row in scenario_rows:
        enabled = "enabled" if row["enabled"] else "disabled"
        print(
            f"  {row['scenario_id']}: "
            f"{row['source_a_pool']}({row['source_a_count']}) x "
            f"{row['source_b_pool']}({row['source_b_count']}) -> "
            f"{row['pair_count_total']} pairs [{enabled}]"
        )

    print(f"Selected pairs to process: {len(selected_pairs)}")


def main():
    args = _parse_args()
    if args.width_px <= 0 or args.height_px <= 0 or args.dpi <= 0:
        raise ValueError("--width-px, --height-px, and --dpi must be positive integers.")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be a positive integer when provided.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figsize_inches = (args.width_px / args.dpi, args.height_px / args.dpi)

    pools, discovery = _build_source_pools(Path(CONFIG.OUTPUTS_ROOT))
    scenario_rows, all_pairs = _build_scenarios_and_pairs(pools, args.scenario)

    selected_pairs = all_pairs
    if args.max_pairs is not None:
        selected_pairs = all_pairs[: args.max_pairs]

    selected_counts = {}
    for pair in selected_pairs:
        selected_counts[pair.scenario_id] = selected_counts.get(pair.scenario_id, 0) + 1
    for row in scenario_rows:
        row["pair_count_selected"] = int(selected_counts.get(row["scenario_id"], 0))

    _print_discovery(discovery, scenario_rows, selected_pairs)

    summary = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "width_px": int(args.width_px),
            "height_px": int(args.height_px),
            "dpi": int(args.dpi),
            "output_dir": str(output_dir),
            "dry_run": bool(args.dry_run),
            "max_pairs": args.max_pairs,
            "selected_scenarios": list(args.scenario or []),
        },
        "discovery": discovery,
        "scenarios": scenario_rows,
        "pairs": [],
        "totals": {
            "pairs_discovered": int(len(all_pairs)),
            "pairs_selected": int(len(selected_pairs)),
            "analysis_success": 0,
            "analysis_failed": 0,
            "analysis_skipped": 0,
        },
    }

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"

    if args.dry_run:
        _save_metrics_csv(metrics_path, rows=[])
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved metrics CSV: {metrics_path}")
        print(f"Saved summary JSON: {summary_path}")
        print("Dry run complete.")
        return 0

    compare_module = _load_module("freq_compare", THIS_DIR / "Compare.py")
    point_module = _load_module("freq_point_to_point", THIS_DIR / "Point to Point.py")

    metric_rows = []
    failures = 0

    for idx, pair in enumerate(selected_pairs, start=1):
        print(f"[{idx}/{len(selected_pairs)}] {pair.scenario_id} :: {pair.pair_id}")

        pair_summary = {
            "scenario": pair.scenario_id,
            "pair_id": pair.pair_id,
            "source_a": asdict(pair.source_a),
            "source_b": asdict(pair.source_b),
            "analyses": {},
        }

        compare_overrides = {
            "source_a": pair.source_a.path,
            "source_b": pair.source_b.path,
            "label_a": _entry_label(pair.source_a),
            "label_b": _entry_label(pair.source_b),
        }

        try:
            compare_data = compare_module.run_compare_analysis(compare_overrides)
            compare_fig = compare_module.build_compare_figure(compare_data, figsize_inches=figsize_inches, dpi=args.dpi)

            compare_image = output_dir / f"compare__{pair.pair_id}.png"
            compare_fig.savefig(compare_image, dpi=args.dpi, bbox_inches="tight")
            plt.close(compare_fig)

            for entry in compare_data["entries"]:
                metric_rows.append(_make_row(pair, "compare", entry))

            pair_summary["analyses"]["compare"] = {
                "status": "success",
                "image": str(compare_image),
                "entries": _summarize_entries(compare_data["entries"]),
            }
            summary["totals"]["analysis_success"] += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            summary["totals"]["analysis_failed"] += 1
            pair_summary["analyses"]["compare"] = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=10),
            }

        if pair.source_a.kind == "model" and pair.source_b.kind == "model":
            point_overrides = {
                "source_a": pair.source_a.path,
                "source_b": pair.source_b.path,
                "label_a": _entry_label(pair.source_a),
                "label_b": _entry_label(pair.source_b),
            }

            try:
                point_data = point_module.run_point_to_point_analysis(point_overrides)
                point_fig = point_module.build_point_to_point_figure(point_data, figsize_inches=figsize_inches, dpi=args.dpi)

                point_image = output_dir / f"point_to_point__{pair.pair_id}.png"
                point_fig.savefig(point_image, dpi=args.dpi, bbox_inches="tight")
                plt.close(point_fig)

                for entry in point_data["entries"]:
                    metric_rows.append(_make_row(pair, "point_to_point", entry))

                pair_summary["analyses"]["point_to_point"] = {
                    "status": "success",
                    "image": str(point_image),
                    "entries": _summarize_entries(point_data["entries"]),
                }
                summary["totals"]["analysis_success"] += 1
            except Exception as exc:  # noqa: BLE001
                failures += 1
                summary["totals"]["analysis_failed"] += 1
                pair_summary["analyses"]["point_to_point"] = {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=10),
                }
        else:
            summary["totals"]["analysis_skipped"] += 1
            pair_summary["analyses"]["point_to_point"] = {
                "status": "skipped",
                "skipped_reason": "Point-to-point requires model mesh sources for both inputs.",
            }

        summary["pairs"].append(pair_summary)

    _save_metrics_csv(metrics_path, metric_rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics CSV: {metrics_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(
        "Totals: "
        f"success={summary['totals']['analysis_success']}, "
        f"failed={summary['totals']['analysis_failed']}, "
        f"skipped={summary['totals']['analysis_skipped']}"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

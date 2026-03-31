import argparse
import csv
import hashlib
import importlib.util
import json
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    ("dynhamr_vs_wilor", "dynhamr_all", "wilor_all"),
    ("dynhamr_vs_hamba", "dynhamr_all", "hamba_all"),
    ("dynhamr_vs_mediapipe", "dynhamr_all", "mediapipe_all"),
    ("wilor_vs_mediapipe", "wilor_all", "mediapipe_all"),
    ("mediapipe_vs_hamba", "mediapipe_all", "hamba_all"),
)
THREE_MODEL_SCENARIOS = (
    ("wilor_vs_hamba_vs_dynhamr", ("wilor_all", "hamba_all", "dynhamr_all")),
    ("wilor_vs_hamba_vs_mediapipe", ("wilor_all", "hamba_all", "mediapipe_all")),
    ("wilor_vs_dynhamr_vs_mediapipe", ("wilor_all", "dynhamr_all", "mediapipe_all")),
    ("hamba_vs_dynhamr_vs_mediapipe", ("hamba_all", "dynhamr_all", "mediapipe_all")),
)
ALL_MODELS_SCENARIO_ID = "wilor_vs_hamba_vs_dynhamr_vs_mediapipe"
SCENARIO_OPTIONS = (
    tuple(scenario_id for scenario_id, _, _ in SCENARIOS)
    + tuple(scenario_id for scenario_id, _ in THREE_MODEL_SCENARIOS)
    + (ALL_MODELS_SCENARIO_ID,)
)
SAME_CLIP_COMP_SCENARIOS = {
    "hamba_vs_hamba_comp",
    "wilor_vs_wilor_comp",
    "mediapipe_vs_mediapipe_comp",
}
CROSS_MODEL_SAME_CLIP_SCENARIOS = {
    "wilor_vs_hamba",
    "dynhamr_vs_wilor",
    "dynhamr_vs_hamba",
    "dynhamr_vs_mediapipe",
    "wilor_vs_mediapipe",
    "mediapipe_vs_hamba",
}
_ANALYSIS_MODULES = None


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


@dataclass(frozen=True)
class MultiModelsItem:
    scenario_id: str
    sources: tuple[SourceItem, ...]
    group_id: str


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_analysis_modules():
    global _ANALYSIS_MODULES
    if _ANALYSIS_MODULES is None:
        _ANALYSIS_MODULES = (
            _load_module("freq_compare", THIS_DIR / "Compare.py"),
            _load_module("freq_point_to_point", THIS_DIR / "Point to Point.py"),
            _load_module("freq_multi_point_to_point", THIS_DIR / "Multi Point to Point.py"),
        )
    return _ANALYSIS_MODULES


def _int_config(name, fallback):
    try:
        return int(getattr(CONFIG, name, fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _default_output_dir():
    raw = getattr(CONFIG, "ANALYSIS_OUTPUT_DIR", None)
    if raw is not None:
        return Path(raw)

    return CONFIG.OUTPUTS_ROOT / "analysis_images"


def _parse_args():
    parser = argparse.ArgumentParser(description="Run exhaustive frequency analyses and save figure/metric artifacts.")
    parser.add_argument("--width-px", type=int, default=_int_config("ANALYSIS_IMAGE_WIDTH_PX", 1920))
    parser.add_argument("--height-px", type=int, default=_int_config("ANALYSIS_IMAGE_HEIGHT_PX", 1080))
    parser.add_argument("--dpi", type=int, default=_int_config("ANALYSIS_IMAGE_DPI", 100))
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--dry-run", action="store_true", help="Build and report scenario/pair matrix without running analyses.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Run only pair/4-way items whose expected output graph files are not both already present in --output-dir.",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Run only the first N discovered pair/4-way items after deterministic ordering.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes used after pair/group selection.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Repeatable scenario filter. Options: " + ", ".join(SCENARIO_OPTIONS),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        default=True,
        help="Add same-clip WiLoR/Hamba/DynHAMR/MediaPipe 4-way analyses on top of the existing pairwise runs.",
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


def _build_group_id(scenario_id, *sources):
    payload = "|".join([scenario_id] + [source.path for source in sources])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    clip_slug = "__".join(_slug(source.clip_id) for source in sources)
    return f"{scenario_id}__{clip_slug}__{digest}"


def _build_source_pools(outputs_root):
    hamba_all = _discover_model_family(outputs_root, "hamba")
    wilor_all = _discover_model_family(outputs_root, "wilor")
    dynhamr_all = _discover_model_family(outputs_root, "dynhamr")
    mediapipe_all = _discover_mediapipe(outputs_root)

    pools = {
        "hamba_all": hamba_all,
        "hamba_comp": [item for item in hamba_all if item.is_comp],
        "wilor_all": wilor_all,
        "wilor_comp": [item for item in wilor_all if item.is_comp],
        "dynhamr_all": dynhamr_all,
        "dynhamr_comp": [item for item in dynhamr_all if item.is_comp],
        "mediapipe_all": mediapipe_all,
        "mediapipe_comp": [item for item in mediapipe_all if item.is_comp],
    }

    discovery = {
        "hamba": {"all": len(pools["hamba_all"]), "comp": len(pools["hamba_comp"])},
        "wilor": {"all": len(pools["wilor_all"]), "comp": len(pools["wilor_comp"])},
        "dynhamr": {"all": len(pools["dynhamr_all"]), "comp": len(pools["dynhamr_comp"])},
        "mediapipe": {"all": len(pools["mediapipe_all"]), "comp": len(pools["mediapipe_comp"])},
    }

    return pools, discovery


def _build_multi_model_groups(scenario_id, pool_names, pools):
    grouped = []
    for pool_name in pool_names:
        items_by_canonical = {}
        for item in pools[pool_name]:
            items_by_canonical.setdefault(_canonical_clip_id(item.clip_id), []).append(item)
        grouped.append(items_by_canonical)

    shared = sorted(set.intersection(*(set(items_by_canonical) for items_by_canonical in grouped)))
    groups = []
    for canonical in shared:
        source_lists = [items_by_canonical[canonical] for items_by_canonical in grouped]
        stack = [([], 0)]
        while stack:
            chosen, depth = stack.pop()
            if depth == len(source_lists):
                groups.append(
                    MultiModelsItem(
                        scenario_id=scenario_id,
                        sources=tuple(chosen),
                        group_id=_build_group_id(scenario_id, *chosen),
                    )
                )
                continue
            for source in reversed(source_lists[depth]):
                stack.append((chosen + [source], depth + 1))

    return groups


def _build_scenarios_and_pairs(pools, requested_scenarios=None, include_all_models=False):
    allowed = {scenario_id for scenario_id, _, _ in SCENARIOS}
    if include_all_models:
        allowed.update(scenario_id for scenario_id, _ in THREE_MODEL_SCENARIOS)
        allowed.add(ALL_MODELS_SCENARIO_ID)

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
    all_model_items = []

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
            "item_kind": "pairs",
            "item_count_total": total_pairs,
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
                            source_a=source_a,
                            source_b=source_b,
                            scenario_id=scenario_id,
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

    if include_all_models:
        for scenario_id, pool_names in THREE_MODEL_SCENARIOS:
            groups = _build_multi_model_groups(scenario_id, pool_names, pools)
            scenario_rows.append(
                {
                    "scenario_id": scenario_id,
                    "source_a_pool": pool_names[0],
                    "source_b_pool": pool_names[1],
                    "source_c_pool": pool_names[2],
                    "source_a_count": len(pools[pool_names[0]]),
                    "source_b_count": len(pools[pool_names[1]]),
                    "source_c_count": len(pools[pool_names[2]]),
                    "item_kind": "multi_models",
                    "item_count_total": len(groups),
                    "group_size": 3,
                    "enabled": scenario_id in enabled,
                }
            )
            if scenario_id in enabled:
                all_model_items.extend(groups)

        all_groups = _build_multi_model_groups(
            ALL_MODELS_SCENARIO_ID,
            ("wilor_all", "hamba_all", "dynhamr_all", "mediapipe_all"),
            pools,
        )
        scenario_rows.append(
            {
                "scenario_id": ALL_MODELS_SCENARIO_ID,
                "source_a_pool": "wilor_all",
                "source_b_pool": "hamba_all",
                "source_c_pool": "dynhamr_all",
                "source_d_pool": "mediapipe_all",
                "source_a_count": len(pools["wilor_all"]),
                "source_b_count": len(pools["hamba_all"]),
                "source_c_count": len(pools["dynhamr_all"]),
                "source_d_count": len(pools["mediapipe_all"]),
                "item_kind": "multi_models",
                "item_count_total": len(all_groups),
                "group_size": 4,
                "enabled": ALL_MODELS_SCENARIO_ID in enabled,
            }
        )
        if ALL_MODELS_SCENARIO_ID in enabled:
            all_model_items.extend(all_groups)

    return scenario_rows, pairs, all_model_items


def _entry_label(source):
    return f"{source.family.upper()}:{source.clip_id}"


def _item_id(item):
    return getattr(item, "pair_id", getattr(item, "group_id", ""))


def _make_row(item, analysis_name, entry):
    result = entry["result"]
    return {
        "scenario": item.scenario_id,
        "pair_id": _item_id(item),
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


def _print_discovery(discovery, scenario_rows, selected_pairs, selected_all_models):
    print("Discovery counts:")
    for family in ("hamba", "wilor", "dynhamr", "mediapipe"):
        counts = discovery[family]
        print(f"  {family}: all={counts['all']}, comp={counts['comp']}")

    print("Scenario matrix:")
    for row in scenario_rows:
        enabled = "enabled" if row["enabled"] else "disabled"
        if row["item_kind"] == "multi_models":
            group_size = int(row.get("group_size", 0))
            pools_text = " x ".join(
                f"{row[f'source_{slot}_pool']}({row[f'source_{slot}_count']})"
                for slot in ("a", "b", "c", "d")
                if row.get(f"source_{slot}_pool") is not None
            )
            print(
                f"  {row['scenario_id']}: "
                f"{pools_text} -> "
                f"{row['item_count_total']} {group_size}-way groups [{enabled}]"
            )
        else:
            print(
                f"  {row['scenario_id']}: "
                f"{row['source_a_pool']}({row['source_a_count']}) x "
                f"{row['source_b_pool']}({row['source_b_count']}) -> "
                f"{row['item_count_total']} pairs [{enabled}]"
            )

    print(f"Selected pairs to process: {len(selected_pairs)}")
    print(f"Selected multi-model groups to process: {len(selected_all_models)}")


def _expected_pair_outputs(output_dir, pair):
    return (
        output_dir / f"compare__{pair.pair_id}.png",
        output_dir / f"point_to_point__{pair.pair_id}.png",
        output_dir / f"multi_point_to_point__{pair.pair_id}.png",
    )


def _expected_multi_model_outputs(output_dir, item):
    return (
        output_dir / f"compare_all_models__{item.group_id}.png",
        output_dir / f"point_to_point_all_models__{item.group_id}.png",
        output_dir / f"multi_point_to_point_all_models__{item.group_id}.png",
    )


def _filter_missing_items(output_dir, pairs, all_model_items):
    missing_pairs = []
    skipped_pairs = 0
    for pair in pairs:
        expected = _expected_pair_outputs(output_dir, pair)
        if all(path.exists() for path in expected):
            skipped_pairs += 1
            continue
        missing_pairs.append(pair)

    missing_all_models = []
    skipped_all_models = 0
    for item in all_model_items:
        expected = _expected_multi_model_outputs(output_dir, item)
        if all(path.exists() for path in expected):
            skipped_all_models += 1
            continue
        missing_all_models.append(item)

    return missing_pairs, missing_all_models, skipped_pairs, skipped_all_models


def _pair_payload(pair):
    return {
        "kind": "pair",
        "scenario_id": pair.scenario_id,
        "pair_id": pair.pair_id,
        "source_a": asdict(pair.source_a),
        "source_b": asdict(pair.source_b),
    }


def _all_models_payload(item):
    return {
        "kind": "all_models",
        "scenario_id": item.scenario_id,
        "group_id": item.group_id,
        "sources": [asdict(source) for source in item.sources],
    }


def _runtime_item_from_payload(item_kind, item):
    if item_kind == "pair":
        return PairItem(
            scenario_id=item["scenario_id"],
            source_a=SourceItem(**item["source_a"]),
            source_b=SourceItem(**item["source_b"]),
            pair_id=item["pair_id"],
        )
    return MultiModelsItem(
        scenario_id=item["scenario_id"],
        sources=tuple(SourceItem(**source) for source in item["sources"]),
        group_id=item["group_id"],
    )


def _run_worker_job(job):
    compare_module, point_module, multi_point_module = _get_analysis_modules()

    item_kind = job["item_kind"]
    item = job["item"]
    runtime_item = _runtime_item_from_payload(item_kind, item)
    output_dir = Path(job["output_dir"])
    figsize_inches = tuple(job["figsize_inches"])
    dpi = int(job["dpi"])

    metric_rows = []
    failure_count = 0

    if item_kind == "pair":
        pair_sources = [runtime_item.source_a, runtime_item.source_b]
        compare_overrides = {
            "source_a": runtime_item.source_a.path,
            "source_b": runtime_item.source_b.path,
            "label_a": _entry_label(runtime_item.source_a),
            "label_b": _entry_label(runtime_item.source_b),
        }
        point_overrides = dict(compare_overrides)
        multi_point_overrides = {
            "sources": [{"family": source.family, "path": source.path} for source in pair_sources],
        }
        item_summary = {
            "scenario": runtime_item.scenario_id,
            "pair_id": runtime_item.pair_id,
            "source_a": item["source_a"],
            "source_b": item["source_b"],
            "analyses": {},
        }
        compare_image = output_dir / f"compare__{runtime_item.pair_id}.png"
        point_image = output_dir / f"point_to_point__{runtime_item.pair_id}.png"
        multi_point_image = output_dir / f"multi_point_to_point__{runtime_item.pair_id}.png"
    else:
        sources = list(runtime_item.sources)
        labels = [_entry_label(source) for source in sources]
        compare_overrides = {
            "all_models": True,
            "sources": [source.path for source in sources],
            "labels": labels,
        }
        point_overrides = dict(compare_overrides)
        multi_point_overrides = {
            "sources": [{"family": source.family, "path": source.path} for source in sources],
        }
        item_summary = {
            "scenario": runtime_item.scenario_id,
            "group_id": runtime_item.group_id,
            "sources": item["sources"],
            "analyses": {},
        }
        compare_image = output_dir / f"compare_all_models__{runtime_item.group_id}.png"
        point_image = output_dir / f"point_to_point_all_models__{runtime_item.group_id}.png"
        multi_point_image = output_dir / f"multi_point_to_point_all_models__{runtime_item.group_id}.png"

    try:
        compare_data = compare_module.run_compare_analysis(compare_overrides)
        compare_fig = compare_module.build_compare_figure(compare_data, figsize_inches=figsize_inches, dpi=dpi)
        compare_fig.savefig(compare_image, dpi=dpi, bbox_inches="tight")
        plt.close(compare_fig)

        for entry in compare_data["entries"]:
            metric_rows.append(_make_row(runtime_item, "compare", entry))

        item_summary["analyses"]["compare"] = {
            "status": "success",
            "image": str(compare_image),
            "entries": _summarize_entries(compare_data["entries"]),
        }
    except Exception as exc:  # noqa: BLE001
        failure_count += 1
        item_summary["analyses"]["compare"] = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }

    try:
        point_data = point_module.run_point_to_point_analysis(point_overrides)
        point_fig = point_module.build_point_to_point_figure(point_data, figsize_inches=figsize_inches, dpi=dpi)
        point_fig.savefig(point_image, dpi=dpi, bbox_inches="tight")
        plt.close(point_fig)

        for entry in point_data["entries"]:
            metric_rows.append(_make_row(runtime_item, "point_to_point", entry))

        item_summary["analyses"]["point_to_point"] = {
            "status": "success",
            "image": str(point_image),
            "entries": _summarize_entries(point_data["entries"]),
        }
    except Exception as exc:  # noqa: BLE001
        failure_count += 1
        item_summary["analyses"]["point_to_point"] = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }

    try:
        multi_point_data = multi_point_module.run_multi_point_analysis(multi_point_overrides)
        multi_point_fig = multi_point_module.build_multi_point_figure(
            multi_point_data,
            figsize_inches=figsize_inches,
            dpi=dpi,
        )
        multi_point_fig.savefig(multi_point_image, dpi=dpi, bbox_inches="tight")
        plt.close(multi_point_fig)

        for entry in multi_point_data["entries"]:
            metric_rows.append(_make_row(runtime_item, "multi_point_to_point", entry))

        item_summary["analyses"]["multi_point_to_point"] = {
            "status": "success",
            "image": str(multi_point_image),
            "entries": _summarize_entries(multi_point_data["entries"]),
        }
    except Exception as exc:  # noqa: BLE001
        failure_count += 1
        item_summary["analyses"]["multi_point_to_point"] = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }

    success_count = sum(
        1
        for analysis in item_summary["analyses"].values()
        if analysis.get("status") == "success"
    )

    return {
        "item_kind": item_kind,
        "sequence": int(job["sequence"]),
        "scenario_id": item["scenario_id"],
        "item_id": job["item_id"],
        "summary": item_summary,
        "metric_rows": metric_rows,
        "analysis_success": int(success_count),
        "analysis_failed": int(failure_count),
    }


def main():
    args = _parse_args()
    if args.width_px <= 0 or args.height_px <= 0 or args.dpi <= 0:
        raise ValueError("--width-px, --height-px, and --dpi must be positive integers.")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be a positive integer when provided.")
    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figsize_inches = (args.width_px / args.dpi, args.height_px / args.dpi)

    pools, discovery = _build_source_pools(CONFIG.OUTPUTS_ROOT)
    scenario_rows, all_pairs, all_model_items = _build_scenarios_and_pairs(
        pools, args.scenario, include_all_models=args.all_models
    )

    selected_pairs = all_pairs
    selected_all_models = all_model_items

    skipped_existing_pairs = 0
    skipped_existing_all_models = 0
    if args.only_missing:
        selected_pairs, selected_all_models, skipped_existing_pairs, skipped_existing_all_models = _filter_missing_items(
            output_dir, selected_pairs, selected_all_models
        )

    if args.max_pairs is not None:
        selected_pairs = selected_pairs[: args.max_pairs]
        selected_all_models = selected_all_models[: args.max_pairs]

    selected_counts = {}
    for pair in selected_pairs:
        selected_counts[pair.scenario_id] = selected_counts.get(pair.scenario_id, 0) + 1
    for item in selected_all_models:
        selected_counts[item.scenario_id] = selected_counts.get(item.scenario_id, 0) + 1
    for row in scenario_rows:
        row["item_count_selected"] = int(selected_counts.get(row["scenario_id"], 0))

    _print_discovery(discovery, scenario_rows, selected_pairs, selected_all_models)

    summary = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "width_px": int(args.width_px),
            "height_px": int(args.height_px),
            "dpi": int(args.dpi),
            "workers": int(args.workers),
            "output_dir": str(output_dir),
            "dry_run": bool(args.dry_run),
            "only_missing": bool(args.only_missing),
            "max_pairs": args.max_pairs,
            "all_models": bool(args.all_models),
            "selected_scenarios": list(args.scenario or []),
        },
        "discovery": discovery,
        "scenarios": scenario_rows,
        "pairs": [],
        "all_models": [],
        "totals": {
            "pairs_discovered": int(len(all_pairs)),
            "pairs_selected": int(len(selected_pairs)),
            "all_models_discovered": int(len(all_model_items)),
            "all_models_selected": int(len(selected_all_models)),
            "pairs_skipped_existing": int(skipped_existing_pairs),
            "all_models_skipped_existing": int(skipped_existing_all_models),
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

    metric_rows = []
    failures = 0
    jobs = []
    for idx, pair in enumerate(selected_pairs, start=1):
        jobs.append(
            {
                "item_kind": "pair",
                "sequence": idx,
                "item_id": pair.pair_id,
                "item": _pair_payload(pair),
                "output_dir": str(output_dir),
                "figsize_inches": figsize_inches,
                "dpi": args.dpi,
            }
        )

    for idx, item in enumerate(selected_all_models, start=1):
        jobs.append(
            {
                "item_kind": "all_models",
                "sequence": idx,
                "item_id": item.group_id,
                "item": _all_models_payload(item),
                "output_dir": str(output_dir),
                "figsize_inches": figsize_inches,
                "dpi": args.dpi,
            }
        )

    pair_results = []
    all_model_results = []
    total_jobs = len(jobs)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(_run_worker_job, job): job for job in jobs}
        completed = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            print(f"[{completed}/{total_jobs}] {job['item']['scenario_id']} :: {job['item_id']}")

            result = future.result()
            metric_rows.extend(result["metric_rows"])
            failures += int(result["analysis_failed"])
            summary["totals"]["analysis_success"] += int(result["analysis_success"])
            summary["totals"]["analysis_failed"] += int(result["analysis_failed"])

            if result["item_kind"] == "pair":
                pair_results.append(result)
            else:
                all_model_results.append(result)

    pair_results.sort(key=lambda result: result["sequence"])
    all_model_results.sort(key=lambda result: result["sequence"])
    summary["pairs"] = [result["summary"] for result in pair_results]
    summary["all_models"] = [result["summary"] for result in all_model_results]

    metric_rows.sort(
        key=lambda row: (
            row["scenario"],
            row["pair_id"],
            row["analysis"],
            row["slot"],
            row["label"],
        )
    )
    _save_metrics_csv(metrics_path, metric_rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics CSV: {metrics_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(
        "Totals: "
        f"success={summary['totals']['analysis_success']}, "
        f"failed={summary['totals']['analysis_failed']}, "
        f"skipped={summary['totals']['analysis_skipped']}, "
        f"existing_pairs_skipped={summary['totals']['pairs_skipped_existing']}, "
        f"existing_all_models_skipped={summary['totals']['all_models_skipped_existing']}"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

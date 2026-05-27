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

from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work
import FILENAME as CONFIG

THIS_DIR = Path(__file__).resolve().parent
COMP_SUFFIXES = ("_amplified_modified", "_amplified", "_modified")
WILOR_FINETUNE_EXPERIMENT_ALIASES = {
    "main_static": "main_static_finetuning",
    "main_learnable": "main_learnable_finetnuing",
    "lora": "lora_finetuning",
}
SELECTED_WILOR_FINETUNE_ALIASES = (
    "main_static",
    "main_learnable",
    "lora",
)


def _dedupe_preserve_order(values):
    ordered = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _wilor_finetune_pool_names(alias):
    return f"wilor_finetune_{alias}_all", f"wilor_finetune_{alias}_comp"


def _wilor_finetune_all_models_scenario_id(alias):
    return f"all_models_with_{alias}"


def _selected_wilor_finetune_all_models_scenarios(selected_aliases):
    scenarios = []
    for alias in _dedupe_preserve_order(selected_aliases):
        alias_pool_name, _ = _wilor_finetune_pool_names(alias)
        scenarios.append(
            (
                _wilor_finetune_all_models_scenario_id(alias),
                ("wilor_all", alias_pool_name, "hamba_all", "dynhamr_all", "stride_all", "mediapipe_all"),
            )
        )
    return tuple(scenarios)


FINETUNE_ALL_MODEL_SCENARIOS = _selected_wilor_finetune_all_models_scenarios(SELECTED_WILOR_FINETUNE_ALIASES)
PAIRWISE_ANALYSES = (
    "compare",
    "point_to_point",
    "point_to_point_neighbor_sweep",
    "multi_point_to_point",
)
SINGLE_SOURCE_ANALYSES = ("wilor_camera_space", "beta_comparison", "beta_multi_point_to_point")
ANALYSIS_IDS = PAIRWISE_ANALYSES + SINGLE_SOURCE_ANALYSES
ANALYSIS_FOLDERS = {analysis_id: analysis_id for analysis_id in ANALYSIS_IDS}
MODEL_ONLY_PAIRWISE_ANALYSES = {"point_to_point_neighbor_sweep"}
SINGLE_SOURCE_ANALYSIS_FAMILIES = {
    "wilor_camera_space": {"wilor", "wilor_finetune", "dynhamr", "stride"},
    "beta_comparison": {"wilor", "wilor_finetune", "dynhamr", "stride"},
    "beta_multi_point_to_point": {"wilor", "wilor_finetune", "dynhamr", "stride"},
}

SCENARIOS = (
    ("hamba_vs_hamba_comp", "hamba_all", "hamba_comp"),
    ("wilor_vs_wilor_comp", "wilor_all", "wilor_comp"),
    ("wilor_finetune_vs_wilor_finetune", "wilor_finetune_all", "wilor_finetune_all"),
    ("wilor_finetune_vs_wilor_finetune_comp", "wilor_finetune_all", "wilor_finetune_comp"),
    ("stride_vs_stride_comp", "stride_all", "stride_comp"),
    ("mediapipe_vs_mediapipe_comp", "mediapipe_all", "mediapipe_comp"),
    ("wilor_finetune_vs_wilor", "wilor_finetune_all", "wilor_all"),
    ("wilor_finetune_vs_hamba", "wilor_finetune_all", "hamba_all"),
    ("wilor_finetune_vs_dynhamr", "wilor_finetune_all", "dynhamr_all"),
    ("wilor_finetune_vs_stride", "wilor_finetune_all", "stride_all"),
    ("wilor_finetune_vs_mediapipe", "wilor_finetune_all", "mediapipe_all"),
    ("wilor_vs_hamba", "wilor_all", "hamba_all"),
    ("dynhamr_vs_wilor", "dynhamr_all", "wilor_all"),
    ("dynhamr_vs_hamba", "dynhamr_all", "hamba_all"),
    ("dynhamr_vs_stride", "dynhamr_all", "stride_all"),
    ("dynhamr_vs_mediapipe", "dynhamr_all", "mediapipe_all"),
    ("wilor_vs_stride", "wilor_all", "stride_all"),
    ("wilor_vs_mediapipe", "wilor_all", "mediapipe_all"),
    ("stride_vs_hamba", "stride_all", "hamba_all"),
    ("stride_vs_mediapipe", "stride_all", "mediapipe_all"),
    ("mediapipe_vs_hamba", "mediapipe_all", "hamba_all"),
)
THREE_MODEL_SCENARIOS = (
    ("wilor_vs_hamba_vs_dynhamr", ("wilor_all", "hamba_all", "dynhamr_all")),
    ("wilor_vs_hamba_vs_stride", ("wilor_all", "hamba_all", "stride_all")),
    ("wilor_vs_hamba_vs_mediapipe", ("wilor_all", "hamba_all", "mediapipe_all")),
    ("wilor_vs_dynhamr_vs_stride", ("wilor_all", "dynhamr_all", "stride_all")),
    ("wilor_vs_dynhamr_vs_mediapipe", ("wilor_all", "dynhamr_all", "mediapipe_all")),
    ("wilor_vs_stride_vs_mediapipe", ("wilor_all", "stride_all", "mediapipe_all")),
    ("hamba_vs_dynhamr_vs_mediapipe", ("hamba_all", "dynhamr_all", "mediapipe_all")),
    ("hamba_vs_dynhamr_vs_stride", ("hamba_all", "dynhamr_all", "stride_all")),
    ("hamba_vs_stride_vs_mediapipe", ("hamba_all", "stride_all", "mediapipe_all")),
    ("dynhamr_vs_stride_vs_mediapipe", ("dynhamr_all", "stride_all", "mediapipe_all")),
)
ALL_MODELS_SCENARIO_ID = "wilor_vs_hamba_vs_dynhamr_vs_stride_vs_mediapipe"
SCENARIO_OPTIONS = (
    tuple(scenario_id for scenario_id, _, _ in SCENARIOS)
    + tuple(scenario_id for scenario_id, _ in THREE_MODEL_SCENARIOS)
    + tuple(scenario_id for scenario_id, _ in FINETUNE_ALL_MODEL_SCENARIOS)
    + (ALL_MODELS_SCENARIO_ID,)
)
SAME_CLIP_COMP_SCENARIOS = {
    "hamba_vs_hamba_comp",
    "wilor_vs_wilor_comp",
    "wilor_finetune_vs_wilor_finetune_comp",
    "stride_vs_stride_comp",
    "mediapipe_vs_mediapipe_comp",
}
SAME_CLIP_WITHIN_POOL_SCENARIOS = {
    "wilor_finetune_vs_wilor_finetune",
}
CROSS_MODEL_SAME_CLIP_SCENARIOS = {
    "wilor_finetune_vs_wilor",
    "wilor_finetune_vs_hamba",
    "wilor_finetune_vs_dynhamr",
    "wilor_finetune_vs_stride",
    "wilor_finetune_vs_mediapipe",
    "wilor_vs_hamba",
    "dynhamr_vs_wilor",
    "dynhamr_vs_hamba",
    "dynhamr_vs_stride",
    "dynhamr_vs_mediapipe",
    "wilor_vs_stride",
    "wilor_vs_mediapipe",
    "stride_vs_hamba",
    "stride_vs_mediapipe",
    "mediapipe_vs_hamba",
}
_ANALYSIS_MODULES = None


@dataclass(frozen=True)
class SourceItem:
    family: str
    kind: str
    clip_id: str
    display_id: str
    match_id: str
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


@dataclass(frozen=True)
class SingleSourceItem:
    scenario_id: str
    source: SourceItem
    source_item_id: str


@dataclass(frozen=True)
class FinetuneSelection:
    alias: str
    experiment: str


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
        _ANALYSIS_MODULES = {
            "compare": _load_module("freq_compare", THIS_DIR / "Compare.py"),
            "point_to_point": _load_module("freq_point_to_point", THIS_DIR / "Point to Point.py"),
            "point_to_point_neighbor_sweep": _load_module(
                "freq_point_to_point_neighbor_sweep",
                THIS_DIR / "Point to Point Neighbor Sweep.py",
            ),
            "multi_point_to_point": _load_module("freq_multi_point_to_point", THIS_DIR / "Multi Point to Point.py"),
            "wilor_camera_space": _load_module("freq_wilor_camera_space", THIS_DIR / "WiLoR Camera Space.py"),
            "beta_comparison": _load_module("freq_beta_comparison", THIS_DIR / "beta comparison.py"),
            "beta_multi_point_to_point": _load_module(
                "freq_beta_multi_point_to_point",
                THIS_DIR / "beta multi point to point.py",
            ),
        }
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
    parser = argparse.ArgumentParser(
        description="Run exhaustive frequency analyses, save figures into per-analysis folders, and write aggregate summaries."
    )
    parser.add_argument("--width-px", type=int, default=_int_config("ANALYSIS_IMAGE_WIDTH_PX", 1920))
    parser.add_argument("--height-px", type=int, default=_int_config("ANALYSIS_IMAGE_HEIGHT_PX", 1080))
    parser.add_argument("--dpi", type=int, default=_int_config("ANALYSIS_IMAGE_DPI", 100))
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--dry-run", action="store_true", help="Build and report the job matrix without running analyses.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Run only analysis jobs whose expected output graph file is not already present in --output-dir.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Run only the first N discovered items before per-analysis job fan-out.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes used after job selection.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Repeatable scenario filter for pair and multi-model analyses. Options: " + ", ".join(SCENARIO_OPTIONS),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        default=True,
        help="Add same-clip WiLoR/Hamba/DynHAMR/Stride/MediaPipe multi-model analyses on top of the pairwise runs.",
    )
    return parser.parse_args()


def _normalize_clip_name(name):
    normalized = Path(str(name)).name
    if normalized.lower().endswith(".mp4"):
        normalized = Path(normalized).stem
    return normalized


def _strip_comp_suffixes(name):
    base = str(name)
    changed = True
    while changed:
        changed = False
        for suffix in COMP_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                changed = True
                break
    return base


def _match_id(name):
    return _strip_comp_suffixes(_normalize_clip_name(name))


def _is_comp_clip(name):
    normalized = _normalize_clip_name(name)
    return normalized != _match_id(normalized)


def _read_stride_video_name(clip_dir):
    metadata_path = Path(clip_dir) / "stride_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    video_name = payload.get("video", None)
    if video_name is None:
        return None
    return str(video_name)


def _discover_model_family(outputs_root, family):
    family_root = Path(outputs_root) / family
    items = []
    if not family_root.exists():
        return items

    for mesh_dir in sorted(family_root.glob("*/meshes"), key=lambda path: path.parent.name):
        if not mesh_dir.is_dir():
            continue

        clip_id = mesh_dir.parent.name
        items.append(
            SourceItem(
                family=family,
                kind="model",
                clip_id=clip_id,
                display_id=clip_id,
                match_id=_match_id(clip_id),
                path=str(mesh_dir.resolve()),
                is_comp=_is_comp_clip(clip_id),
            )
        )

    return items


def _discover_experiment_model_family(outputs_root, family, experiments=None):
    family_root = Path(outputs_root) / family
    items = []
    if not family_root.exists():
        return items

    if experiments is None:
        experiment_names = [
            experiment_root.name
            for experiment_root in sorted(family_root.iterdir(), key=lambda path: path.name)
            if experiment_root.is_dir()
        ]
    else:
        experiment_names = [str(experiment) for experiment in experiments]

    for experiment in experiment_names:
        experiment_root = family_root / experiment
        if not experiment_root.is_dir():
            continue

        for mesh_dir in sorted(experiment_root.glob("*/meshes"), key=lambda path: path.parent.name):
            if not mesh_dir.is_dir():
                continue

            clip_id = mesh_dir.parent.name
            items.append(
                SourceItem(
                    family=family,
                    kind="model",
                    clip_id=clip_id,
                    display_id=f"{experiment}/{clip_id}",
                    match_id=_match_id(clip_id),
                    path=str(mesh_dir.resolve()),
                    is_comp=_is_comp_clip(clip_id),
                )
            )

    return items


def _resolve_selected_wilor_finetune_experiments(outputs_root, selected_aliases=None):
    aliases = SELECTED_WILOR_FINETUNE_ALIASES if selected_aliases is None else tuple(selected_aliases)
    family_root = Path(outputs_root) / "wilor_finetune"
    resolved = []

    for raw_alias in _dedupe_preserve_order(aliases):
        alias = str(raw_alias).strip()
        experiment = WILOR_FINETUNE_EXPERIMENT_ALIASES.get(alias)
        if experiment is None:
            valid = ", ".join(sorted(WILOR_FINETUNE_EXPERIMENT_ALIASES))
            raise ValueError(f"Unknown WiLoR finetune alias '{alias}'. Valid aliases: {valid}")

        experiment_root = family_root / experiment
        if not experiment_root.is_dir():
            raise ValueError(
                f"Selected WiLoR finetune alias '{alias}' maps to missing experiment folder: {experiment_root}"
            )

        resolved.append(FinetuneSelection(alias=alias, experiment=experiment))

    return tuple(resolved)


def _discover_mediapipe(outputs_root):
    keypoints_dir = Path(outputs_root) / "mediapipe" / "keypoints"
    items = []
    if not keypoints_dir.exists():
        return items

    for csv_path in sorted(keypoints_dir.glob("*_keypoints.csv"), key=lambda path: path.name):
        if not csv_path.is_file() or csv_path.name.startswith(".~lock."):
            continue

        clip_id = csv_path.name[: -len("_keypoints.csv")]
        items.append(
            SourceItem(
                family="mediapipe",
                kind="mediapipe",
                clip_id=clip_id,
                display_id=clip_id,
                match_id=_match_id(clip_id),
                path=str(csv_path.resolve()),
                is_comp=_is_comp_clip(clip_id),
            )
        )

    return items


def _discover_stride(outputs_root):
    stride_root = Path(outputs_root) / "stride"
    items = []
    if not stride_root.exists():
        return items

    for clip_dir in sorted(stride_root.iterdir(), key=lambda path: path.name):
        if not clip_dir.is_dir() or clip_dir.name.startswith("_"):
            continue
        if not (clip_dir / "refined_sequence.npz").is_file():
            continue

        clip_id = clip_dir.name
        real_clip_id = _read_stride_video_name(clip_dir) or clip_id
        items.append(
            SourceItem(
                family="stride",
                kind="model",
                clip_id=clip_id,
                display_id=clip_id,
                match_id=_match_id(real_clip_id),
                path=str(clip_dir.resolve()),
                is_comp=_is_comp_clip(real_clip_id),
            )
        )

    return items


def _source_display_id(source):
    return source.display_id or source.clip_id


def _slug(text):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "item"


def _build_pair_id(scenario_id, source_a, source_b):
    payload = f"{scenario_id}|{source_a.path}|{source_b.path}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{scenario_id}__{_slug(_source_display_id(source_a))}__{_slug(_source_display_id(source_b))}__{digest}"


def _build_group_id(scenario_id, *sources):
    payload = "|".join([scenario_id] + [source.path for source in sources])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    display_slug = "__".join(_slug(_source_display_id(source)) for source in sources)
    return f"{scenario_id}__{display_slug}__{digest}"


def _build_single_source_id(source):
    payload = f"{source.family}|{source.path}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"single_source__{source.family}__{_slug(_source_display_id(source))}__{digest}"


def _build_source_pools(outputs_root, selected_wilor_finetune_aliases=None):
    finetune_selections = _resolve_selected_wilor_finetune_experiments(outputs_root, selected_wilor_finetune_aliases)
    hamba_all = _discover_model_family(outputs_root, "hamba")
    wilor_all = _discover_model_family(outputs_root, "wilor")
    dynhamr_all = _discover_model_family(outputs_root, "dynhamr")
    stride_all = _discover_stride(outputs_root)
    mediapipe_all = _discover_mediapipe(outputs_root)

    wilor_finetune_all = []
    wilor_finetune_comp = []
    wilor_finetune_discovery = {}

    pools = {
        "hamba_all": hamba_all,
        "hamba_comp": [item for item in hamba_all if item.is_comp],
        "wilor_all": wilor_all,
        "wilor_comp": [item for item in wilor_all if item.is_comp],
        "dynhamr_all": dynhamr_all,
        "dynhamr_comp": [item for item in dynhamr_all if item.is_comp],
        "stride_all": stride_all,
        "stride_comp": [item for item in stride_all if item.is_comp],
        "mediapipe_all": mediapipe_all,
        "mediapipe_comp": [item for item in mediapipe_all if item.is_comp],
    }

    for selection in finetune_selections:
        pool_all_name, pool_comp_name = _wilor_finetune_pool_names(selection.alias)
        selection_items = _discover_experiment_model_family(
            outputs_root,
            "wilor_finetune",
            experiments=(selection.experiment,),
        )
        selection_comp_items = [item for item in selection_items if item.is_comp]
        pools[pool_all_name] = selection_items
        pools[pool_comp_name] = selection_comp_items
        wilor_finetune_all.extend(selection_items)
        wilor_finetune_comp.extend(selection_comp_items)
        wilor_finetune_discovery[selection.alias] = {
            "experiment": selection.experiment,
            "all": len(selection_items),
            "comp": len(selection_comp_items),
        }

    pools["wilor_finetune_all"] = wilor_finetune_all
    pools["wilor_finetune_comp"] = wilor_finetune_comp

    discovery = {
        "hamba": {"all": len(pools["hamba_all"]), "comp": len(pools["hamba_comp"])},
        "wilor": {"all": len(pools["wilor_all"]), "comp": len(pools["wilor_comp"])},
        "wilor_finetune": {
            "all": len(pools["wilor_finetune_all"]),
            "comp": len(pools["wilor_finetune_comp"]),
            "aliases": wilor_finetune_discovery,
        },
        "dynhamr": {"all": len(pools["dynhamr_all"]), "comp": len(pools["dynhamr_comp"])},
        "stride": {"all": len(pools["stride_all"]), "comp": len(pools["stride_comp"])},
        "mediapipe": {"all": len(pools["mediapipe_all"]), "comp": len(pools["mediapipe_comp"])},
    }

    return pools, discovery, finetune_selections


def _build_multi_model_groups(scenario_id, pool_names, pools):
    grouped = []
    for pool_name in pool_names:
        items_by_match = {}
        for item in pools[pool_name]:
            items_by_match.setdefault(item.match_id, []).append(item)
        grouped.append(items_by_match)

    shared = sorted(set.intersection(*(set(items_by_match) for items_by_match in grouped)))
    groups = []
    for match_id in shared:
        source_lists = [items_by_match[match_id] for items_by_match in grouped]
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


def _build_multi_model_scenario_row(scenario_id, pool_names, pools, item_count_total, enabled):
    pool_names = tuple(pool_names)
    pool_counts = [len(pools[pool_name]) for pool_name in pool_names]
    row = {
        "scenario_id": scenario_id,
        "item_kind": "multi_models",
        "item_count_total": int(item_count_total),
        "group_size": len(pool_names),
        "enabled": bool(enabled),
        "source_pools": list(pool_names),
        "source_counts": pool_counts,
    }
    for index, (pool_name, pool_count) in enumerate(zip(pool_names, pool_counts)):
        slot = chr(ord("a") + index)
        row[f"source_{slot}_pool"] = pool_name
        row[f"source_{slot}_count"] = pool_count
    return row


def _build_scenarios_and_items(pools, finetune_selections, requested_scenarios=None, include_all_models=False):
    finetune_all_model_scenarios = _selected_wilor_finetune_all_models_scenarios(
        [selection.alias for selection in finetune_selections]
    )
    allowed = {scenario_id for scenario_id, _, _ in SCENARIOS}
    if include_all_models:
        allowed.update(scenario_id for scenario_id, _ in THREE_MODEL_SCENARIOS)
        allowed.update(scenario_id for scenario_id, _ in finetune_all_model_scenarios)
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
    multi_model_items = []

    for scenario_id, a_pool_name, b_pool_name in SCENARIOS:
        a_items = pools[a_pool_name]
        b_items = pools[b_pool_name]
        if scenario_id in SAME_CLIP_COMP_SCENARIOS:
            total_pairs = 0
            b_by_match = {}
            for item in b_items:
                b_by_match.setdefault(item.match_id, []).append(item)

            for item in a_items:
                if item.is_comp:
                    continue
                total_pairs += len(b_by_match.get(item.match_id, []))
        elif scenario_id in SAME_CLIP_WITHIN_POOL_SCENARIOS:
            total_pairs = 0
            items_by_match = {}
            for item in a_items:
                items_by_match.setdefault(item.match_id, []).append(item)

            for items in items_by_match.values():
                count = len(items)
                if count > 1:
                    total_pairs += count * (count - 1) // 2
        elif scenario_id in CROSS_MODEL_SAME_CLIP_SCENARIOS:
            total_pairs = 0
            b_by_match = {}
            for item in b_items:
                b_by_match.setdefault(item.match_id, []).append(item)

            for item in a_items:
                total_pairs += len(b_by_match.get(item.match_id, []))
        else:
            total_pairs = len(a_items) * len(b_items)

        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "source_a_pool": a_pool_name,
                "source_b_pool": b_pool_name,
                "source_a_count": len(a_items),
                "source_b_count": len(b_items),
                "item_kind": "pairs",
                "item_count_total": total_pairs,
                "enabled": scenario_id in enabled,
            }
        )

        if scenario_id not in enabled:
            continue

        if scenario_id in SAME_CLIP_COMP_SCENARIOS:
            b_by_match = {}
            for item in b_items:
                b_by_match.setdefault(item.match_id, []).append(item)

            for source_a in a_items:
                if source_a.is_comp:
                    continue
                matches = b_by_match.get(source_a.match_id, [])
                for source_b in matches:
                    pairs.append(
                        PairItem(
                            scenario_id=scenario_id,
                            source_a=source_a,
                            source_b=source_b,
                            pair_id=_build_pair_id(scenario_id, source_a, source_b),
                        )
                    )
        elif scenario_id in SAME_CLIP_WITHIN_POOL_SCENARIOS:
            items_by_match = {}
            for item in a_items:
                items_by_match.setdefault(item.match_id, []).append(item)

            for match_id in sorted(items_by_match):
                items = sorted(items_by_match[match_id], key=lambda item: (item.path, item.display_id))
                for idx, source_a in enumerate(items):
                    for source_b in items[idx + 1 :]:
                        if source_a.path == source_b.path:
                            continue
                        pairs.append(
                            PairItem(
                                scenario_id=scenario_id,
                                source_a=source_a,
                                source_b=source_b,
                                pair_id=_build_pair_id(scenario_id, source_a, source_b),
                            )
                        )
        elif scenario_id in CROSS_MODEL_SAME_CLIP_SCENARIOS:
            b_by_match = {}
            for item in b_items:
                b_by_match.setdefault(item.match_id, []).append(item)

            for source_a in a_items:
                matches = b_by_match.get(source_a.match_id, [])
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
            scenario_rows.append(_build_multi_model_scenario_row(scenario_id, pool_names, pools, len(groups), scenario_id in enabled))
            if scenario_id in enabled:
                multi_model_items.extend(groups)

        all_groups = _build_multi_model_groups(
            ALL_MODELS_SCENARIO_ID,
            ("wilor_all", "hamba_all", "dynhamr_all", "stride_all", "mediapipe_all"),
            pools,
        )
        scenario_rows.append(
            _build_multi_model_scenario_row(
                ALL_MODELS_SCENARIO_ID,
                ("wilor_all", "hamba_all", "dynhamr_all", "stride_all", "mediapipe_all"),
                pools,
                len(all_groups),
                ALL_MODELS_SCENARIO_ID in enabled,
            )
        )
        if ALL_MODELS_SCENARIO_ID in enabled:
            multi_model_items.extend(all_groups)

        for scenario_id, pool_names in finetune_all_model_scenarios:
            groups = _build_multi_model_groups(scenario_id, pool_names, pools)
            scenario_rows.append(_build_multi_model_scenario_row(scenario_id, pool_names, pools, len(groups), scenario_id in enabled))
            if scenario_id in enabled:
                multi_model_items.extend(groups)

    return scenario_rows, pairs, multi_model_items


def _build_single_source_items(pools):
    single_source_items = []
    for pool_name in ("wilor_all", "wilor_finetune_all", "dynhamr_all", "stride_all"):
        for source in pools[pool_name]:
            single_source_items.append(
                SingleSourceItem(
                    scenario_id=f"single_source_{source.family}",
                    source=source,
                    source_item_id=_build_single_source_id(source),
                )
            )

    single_source_items.sort(key=lambda item: (item.source.family, item.source.display_id, item.source.path))
    return single_source_items


def _entry_label(source):
    return f"{source.family.upper()}:{_source_display_id(source)}"


def _item_id(item):
    if hasattr(item, "pair_id"):
        return item.pair_id
    if hasattr(item, "group_id"):
        return item.group_id
    return item.source_item_id


def _save_metrics_csv(path, rows):
    fieldnames = [
        "scenario",
        "item_kind",
        "item_id",
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

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_metric_row(job, slot, label, source, kind, result):
    return {
        "scenario": job["scenario_id"],
        "item_kind": job["item_kind"],
        "item_id": job["item_id"],
        "analysis": job["analysis_id"],
        "status": "success",
        "slot": slot,
        "label": label,
        "source": source,
        "kind": kind,
        "dominant_hz": f"{float(result['dominant']):.8f}",
        "rms_amplitude": f"{float(result['rms']):.12f}",
        "num_samples": int(len(result["magnitude"])),
    }


def _summarize_result_entry(slot, label, source, kind, result):
    return {
        "slot": slot,
        "label": label,
        "source": source,
        "kind": kind,
        "dominant_hz": float(result["dominant"]),
        "rms_amplitude": float(result["rms"]),
        "num_samples": int(len(result["magnitude"])),
    }


def _summarize_standard_entries(entries):
    summarized = []
    for entry in entries:
        summarized.append(
            _summarize_result_entry(
                slot=entry.get("slot", ""),
                label=entry["label"],
                source=entry.get("source", ""),
                kind=entry.get("kind", "model"),
                result=entry["result"],
            )
        )
    return summarized


def _make_neighbor_sweep_metric_rows(job, entries):
    rows = []
    for entry in entries:
        for series_row in entry["series"]:
            rows.append(
                {
                    "scenario": job["scenario_id"],
                    "item_kind": job["item_kind"],
                    "item_id": job["item_id"],
                    "analysis": job["analysis_id"],
                    "status": "success",
                    "slot": f"{entry.get('slot', '')}:p{int(series_row['point_count'])}",
                    "label": f"{entry['label']} ({int(series_row['point_count'])} pts)",
                    "source": entry.get("source", ""),
                    "kind": entry.get("kind", "model"),
                    "dominant_hz": f"{float(series_row['dominant']):.8f}",
                    "rms_amplitude": f"{float(series_row['rms']):.12f}",
                    "num_samples": int(series_row["num_samples"]),
                }
            )
    return rows


def _summarize_neighbor_sweep_entries(entries):
    summarized = []
    for entry in entries:
        for series_row in entry["series"]:
            summarized.append(
                {
                    "slot": f"{entry.get('slot', '')}:p{int(series_row['point_count'])}",
                    "label": f"{entry['label']} ({int(series_row['point_count'])} pts)",
                    "source": entry.get("source", ""),
                    "kind": entry.get("kind", "model"),
                    "point_count": int(series_row["point_count"]),
                    "n_neighbors": int(series_row["n_neighbors"]),
                    "dominant_hz": float(series_row["dominant"]),
                    "rms_amplitude": float(series_row["rms"]),
                    "num_samples": int(series_row["num_samples"]),
                }
            )
    return summarized


def _analysis_output_path(output_dir, analysis_id, item_id):
    return Path(output_dir) / ANALYSIS_FOLDERS[analysis_id] / f"{item_id}.svg"


def _pair_payload(pair):
    return {
        "kind": "pair",
        "scenario_id": pair.scenario_id,
        "pair_id": pair.pair_id,
        "source_a": asdict(pair.source_a),
        "source_b": asdict(pair.source_b),
    }


def _multi_model_payload(item):
    return {
        "kind": "multi_model",
        "scenario_id": item.scenario_id,
        "group_id": item.group_id,
        "sources": [asdict(source) for source in item.sources],
    }


def _single_source_payload(item):
    return {
        "kind": "single_source",
        "scenario_id": item.scenario_id,
        "source_item_id": item.source_item_id,
        "source": asdict(item.source),
    }


def _runtime_item_from_payload(item_kind, item):
    if item_kind == "pair":
        return PairItem(
            scenario_id=item["scenario_id"],
            source_a=SourceItem(**item["source_a"]),
            source_b=SourceItem(**item["source_b"]),
            pair_id=item["pair_id"],
        )
    if item_kind == "multi_model":
        return MultiModelsItem(
            scenario_id=item["scenario_id"],
            sources=tuple(SourceItem(**source) for source in item["sources"]),
            group_id=item["group_id"],
        )
    return SingleSourceItem(
        scenario_id=item["scenario_id"],
        source=SourceItem(**item["source"]),
        source_item_id=item["source_item_id"],
    )


def _analysis_ids_for_item(item_kind, runtime_item):
    if item_kind in {"pair", "multi_model"}:
        if item_kind == "pair":
            sources = [runtime_item.source_a, runtime_item.source_b]
        else:
            sources = list(runtime_item.sources)

        analysis_ids = []
        for analysis_id in PAIRWISE_ANALYSES:
            if analysis_id in MODEL_ONLY_PAIRWISE_ANALYSES and any(source.kind != "model" for source in sources):
                continue
            analysis_ids.append(analysis_id)
        return analysis_ids
    family = runtime_item.source.family
    analysis_ids = []
    for analysis_id in SINGLE_SOURCE_ANALYSES:
        if family in SINGLE_SOURCE_ANALYSIS_FAMILIES[analysis_id]:
            analysis_ids.append(analysis_id)
    return analysis_ids


def _build_base_items(pairs, multi_model_items, single_source_items):
    base_items = []
    for pair in pairs:
        base_items.append(("pair", pair))
    for item in multi_model_items:
        base_items.append(("multi_model", item))
    for item in single_source_items:
        base_items.append(("single_source", item))
    return base_items


def _build_jobs(output_dir, base_items, figsize_inches, dpi):
    jobs = []
    sequence = 0
    for item_kind, item in base_items:
        runtime_item = item
        analysis_ids = _analysis_ids_for_item(item_kind, runtime_item)
        payload = {
            "pair": _pair_payload,
            "multi_model": _multi_model_payload,
            "single_source": _single_source_payload,
        }[item_kind](runtime_item)
        item_id = _item_id(runtime_item)

        for analysis_id in analysis_ids:
            sequence += 1
            jobs.append(
                {
                    "analysis_id": analysis_id,
                    "item_kind": item_kind,
                    "scenario_id": runtime_item.scenario_id,
                    "sequence": sequence,
                    "item_id": item_id,
                    "item": payload,
                    "output_path": str(_analysis_output_path(output_dir, analysis_id, item_id)),
                    "figsize_inches": tuple(figsize_inches),
                    "dpi": int(dpi),
                }
            )

    return jobs


def _job_key(job):
    return f"{job['analysis_id']}::{job['item_id']}"


def _job_inputs(job):
    item = job["item"]
    if job["item_kind"] == "pair":
        return {
            "source_a": item["source_a"],
            "source_b": item["source_b"],
        }
    if job["item_kind"] == "multi_model":
        return {
            "sources": item["sources"],
        }
    return {
        "source": item["source"],
    }


def _planned_job_summary(job, status):
    return {
        "analysis_id": job["analysis_id"],
        "item_kind": job["item_kind"],
        "scenario": job["scenario_id"],
        "item_id": job["item_id"],
        "status": status,
        "output_path": job["output_path"],
        "inputs": _job_inputs(job),
    }


def _save_figure(fig, output_path, dpi):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _run_compare_job(job, module, runtime_item, output_path, figsize_inches, dpi):
    if job["item_kind"] == "pair":
        overrides = {
            "source_a": runtime_item.source_a.path,
            "source_b": runtime_item.source_b.path,
            "label_a": _entry_label(runtime_item.source_a),
            "label_b": _entry_label(runtime_item.source_b),
        }
    else:
        sources = list(runtime_item.sources)
        overrides = {
            "all_models": True,
            "sources": [source.path for source in sources],
            "labels": [_entry_label(source) for source in sources],
        }

    analysis_data = module.run_compare_analysis(overrides)
    fig = module.build_compare_figure(analysis_data, figsize_inches=figsize_inches, dpi=dpi)
    _save_figure(fig, output_path, dpi)

    metric_rows = []
    for entry in analysis_data["entries"]:
        metric_rows.append(
            _make_metric_row(
                job,
                slot=entry.get("slot", ""),
                label=entry["label"],
                source=entry.get("source", ""),
                kind=entry.get("kind", "model"),
                result=entry["result"],
            )
        )

    summary = _planned_job_summary(job, status="success")
    summary["entries"] = _summarize_standard_entries(analysis_data["entries"])
    return summary, metric_rows


def _run_point_to_point_job(job, module, runtime_item, output_path, figsize_inches, dpi):
    if job["item_kind"] == "pair":
        overrides = {
            "source_a": runtime_item.source_a.path,
            "source_b": runtime_item.source_b.path,
            "label_a": _entry_label(runtime_item.source_a),
            "label_b": _entry_label(runtime_item.source_b),
        }
    else:
        sources = list(runtime_item.sources)
        overrides = {
            "all_models": True,
            "sources": [source.path for source in sources],
            "labels": [_entry_label(source) for source in sources],
        }

    analysis_data = module.run_point_to_point_analysis(overrides)
    fig = module.build_point_to_point_figure(analysis_data, figsize_inches=figsize_inches, dpi=dpi)
    _save_figure(fig, output_path, dpi)

    metric_rows = []
    for entry in analysis_data["entries"]:
        metric_rows.append(
            _make_metric_row(
                job,
                slot=entry.get("slot", ""),
                label=entry["label"],
                source=entry.get("source", ""),
                kind=entry.get("kind", "model"),
                result=entry["result"],
            )
        )

    summary = _planned_job_summary(job, status="success")
    summary["entries"] = _summarize_standard_entries(analysis_data["entries"])
    return summary, metric_rows


def _run_point_to_point_neighbor_sweep_job(job, module, runtime_item, output_path, figsize_inches, dpi):
    if job["item_kind"] == "pair":
        overrides = {
            "source_a": runtime_item.source_a.path,
            "source_b": runtime_item.source_b.path,
            "label_a": _entry_label(runtime_item.source_a),
            "label_b": _entry_label(runtime_item.source_b),
        }
    else:
        sources = list(runtime_item.sources)
        overrides = {
            "all_models": True,
            "sources": [source.path for source in sources],
            "labels": [_entry_label(source) for source in sources],
        }

    analysis_data = module.run_point_to_point_neighbor_sweep_analysis(overrides)
    fig = module.build_point_to_point_neighbor_sweep_figure(analysis_data, figsize_inches=figsize_inches, dpi=dpi)
    _save_figure(fig, output_path, dpi)

    summary = _planned_job_summary(job, status="success")
    summary["entries"] = _summarize_neighbor_sweep_entries(analysis_data["entries"])
    return summary, _make_neighbor_sweep_metric_rows(job, analysis_data["entries"])


def _run_multi_point_job(job, module, runtime_item, output_path, figsize_inches, dpi):
    if job["item_kind"] == "pair":
        sources = [runtime_item.source_a, runtime_item.source_b]
    else:
        sources = list(runtime_item.sources)

    overrides = {
        "sources": [{"family": source.family, "path": source.path} for source in sources],
    }
    analysis_data = module.run_multi_point_analysis(overrides)
    fig = module.build_multi_point_figure(analysis_data, figsize_inches=figsize_inches, dpi=dpi)
    _save_figure(fig, output_path, dpi)

    metric_rows = []
    for entry in analysis_data["entries"]:
        metric_rows.append(
            _make_metric_row(
                job,
                slot=entry.get("slot", ""),
                label=entry["label"],
                source=entry.get("source", ""),
                kind=entry.get("kind", "model"),
                result=entry["result"],
            )
        )

    summary = _planned_job_summary(job, status="success")
    summary["entries"] = _summarize_standard_entries(analysis_data["entries"])
    return summary, metric_rows


def _run_camera_space_job(job, module, runtime_item, output_path, figsize_inches, dpi):
    analysis_data = module.run_camera_space_frequency_analysis(
        source_path=runtime_item.source.path,
        hand_idx=int(CONFIG.HAND_IDX),
        wrist_joint_idx=int(CONFIG.WRIST_JOINT_IDX),
        n_neighbors=int(CONFIG.N_NEIGHBORS),
    )
    fig = module.build_camera_space_figure(
        analysis_data["entries"],
        source_label=_entry_label(runtime_item.source),
        figsize_inches=figsize_inches,
        dpi=dpi,
    )
    _save_figure(fig, output_path, dpi)

    metric_rows = []
    summary_entries = []
    for entry in analysis_data["entries"]:
        result = entry["result"]
        metric_rows.append(
            _make_metric_row(
                job,
                slot=entry["pair_label"],
                label=entry["pair_label"],
                source=analysis_data["source_path"],
                kind="model",
                result=result,
            )
        )
        summary_entries.append(
            _summarize_result_entry(
                slot=entry["pair_label"],
                label=entry["pair_label"],
                source=analysis_data["source_path"],
                kind="model",
                result=result,
            )
        )

    summary = _planned_job_summary(job, status="success")
    summary["entries"] = summary_entries
    return summary, metric_rows


def _run_beta_comparison_job(job, module, runtime_item, output_path, figsize_inches, dpi):
    analysis_data = module.run_beta_comparison_analysis(
        {
            "source_path": runtime_item.source.path,
            "mano_model_path": str(CONFIG.MANO_RIGHT_PATH),
            "hand_idx": int(CONFIG.HAND_IDX),
            "n_neighbors": int(CONFIG.N_NEIGHBORS),
            "vertex_a": int(CONFIG.MODEL_SPECIFIC_VERTEX_A),
            "vertex_b": int(CONFIG.MODEL_SPECIFIC_VERTEX_B),
        }
    )
    fig = module.build_beta_comparison_figure(analysis_data, figsize_inches=figsize_inches, dpi=dpi)
    _save_figure(fig, output_path, dpi)

    metric_rows = []
    summary_entries = []
    source_path = analysis_data.get("source_path", analysis_data.get("mesh_dir", runtime_item.source.path))
    for entry in analysis_data["entries"]:
        result = entry["result"]
        metric_rows.append(
            _make_metric_row(
                job,
                slot=entry["label"],
                label=entry["label"],
                source=source_path,
                kind="model",
                result=result,
            )
        )
        summary_entries.append(
            _summarize_result_entry(
                slot=entry["label"],
                label=entry["label"],
                source=source_path,
                kind="model",
                result=result,
            )
        )

    summary = _planned_job_summary(job, status="success")
    summary["entries"] = summary_entries
    return summary, metric_rows


def _run_beta_multi_point_job(job, module, runtime_item, output_path, figsize_inches, dpi):
    analysis_data = module.run_beta_multi_point_analysis(
        {
            "source_path": runtime_item.source.path,
            "mano_model_path": str(CONFIG.MANO_RIGHT_PATH),
            "hand_idx": int(CONFIG.HAND_IDX),
            "wrist_joint_idx": int(CONFIG.WRIST_JOINT_IDX),
            "n_neighbors": int(CONFIG.N_NEIGHBORS),
        }
    )
    fig = module.build_beta_multi_point_figure(analysis_data, figsize_inches=figsize_inches, dpi=dpi)
    _save_figure(fig, output_path, dpi)

    metric_rows = []
    summary_entries = []
    source_path = analysis_data.get("source_path", analysis_data.get("mesh_dir", runtime_item.source.path))
    for entry in analysis_data["entries"]:
        result = entry["result"]
        metric_rows.append(
            _make_metric_row(
                job,
                slot=entry.get("slot", entry["pair_label"]),
                label=entry["label"],
                source=source_path,
                kind="model",
                result=result,
            )
        )
        summary_entries.append(
            _summarize_result_entry(
                slot=entry.get("slot", entry["pair_label"]),
                label=entry["label"],
                source=source_path,
                kind="model",
                result=result,
            )
        )

    summary = _planned_job_summary(job, status="success")
    summary["entries"] = summary_entries
    return summary, metric_rows


def _run_worker_job(job):
    try:
        modules = _get_analysis_modules()
        runtime_item = _runtime_item_from_payload(job["item_kind"], job["item"])
        output_path = Path(job["output_path"])
        figsize_inches = tuple(job["figsize_inches"])
        dpi = int(job["dpi"])

        if job["analysis_id"] == "compare":
            summary, metric_rows = _run_compare_job(job, modules["compare"], runtime_item, output_path, figsize_inches, dpi)
        elif job["analysis_id"] == "point_to_point":
            summary, metric_rows = _run_point_to_point_job(job, modules["point_to_point"], runtime_item, output_path, figsize_inches, dpi)
        elif job["analysis_id"] == "point_to_point_neighbor_sweep":
            summary, metric_rows = _run_point_to_point_neighbor_sweep_job(
                job,
                modules["point_to_point_neighbor_sweep"],
                runtime_item,
                output_path,
                figsize_inches,
                dpi,
            )
        elif job["analysis_id"] == "multi_point_to_point":
            summary, metric_rows = _run_multi_point_job(job, modules["multi_point_to_point"], runtime_item, output_path, figsize_inches, dpi)
        elif job["analysis_id"] == "wilor_camera_space":
            summary, metric_rows = _run_camera_space_job(job, modules["wilor_camera_space"], runtime_item, output_path, figsize_inches, dpi)
        elif job["analysis_id"] == "beta_comparison":
            summary, metric_rows = _run_beta_comparison_job(job, modules["beta_comparison"], runtime_item, output_path, figsize_inches, dpi)
        elif job["analysis_id"] == "beta_multi_point_to_point":
            summary, metric_rows = _run_beta_multi_point_job(
                job,
                modules["beta_multi_point_to_point"],
                runtime_item,
                output_path,
                figsize_inches,
                dpi,
            )
        else:
            raise ValueError(f"Unsupported analysis_id: {job['analysis_id']}")

        return {
            "analysis_id": job["analysis_id"],
            "item_kind": job["item_kind"],
            "sequence": int(job["sequence"]),
            "item_id": job["item_id"],
            "summary": summary,
            "metric_rows": metric_rows,
            "analysis_success": 1,
            "analysis_failed": 0,
        }
    except Exception as exc:  # noqa: BLE001
        summary = _planned_job_summary(job, status="failed")
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc(limit=10)
        return {
            "analysis_id": job["analysis_id"],
            "item_kind": job["item_kind"],
            "sequence": int(job["sequence"]),
            "item_id": job["item_id"],
            "summary": summary,
            "metric_rows": [],
            "analysis_success": 0,
            "analysis_failed": 1,
        }


def _analysis_counts_template():
    counts = {}
    for analysis_id in ANALYSIS_IDS:
        counts[analysis_id] = {
            "jobs_discovered": 0,
            "jobs_selected": 0,
            "jobs_skipped_existing": 0,
            "success": 0,
            "failed": 0,
        }
    return counts


def _count_jobs_by_analysis(jobs, key_name):
    counts = {analysis_id: 0 for analysis_id in ANALYSIS_IDS}
    for job in jobs:
        counts[job[key_name]] += 1
    return counts


def _selected_item_sets(selected_jobs):
    item_sets = {
        "pair": set(),
        "multi_model": set(),
        "single_source": set(),
    }
    scenario_item_counts = {}
    seen_scenario_items = set()
    for job in selected_jobs:
        item_sets[job["item_kind"]].add(job["item_id"])
        if job["item_kind"] == "single_source":
            continue
        key = (job["scenario_id"], job["item_id"])
        if key in seen_scenario_items:
            continue
        seen_scenario_items.add(key)
        scenario_item_counts[job["scenario_id"]] = scenario_item_counts.get(job["scenario_id"], 0) + 1
    return item_sets, scenario_item_counts


def _print_discovery(discovery, scenario_rows, selected_item_sets, selected_jobs, skipped_existing_jobs):
    print("Discovery counts:")
    for family in ("hamba", "wilor", "wilor_finetune", "dynhamr", "stride", "mediapipe"):
        counts = discovery[family]
        print(f"  {family}: all={counts['all']}, comp={counts['comp']}")
        if family == "wilor_finetune":
            alias_counts = counts.get("aliases", {})
            if alias_counts:
                for alias, alias_data in alias_counts.items():
                    print(
                        f"    {alias} -> {alias_data['experiment']}: "
                        f"all={alias_data['all']}, comp={alias_data['comp']}"
                    )
            else:
                print("    selected aliases: none")

    print("Scenario matrix:")
    for row in scenario_rows:
        enabled = "enabled" if row["enabled"] else "disabled"
        if row["item_kind"] == "multi_models":
            group_size = int(row.get("group_size", 0))
            pools_text = " x ".join(
                f"{pool_name}({pool_count})"
                for pool_name, pool_count in zip(row.get("source_pools", []), row.get("source_counts", []))
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

    print(f"Selected pair items with pending jobs: {len(selected_item_sets['pair'])}")
    print(f"Selected multi-model items with pending jobs: {len(selected_item_sets['multi_model'])}")
    print(f"Selected single-source items with pending jobs: {len(selected_item_sets['single_source'])}")

    selected_job_counts = {analysis_id: 0 for analysis_id in ANALYSIS_IDS}
    for job in selected_jobs:
        selected_job_counts[job["analysis_id"]] += 1

    skipped_job_counts = {analysis_id: 0 for analysis_id in ANALYSIS_IDS}
    for job in skipped_existing_jobs:
        skipped_job_counts[job["analysis_id"]] += 1

    print("Selected jobs by analysis:")
    for analysis_id in ANALYSIS_IDS:
        print(
            f"  {analysis_id}: "
            f"selected={selected_job_counts[analysis_id]}, "
            f"skipped_existing={skipped_job_counts[analysis_id]}"
        )


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

    pools, discovery, finetune_selections = _build_source_pools(CONFIG.OUTPUTS_ROOT)
    scenario_rows, all_pairs, all_multi_model_items = _build_scenarios_and_items(
        pools,
        finetune_selections,
        args.scenario,
        include_all_models=args.all_models,
    )
    all_single_source_items = _build_single_source_items(pools)

    base_items = _build_base_items(all_pairs, all_multi_model_items, all_single_source_items)
    selected_base_items = list(base_items)
    if args.max_pairs is not None:
        selected_base_items = selected_base_items[: args.max_pairs]

    planned_jobs = _build_jobs(output_dir, selected_base_items, figsize_inches, args.dpi)
    selected_jobs = []
    skipped_existing_jobs = []
    summary_jobs = []
    summary_indexes = {}
    analysis_totals = _analysis_counts_template()

    for job in planned_jobs:
        analysis_totals[job["analysis_id"]]["jobs_discovered"] += 1
        output_path = Path(job["output_path"])
        if args.only_missing and output_path.exists():
            skipped_existing_jobs.append(job)
            analysis_totals[job["analysis_id"]]["jobs_skipped_existing"] += 1
            summary_entry = _planned_job_summary(job, status="skipped_existing")
        else:
            selected_jobs.append(job)
            analysis_totals[job["analysis_id"]]["jobs_selected"] += 1
            summary_entry = _planned_job_summary(job, status="pending")

        summary_indexes[_job_key(job)] = len(summary_jobs)
        summary_jobs.append(summary_entry)

    selected_item_sets, scenario_item_counts = _selected_item_sets(selected_jobs)
    for row in scenario_rows:
        row["item_count_selected"] = int(scenario_item_counts.get(row["scenario_id"], 0))

    _print_discovery(discovery, scenario_rows, selected_item_sets, selected_jobs, skipped_existing_jobs)

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
            "selected_wilor_finetune_aliases": [selection.alias for selection in finetune_selections],
            "selected_wilor_finetune_experiments": [
                {
                    "alias": selection.alias,
                    "experiment": selection.experiment,
                }
                for selection in finetune_selections
            ],
        },
        "discovery": discovery,
        "scenarios": scenario_rows,
        "jobs": summary_jobs,
        "totals": {
            "pair_items_discovered": int(len(all_pairs)),
            "pair_items_selected": int(len(selected_item_sets["pair"])),
            "multi_model_items_discovered": int(len(all_multi_model_items)),
            "multi_model_items_selected": int(len(selected_item_sets["multi_model"])),
            "single_source_items_discovered": int(len(all_single_source_items)),
            "single_source_items_selected": int(len(selected_item_sets["single_source"])),
            "base_items_discovered": int(len(base_items)),
            "base_items_after_cap": int(len(selected_base_items)),
            "jobs_discovered": int(len(planned_jobs)),
            "jobs_selected": int(len(selected_jobs)),
            "jobs_skipped_existing": int(len(skipped_existing_jobs)),
            "analysis_success": 0,
            "analysis_failed": 0,
        },
        "analysis_totals": analysis_totals,
    }

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"

    if args.dry_run:
        _save_metrics_csv(metrics_path, rows=[])
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print(f"Saved metrics CSV: {metrics_path}")
        print(f"Saved summary JSON: {summary_path}")
        print("Dry run complete.")
        return 0

    metric_rows = []
    if selected_jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(_run_worker_job, job): job for job in selected_jobs}
            completed = 0
            total_jobs = len(selected_jobs)
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                completed += 1
                print(f"[{completed}/{total_jobs}] {job['analysis_id']} :: {job['item_id']}")

                result = future.result()
                metric_rows.extend(result["metric_rows"])
                summary["totals"]["analysis_success"] += int(result["analysis_success"])
                summary["totals"]["analysis_failed"] += int(result["analysis_failed"])
                summary["analysis_totals"][result["analysis_id"]]["success"] += int(result["analysis_success"])
                summary["analysis_totals"][result["analysis_id"]]["failed"] += int(result["analysis_failed"])

                summary_index = summary_indexes[_job_key(job)]
                summary["jobs"][summary_index] = result["summary"]

    metric_rows.sort(
        key=lambda row: (
            row["scenario"],
            row["item_kind"],
            row["item_id"],
            row["analysis"],
            row["slot"],
            row["label"],
        )
    )
    _save_metrics_csv(metrics_path, metric_rows)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved metrics CSV: {metrics_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(
        "Totals: "
        f"success={summary['totals']['analysis_success']}, "
        f"failed={summary['totals']['analysis_failed']}, "
        f"jobs_selected={summary['totals']['jobs_selected']}, "
        f"jobs_skipped_existing={summary['totals']['jobs_skipped_existing']}"
    )

    return 1 if summary["totals"]["analysis_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

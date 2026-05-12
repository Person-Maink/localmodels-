from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from experiment_config import DEFAULT_EXPERIMENT_CONFIG, resolve_experiment_config
from lora_config import normalize_lora_config


@dataclass(frozen=True)
class ConfigBinding:
    arg_name: str
    config_path: tuple[str, ...]
    to_arg: Callable[[Any], Any] | None = None


REQUIRED_BINDINGS = (
    ConfigBinding("train_scope", ("train_scope",)),
    ConfigBinding("validation_split", ("validation_split",)),
    ConfigBinding("sample_limit", ("sample_limit",)),
    ConfigBinding("detection_conf", ("detection_conf",)),
    ConfigBinding("rescale_factor", ("rescale_factor",)),
    ConfigBinding("batch_size", ("batch_size",)),
    ConfigBinding("num_workers", ("num_workers",)),
    ConfigBinding("max_steps", ("max_steps",)),
    ConfigBinding("log_every", ("log_every",)),
    ConfigBinding("save_every", ("save_every",)),
    ConfigBinding("seed", ("seed",)),
    ConfigBinding("lr", ("optimizer", "lr")),
    ConfigBinding("weight_decay", ("optimizer", "weight_decay")),
)

OPTIONAL_BINDINGS = (
    ConfigBinding("temporal_window_size", ("temporal", "window_size")),
    ConfigBinding("temporal_window_stride", ("temporal", "window_stride")),
    ConfigBinding("temporal_max_frame_gap", ("temporal", "max_frame_gap")),
    ConfigBinding("temporal_reduction", ("temporal", "reduction")),
    ConfigBinding("temporal_scorer_hidden_dim", ("temporal", "scorer_hidden_dim")),
    ConfigBinding("temporal_scorer_layers", ("temporal", "scorer_layers")),
    ConfigBinding("temporal_scorer_dropout", ("temporal", "scorer_dropout")),
    ConfigBinding("vipe_camera_enabled", ("losses", "vipe_camera", "enabled")),
    ConfigBinding("vipe_camera_weight", ("losses", "vipe_camera", "weight")),
    ConfigBinding("temporal_camera_enabled", ("losses", "temporal_camera", "enabled")),
    ConfigBinding("temporal_camera_formulation", ("losses", "temporal_camera", "formulation")),
    ConfigBinding("temporal_camera_weight", ("losses", "temporal_camera", "weight")),
    ConfigBinding("temporal_camera_scorer_weight", ("losses", "temporal_camera", "scorer_weight")),
    ConfigBinding(
        "temporal_bbox_projected_enabled",
        ("losses", "temporal_bbox_projected", "enabled"),
    ),
    ConfigBinding(
        "temporal_bbox_projected_formulation",
        ("losses", "temporal_bbox_projected", "formulation"),
    ),
    ConfigBinding(
        "temporal_bbox_projected_weight",
        ("losses", "temporal_bbox_projected", "weight"),
    ),
    ConfigBinding(
        "temporal_bbox_projected_scorer_weight",
        ("losses", "temporal_bbox_projected", "scorer_weight"),
    ),
    ConfigBinding("lora_enabled", ("lora", "enabled")),
    ConfigBinding("lora_rank", ("lora", "rank")),
    ConfigBinding("lora_alpha", ("lora", "alpha")),
    ConfigBinding("lora_dropout", ("lora", "dropout")),
    ConfigBinding("lora_block_start", ("lora", "block_start")),
    ConfigBinding("lora_block_end", ("lora", "block_end")),
    ConfigBinding(
        "lora_target_modules",
        ("lora", "target_modules"),
        to_arg=lambda value: ",".join(value),
    ),
)


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        current = current[key]
    return current


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = copy.deepcopy(value)


def _set_arg_from_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    dest: str,
    value: Any,
) -> None:
    current = getattr(args, dest)
    default = parser.get_default(dest)
    if isinstance(current, list):
        if not current or current == default:
            setattr(args, dest, copy.deepcopy(value))
        return
    if current is None or current == default:
        setattr(args, dest, copy.deepcopy(value))


def apply_experiment_defaults(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    resolved_experiment: dict[str, Any],
) -> None:
    for binding in (*REQUIRED_BINDINGS, *OPTIONAL_BINDINGS):
        value = _get_nested(resolved_experiment, binding.config_path)
        if binding.to_arg is not None:
            value = binding.to_arg(value)
        _set_arg_from_config(args, parser, binding.arg_name, value)

    if not args.videos and args.all_videos == parser.get_default("all_videos"):
        if resolved_experiment["all_videos"]:
            args.all_videos = True
        elif resolved_experiment["videos"]:
            args.videos = list(resolved_experiment["videos"])


def build_resolved_experiment_snapshot(
    args: argparse.Namespace,
    base_experiment: dict[str, Any] | None,
) -> dict[str, Any]:
    snapshot = copy.deepcopy(base_experiment or DEFAULT_EXPERIMENT_CONFIG)
    snapshot["name"] = args.experiment_name or snapshot.get("name") or "manual_distill_run"
    snapshot["source_loss_config"] = (
        str(Path(args.loss_config).expanduser().resolve()) if args.loss_config else None
    )
    snapshot["run_name_suffix"] = str(snapshot.get("run_name_suffix", "") or "")
    snapshot["train_mode"] = str(snapshot.get("train_mode", "distill") or "distill")
    snapshot["videos"] = [] if args.all_videos else sorted(set(args.videos))
    snapshot["all_videos"] = bool(args.all_videos)

    for binding in REQUIRED_BINDINGS:
        _set_nested(snapshot, binding.config_path, getattr(args, binding.arg_name))

    for binding in OPTIONAL_BINDINGS:
        value = getattr(args, binding.arg_name)
        if value is None:
            continue
        _set_nested(snapshot, binding.config_path, value)

    if args.vipe_camera_weight is not None:
        snapshot["losses"]["vipe_camera"]["weight"] = float(args.vipe_camera_weight)
    else:
        snapshot["losses"]["vipe_camera"]["weight"] = float(args.camera_loss_weight)

    snapshot["lora"] = normalize_lora_config(snapshot["lora"])
    return snapshot


def resolve_runtime_experiment(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    loaded_experiment = None
    if args.loss_config:
        if log_fn is not None:
            log_fn(
                f"Resolving experiment config from {args.loss_config}"
                + (
                    f" (experiment={args.experiment_name})"
                    if args.experiment_name
                    else ""
                )
            )
        loaded_experiment = resolve_experiment_config(args.loss_config, args.experiment_name)
        apply_experiment_defaults(args, parser, loaded_experiment)
        args.experiment_name = args.experiment_name or loaded_experiment["name"]
        if log_fn is not None:
            log_fn(f"Resolved experiment '{args.experiment_name}'.")

    return loaded_experiment, build_resolved_experiment_snapshot(args, loaded_experiment)

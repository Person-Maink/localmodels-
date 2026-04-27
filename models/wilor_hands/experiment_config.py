import argparse
import base64
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from lora_config import DEFAULT_LORA_CONFIG, normalize_lora_config


DEFAULT_TEMPORAL_CONFIG = {
    "window_size": 8,
    "window_stride": 4,
    "max_frame_gap": 1,
    "reduction": "smooth_l1",
    "scorer_hidden_dim": 64,
    "scorer_layers": 2,
    "scorer_dropout": 0.0,
}

DEFAULT_LOSS_CONFIG = {
    "vipe_camera": {
        "enabled": True,
        "weight": 0.01,
        "scorer_weight": 0.0,
    },
    "temporal_camera": {
        "enabled": False,
        "formulation": "static",
        "weight": 0.0,
        "scorer_weight": 0.0,
    },
    "temporal_bbox_projected": {
        "enabled": False,
        "formulation": "static",
        "weight": 0.0,
        "scorer_weight": 0.0,
    },
    "temporal_vipe_camera": {
        "enabled": False,
        "formulation": "learnable",
        "weight": 0.0,
        "scorer_weight": 0.0,
        "smoothness_weight": 0.0,
        "anchor_weight": 0.0,
    },
}

DEFAULT_EXPERIMENT_CONFIG = {
    "name": None,
    "run_name_suffix": "",
    "train_mode": "distill",
    "videos": [],
    "all_videos": False,
    "train_scope": "refine_net",
    "validation_split": 0.15,
    "sample_limit": 0,
    "detection_conf": 0.3,
    "rescale_factor": 2.0,
    "batch_size": 8,
    "num_workers": 4,
    "max_steps": 10000,
    "log_every": 25,
    "save_every": 250,
    "seed": 42,
    "optimizer": {
        "lr": 1e-5,
        "weight_decay": 1e-4,
    },
    "temporal": DEFAULT_TEMPORAL_CONFIG,
    "lora": DEFAULT_LORA_CONFIG,
    "losses": DEFAULT_LOSS_CONFIG,
}

ENV_KEY_MAP = {
    "train_mode": "TRAIN_MODE",
    "train_scope": "TRAIN_SCOPE",
    "validation_split": "VALIDATION_SPLIT",
    "sample_limit": "SAMPLE_LIMIT",
    "detection_conf": "DETECTION_CONF",
    "rescale_factor": "RESCALE_FACTOR",
    "batch_size": "BATCH_SIZE",
    "num_workers": "NUM_WORKERS",
    "max_steps": "MAX_STEPS",
    "log_every": "LOG_EVERY",
    "save_every": "SAVE_EVERY",
    "seed": "SEED",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _normalize_experiment(resolved: dict[str, Any]) -> dict[str, Any]:
    normalized = _deep_merge(DEFAULT_EXPERIMENT_CONFIG, resolved)
    normalized["name"] = resolved.get("name", normalized.get("name"))
    normalized["run_name_suffix"] = str(normalized.get("run_name_suffix") or "")
    normalized["train_mode"] = str(normalized.get("train_mode") or "distill")
    if normalized["train_mode"] not in {"distill", "test"}:
        raise ValueError(
            f"Unsupported train_mode '{normalized['train_mode']}'. "
            "WiLoR experiment configs now support only 'distill' and 'test'."
        )
    normalized["videos"] = _ensure_list(normalized.get("videos"))
    normalized["all_videos"] = _to_bool(normalized.get("all_videos", False))

    if normalized["all_videos"] and normalized["videos"]:
        raise ValueError("Experiment config cannot set both 'all_videos' and 'videos'.")

    normalized["validation_split"] = float(normalized["validation_split"])
    normalized["sample_limit"] = int(normalized["sample_limit"])
    normalized["detection_conf"] = float(normalized["detection_conf"])
    normalized["rescale_factor"] = float(normalized["rescale_factor"])
    normalized["batch_size"] = int(normalized["batch_size"])
    normalized["num_workers"] = int(normalized["num_workers"])
    normalized["max_steps"] = int(normalized["max_steps"])
    normalized["log_every"] = int(normalized["log_every"])
    normalized["save_every"] = int(normalized["save_every"])
    normalized["seed"] = int(normalized["seed"])
    normalized["optimizer"]["lr"] = float(normalized["optimizer"]["lr"])
    normalized["optimizer"]["weight_decay"] = float(
        normalized["optimizer"]["weight_decay"]
    )

    normalized["temporal"] = _deep_merge(
        DEFAULT_TEMPORAL_CONFIG,
        normalized.get("temporal", {}),
    )
    normalized["temporal"]["window_size"] = int(normalized["temporal"]["window_size"])
    normalized["temporal"]["window_stride"] = int(normalized["temporal"]["window_stride"])
    normalized["temporal"]["max_frame_gap"] = int(normalized["temporal"]["max_frame_gap"])
    normalized["temporal"]["reduction"] = str(normalized["temporal"]["reduction"])
    normalized["temporal"]["scorer_hidden_dim"] = int(
        normalized["temporal"]["scorer_hidden_dim"]
    )
    normalized["temporal"]["scorer_layers"] = int(normalized["temporal"]["scorer_layers"])
    normalized["temporal"]["scorer_dropout"] = float(
        normalized["temporal"]["scorer_dropout"]
    )

    normalized["lora"] = normalize_lora_config(normalized.get("lora", {}))

    normalized["losses"] = _deep_merge(
        DEFAULT_LOSS_CONFIG,
        normalized.get("losses", {}),
    )
    allowed_loss_families = set(DEFAULT_LOSS_CONFIG)
    unknown_loss_families = sorted(set(normalized["losses"]) - allowed_loss_families)
    if unknown_loss_families:
        allowed_list = ", ".join(sorted(allowed_loss_families))
        unknown_list = ", ".join(unknown_loss_families)
        raise ValueError(
            "Unknown loss family(s) in experiment config: "
            f"{unknown_list}. Allowed families: {allowed_list}"
        )
    for family_name, family_cfg in normalized["losses"].items():
        family_cfg["enabled"] = _to_bool(family_cfg["enabled"])
        family_cfg["weight"] = float(family_cfg["weight"])
        family_cfg["scorer_weight"] = float(family_cfg.get("scorer_weight", 0.0))
        if family_name != "vipe_camera":
            family_cfg["formulation"] = str(family_cfg.get("formulation", "static"))
            allowed_formulations = {"static", "learnable"}
            if family_name == "temporal_vipe_camera":
                allowed_formulations = {"learnable"}
                family_cfg["smoothness_weight"] = float(
                    family_cfg.get("smoothness_weight", 0.0)
                )
                family_cfg["anchor_weight"] = float(
                    family_cfg.get("anchor_weight", 0.0)
                )
            if family_cfg["formulation"] not in allowed_formulations:
                allowed_display = " or ".join(sorted(allowed_formulations))
                raise ValueError(
                    f"Unsupported formulation '{family_cfg['formulation']}' for loss family "
                    f"'{family_name}'. Use {allowed_display}."
                )

    return normalized


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")
    return payload


def _normalize_experiment_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    experiments = payload.get("experiments", [])
    if isinstance(experiments, dict):
        normalized = {}
        for name, cfg in experiments.items():
            entry = copy.deepcopy(cfg or {})
            entry.setdefault("name", name)
            normalized[name] = entry
        return normalized

    normalized = {}
    for entry in experiments:
        if not isinstance(entry, dict):
            raise ValueError("Each experiment entry must be a mapping.")
        name = entry.get("name")
        if not name:
            raise ValueError("Each experiment entry must define a 'name'.")
        normalized[str(name)] = copy.deepcopy(entry)
    return normalized


def resolve_experiment_config(
    loss_config_path: str | Path,
    experiment_name: str | None,
) -> dict[str, Any]:
    payload = _load_yaml(loss_config_path)
    defaults = payload.get("defaults", {})
    if defaults and not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping.")

    experiment_entries = _normalize_experiment_entries(payload)
    if not experiment_entries:
        raise ValueError("No experiments were found in the experiment config.")

    selected_name = experiment_name
    if not selected_name:
        if len(experiment_entries) != 1:
            available = ", ".join(sorted(experiment_entries))
            raise ValueError(
                "experiment_name must be provided when the config defines multiple "
                f"experiments. Available: {available}"
            )
        selected_name = next(iter(experiment_entries))

    if selected_name not in experiment_entries:
        available = ", ".join(sorted(experiment_entries))
        raise ValueError(
            f"Unknown experiment '{selected_name}'. Available: {available}"
        )

    resolved = _deep_merge(defaults, experiment_entries[selected_name])
    resolved.setdefault("name", selected_name)
    return _normalize_experiment(resolved)


def list_experiment_names(loss_config_path: str | Path) -> list[str]:
    payload = _load_yaml(loss_config_path)
    return sorted(_normalize_experiment_entries(payload).keys())


def experiment_to_env_map(resolved: dict[str, Any]) -> dict[str, str]:
    env_map: dict[str, str] = {}
    for key, env_name in ENV_KEY_MAP.items():
        env_map[env_name] = str(resolved[key])

    env_map["LR"] = str(resolved["optimizer"]["lr"])
    env_map["WEIGHT_DECAY"] = str(resolved["optimizer"]["weight_decay"])
    env_map["ALL_VIDEOS"] = "true" if resolved["all_videos"] else "false"
    if resolved["videos"]:
        if len(resolved["videos"]) == 1:
            env_map["VIDEO_NAME"] = resolved["videos"][0]
        else:
            env_map["VIDEO_NAMES"] = "|".join(resolved["videos"])

    temporal = resolved["temporal"]
    env_map["TEMPORAL_WINDOW_SIZE"] = str(temporal["window_size"])
    env_map["TEMPORAL_WINDOW_STRIDE"] = str(temporal["window_stride"])
    env_map["TEMPORAL_MAX_FRAME_GAP"] = str(temporal["max_frame_gap"])
    env_map["TEMPORAL_REDUCTION"] = temporal["reduction"]
    env_map["TEMPORAL_SCORER_HIDDEN_DIM"] = str(temporal["scorer_hidden_dim"])
    env_map["TEMPORAL_SCORER_LAYERS"] = str(temporal["scorer_layers"])
    env_map["TEMPORAL_SCORER_DROPOUT"] = str(temporal["scorer_dropout"])

    lora = resolved["lora"]
    env_map["LORA_ENABLED"] = "true" if lora["enabled"] else "false"
    env_map["LORA_RANK"] = str(lora["rank"])
    env_map["LORA_ALPHA"] = str(lora["alpha"])
    env_map["LORA_DROPOUT"] = str(lora["dropout"])
    env_map["LORA_BLOCK_START"] = str(lora["block_start"])
    env_map["LORA_BLOCK_END"] = str(lora["block_end"])
    env_map["LORA_TARGET_MODULES"] = ",".join(lora["target_modules"])

    for family_name, family_cfg in resolved["losses"].items():
        prefix = family_name.upper()
        env_map[f"{prefix}_ENABLED"] = "true" if family_cfg["enabled"] else "false"
        env_map[f"{prefix}_WEIGHT"] = str(family_cfg["weight"])
        env_map[f"{prefix}_SCORER_WEIGHT"] = str(family_cfg["scorer_weight"])
        if family_name != "vipe_camera":
            env_map[f"{prefix}_FORMULATION"] = str(family_cfg["formulation"])
        if family_name == "temporal_vipe_camera":
            env_map[f"{prefix}_SMOOTHNESS_WEIGHT"] = str(
                family_cfg["smoothness_weight"]
            )
            env_map[f"{prefix}_ANCHOR_WEIGHT"] = str(family_cfg["anchor_weight"])

    return env_map


def _emit_shell_env(env_map: dict[str, str]) -> None:
    for key, value in env_map.items():
        encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
        print(f"{key}\t{encoded}")


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve WiLoR temporal experiment configs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve a named experiment.")
    resolve_parser.add_argument("--loss-config", type=str, required=True)
    resolve_parser.add_argument("--experiment-name", type=str, default=None)
    resolve_parser.add_argument(
        "--format",
        type=str,
        choices=["json", "shell"],
        default="json",
    )

    list_parser = subparsers.add_parser("list", help="List experiment names.")
    list_parser.add_argument("--loss-config", type=str, required=True)
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    if args.command == "list":
        for name in list_experiment_names(args.loss_config):
            print(name)
        return

    resolved = resolve_experiment_config(args.loss_config, args.experiment_name)
    if args.format == "shell":
        _emit_shell_env(experiment_to_env_map(resolved))
    else:
        print(json.dumps(resolved, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

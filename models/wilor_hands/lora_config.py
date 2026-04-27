from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any


SUPPORTED_LORA_TARGET_MODULES = ("qkv", "proj")

DEFAULT_LORA_CONFIG = {
    "enabled": False,
    "rank": 8,
    "alpha": 16.0,
    "dropout": 0.0,
    "block_start": 24,
    "block_end": 32,
    "target_modules": ["qkv"],
}


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


def _ensure_target_modules(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_LORA_CONFIG["target_modules"])
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
        items = [item for item in raw_items if item]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in SUPPORTED_LORA_TARGET_MODULES:
            allowed = ", ".join(SUPPORTED_LORA_TARGET_MODULES)
            raise ValueError(
                f"Unsupported LoRA target module '{item}'. Allowed values: {allowed}"
            )
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    if not normalized:
        raise ValueError("LoRA target_modules must contain at least one module.")
    return normalized


def normalize_lora_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_LORA_CONFIG)
    if value:
        for key, raw_value in value.items():
            config[str(key)] = copy.deepcopy(raw_value)

    config["enabled"] = _to_bool(config["enabled"])
    config["rank"] = int(config["rank"])
    config["alpha"] = float(config["alpha"])
    config["dropout"] = float(config["dropout"])
    config["block_start"] = int(config["block_start"])
    config["block_end"] = int(config["block_end"])
    config["target_modules"] = _ensure_target_modules(config["target_modules"])

    if config["rank"] <= 0:
        raise ValueError("LoRA rank must be > 0.")
    if config["alpha"] <= 0.0:
        raise ValueError("LoRA alpha must be > 0.")
    if config["dropout"] < 0.0:
        raise ValueError("LoRA dropout must be >= 0.")
    if config["block_start"] < 0:
        raise ValueError("LoRA block_start must be >= 0.")
    if config["block_end"] <= config["block_start"]:
        raise ValueError("LoRA block_end must be greater than block_start.")

    return config


def infer_lora_config_from_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    explicit_sources = [
        payload.get("lora_config"),
        payload.get("experiment_config", {}).get("lora")
        if isinstance(payload.get("experiment_config"), Mapping)
        else None,
    ]
    for source in explicit_sources:
        if isinstance(source, Mapping):
            config = normalize_lora_config(source)
            if config["enabled"]:
                return config

    finetune_args = payload.get("finetune_args")
    if isinstance(finetune_args, Mapping):
        arg_config = {
            "enabled": finetune_args.get("lora_enabled"),
            "rank": finetune_args.get("lora_rank"),
            "alpha": finetune_args.get("lora_alpha"),
            "dropout": finetune_args.get("lora_dropout"),
            "block_start": finetune_args.get("lora_block_start"),
            "block_end": finetune_args.get("lora_block_end"),
            "target_modules": finetune_args.get("lora_target_modules"),
        }
        if any(value is not None for value in arg_config.values()):
            config = normalize_lora_config(arg_config)
            if config["enabled"]:
                return config

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        return None

    lora_keys = [
        key
        for key in state_dict
        if ".lora_down.weight" in key or ".lora_up.weight" in key
    ]
    if not lora_keys:
        return None

    block_indices: set[int] = set()
    target_modules: list[str] = []
    rank = None
    for key in lora_keys:
        parts = key.split(".")
        if "blocks" in parts:
            block_pos = parts.index("blocks")
            if block_pos + 1 < len(parts):
                try:
                    block_indices.add(int(parts[block_pos + 1]))
                except ValueError:
                    pass
        if "attn" in parts:
            attn_pos = parts.index("attn")
            if attn_pos + 1 < len(parts):
                target = parts[attn_pos + 1]
                if target in SUPPORTED_LORA_TARGET_MODULES and target not in target_modules:
                    target_modules.append(target)
        tensor = state_dict[key]
        if rank is None and key.endswith(".lora_down.weight") and hasattr(tensor, "shape"):
            rank = int(tensor.shape[0])

    inferred = copy.deepcopy(DEFAULT_LORA_CONFIG)
    inferred["enabled"] = True
    if rank is not None:
        inferred["rank"] = rank
        inferred["alpha"] = float(rank)
    if block_indices:
        inferred["block_start"] = min(block_indices)
        inferred["block_end"] = max(block_indices) + 1
    if target_modules:
        inferred["target_modules"] = target_modules
    return normalize_lora_config(inferred)

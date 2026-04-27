from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn

from lora_config import normalize_lora_config


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_down = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_layer.out_features, bias=False)

        for param in self.base.parameters():
            param.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        return base_out + lora_out

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()


def apply_lora_to_wilor(model: nn.Module, lora_config: dict[str, Any] | None) -> list[str]:
    cfg = normalize_lora_config(lora_config)
    if not cfg["enabled"]:
        return []

    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "blocks"):
        raise ValueError("LoRA is currently supported only for WiLoR ViT backbones with .blocks.")

    block_count = len(backbone.blocks)
    block_start = int(cfg["block_start"])
    block_end = min(int(cfg["block_end"]), block_count)
    if block_start >= block_end:
        raise ValueError(
            f"Resolved LoRA block range [{block_start}, {block_end}) is empty for backbone depth {block_count}."
        )

    wrapped_modules: list[str] = []
    for block_idx in range(block_start, block_end):
        attn = backbone.blocks[block_idx].attn
        for target_name in cfg["target_modules"]:
            target_module = getattr(attn, target_name)
            if isinstance(target_module, LoRALinear):
                continue
            if not isinstance(target_module, nn.Linear):
                raise TypeError(
                    f"Expected backbone.blocks[{block_idx}].attn.{target_name} to be nn.Linear, "
                    f"got {type(target_module).__name__}."
                )
            setattr(
                attn,
                target_name,
                LoRALinear(
                    target_module,
                    rank=int(cfg["rank"]),
                    alpha=float(cfg["alpha"]),
                    dropout=float(cfg["dropout"]),
                ),
            )
            wrapped_modules.append(f"backbone.blocks.{block_idx}.attn.{target_name}")

    return wrapped_modules


def has_lora_modules(model: nn.Module) -> bool:
    return any(isinstance(module, LoRALinear) for module in model.modules())


def unfreeze_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        for param in module.lora_parameters():
            param.requires_grad = True
            params.append(param)
    return params

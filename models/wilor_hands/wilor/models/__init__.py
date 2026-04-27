from pathlib import Path

import torch

from lora_config import infer_lora_config_from_checkpoint_payload, normalize_lora_config
from wilor.configs import get_config

from .discriminator import Discriminator
from .lora import apply_lora_to_wilor
from .mano_wrapper import MANO
from .wilor import WiLoR


def _get_model_root(cfg_path):
    cfg_path = Path(cfg_path).resolve()
    if cfg_path.parent.name == "pretrained_models":
        return cfg_path.parent.parent
    return cfg_path.parent


def _prepare_wilor_cfg(cfg_path, *, strip_backbone_pretrained=False, require_backbone_pretrained=False):
    model_root = _get_model_root(cfg_path)
    model_cfg = get_config(str(Path(cfg_path).resolve()), update_cachedir=False)

    if ("vit" in model_cfg.MODEL.BACKBONE.TYPE) and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    model_cfg.defrost()

    if "DATA_DIR" in model_cfg.MANO:
        mano_dir = model_root / "mano_data"
        model_cfg.MANO.DATA_DIR = str(mano_dir)
        model_cfg.MANO.MODEL_PATH = str(mano_dir)
        model_cfg.MANO.MEAN_PARAMS = str(mano_dir / "mano_mean_params.npz")

    backbone_pretrained = model_cfg.MODEL.BACKBONE.get("PRETRAINED_WEIGHTS", None)
    if backbone_pretrained:
        resolved_backbone_path = Path(backbone_pretrained)
        if not resolved_backbone_path.is_absolute():
            resolved_backbone_path = (model_root / resolved_backbone_path).resolve()
        if strip_backbone_pretrained:
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        else:
            if not resolved_backbone_path.exists():
                raise FileNotFoundError(
                    f"Backbone pretrained weights not found: {resolved_backbone_path}"
                )
            model_cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = str(resolved_backbone_path)
    elif require_backbone_pretrained:
        raise ValueError(
            "MODEL.BACKBONE.PRETRAINED_WEIGHTS must be set in the WiLoR config for fresh initialization."
        )

    model_cfg.freeze()
    return model_cfg


def build_wilor(cfg_path):
    model_cfg = _prepare_wilor_cfg(
        cfg_path,
        strip_backbone_pretrained=False,
        require_backbone_pretrained=True,
    )
    model = WiLoR(cfg=model_cfg)
    return model, model_cfg


def _load_checkpoint_payload(checkpoint_path):
    old_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return old_load(*args, **kwargs)

    torch.load = unsafe_load
    try:
        return torch.load(checkpoint_path, map_location="cpu")
    finally:
        torch.load = old_load


def load_wilor(checkpoint_path, cfg_path, lora_config=None):
    print("Loading ", checkpoint_path)
    model_cfg = _prepare_wilor_cfg(cfg_path, strip_backbone_pretrained=True)
    checkpoint_data = _load_checkpoint_payload(checkpoint_path)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)

    checkpoint_lora_config = None
    if isinstance(checkpoint_data, dict):
        checkpoint_lora_config = infer_lora_config_from_checkpoint_payload(checkpoint_data)

    requested_lora_config = normalize_lora_config(lora_config)

    model = WiLoR(cfg=model_cfg)
    if checkpoint_lora_config and checkpoint_lora_config["enabled"]:
        apply_lora_to_wilor(model, checkpoint_lora_config)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
        if requested_lora_config["enabled"]:
            apply_lora_to_wilor(model, requested_lora_config)
    return model, model_cfg

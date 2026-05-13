from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


MODULE_DIR = Path(__file__).resolve().parent
STRIDE_CONFIGS_DIR = MODULE_DIR / "stride_configs"


@dataclass(frozen=True)
class StrideRuntimeConfig:
    use_gpu: bool
    visualize: bool
    mano_model_path: Path


@dataclass(frozen=True)
class HMPPoseWeights:
    rot: float
    pos: float
    root: float
    latent: float


@dataclass(frozen=True)
class HMPBetaWeights:
    joint: float
    prior: float


@dataclass(frozen=True)
class StrideHMPConfig:
    config_path: Path
    backend: str
    runtime: StrideRuntimeConfig
    hmp_assets_root: Path
    hmp_model_config_path: Path
    overlap: int | None
    pose_iters: int
    pose_lr: float
    pose_weights: HMPPoseWeights
    beta_optimize: bool
    beta_iters: int
    beta_lr: float
    beta_weights: HMPBetaWeights


@dataclass(frozen=True)
class SimpleWeights:
    obs: float
    reproj: float
    shape: float
    cam_smooth: float
    pose_smooth: float
    joint_smooth: float
    anchor: float
    fft: float


@dataclass(frozen=True)
class StrideSimpleConfig:
    config_path: Path
    backend: str
    runtime: StrideRuntimeConfig
    iters: int
    lr: float
    weights: SimpleWeights
    fft_band_low_hz: float | None
    fft_band_high_hz: float | None
    fps: float
    pose_rank: int
    cam_rank: int


def default_stride_config_path(backend: str) -> Path:
    return STRIDE_CONFIGS_DIR / f"{backend}.yaml"


def load_stride_config(config_path: str | Path | None, backend: str) -> StrideHMPConfig | StrideSimpleConfig:
    resolved_path = _resolve_config_path(config_path, backend)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"STRIDE config must be a mapping: {resolved_path}")

    payload_backend = _require_str(payload, "backend", "root")
    if payload_backend != backend:
        raise ValueError(
            f"STRIDE config backend '{payload_backend}' does not match requested backend '{backend}'."
        )

    if backend == "hmp":
        return _parse_hmp_config(resolved_path, payload)
    if backend == "simple":
        return _parse_simple_config(resolved_path, payload)
    raise ValueError(f"Unsupported STRIDE backend: {backend}")


def _resolve_config_path(config_path: str | Path | None, backend: str) -> Path:
    if config_path is None:
        path = default_stride_config_path(backend)
    else:
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"STRIDE config not found: {path}")
    return path


def _resolve_module_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (MODULE_DIR / candidate).resolve()


def _parse_runtime(payload: dict[str, Any]) -> StrideRuntimeConfig:
    runtime = _require_mapping(payload, "runtime", "root")
    return StrideRuntimeConfig(
        use_gpu=_require_bool(runtime, "use_gpu", "runtime"),
        visualize=_require_bool(runtime, "visualize", "runtime"),
        mano_model_path=_resolve_module_path(_require_str(runtime, "mano_model_path", "runtime")),
    )


def _parse_hmp_config(path: Path, payload: dict[str, Any]) -> StrideHMPConfig:
    runtime = _require_mapping(payload, "runtime", "root")
    refinement = _require_mapping(payload, "refinement", "root")
    pose = _require_mapping(refinement, "pose", "refinement")
    pose_weights = _require_mapping(pose, "weights", "refinement.pose")
    beta = _require_mapping(refinement, "beta", "refinement")
    beta_weights = _require_mapping(beta, "weights", "refinement.beta")

    overlap = refinement.get("overlap")
    if overlap is not None and not _is_int(overlap):
        raise ValueError("refinement.overlap must be an integer or null.")
    if overlap is not None and int(overlap) < 0:
        raise ValueError("refinement.overlap must be >= 0 when provided.")

    return StrideHMPConfig(
        config_path=path,
        backend="hmp",
        runtime=_parse_runtime(payload),
        hmp_assets_root=_resolve_module_path(_require_str(runtime, "hmp_assets_root", "runtime")),
        hmp_model_config_path=_resolve_module_path(_require_str(runtime, "hmp_model_config", "runtime")),
        overlap=overlap,
        pose_iters=_require_positive_int(pose, "iters", "refinement.pose"),
        pose_lr=_require_positive_float(pose, "lr", "refinement.pose"),
        pose_weights=HMPPoseWeights(
            rot=_require_float(pose_weights, "rot", "refinement.pose.weights"),
            pos=_require_float(pose_weights, "pos", "refinement.pose.weights"),
            root=_require_float(pose_weights, "root", "refinement.pose.weights"),
            latent=_require_float(pose_weights, "latent", "refinement.pose.weights"),
        ),
        beta_optimize=_require_bool(beta, "optimize", "refinement.beta"),
        beta_iters=_require_non_negative_int(beta, "iters", "refinement.beta"),
        beta_lr=_require_positive_float(beta, "lr", "refinement.beta"),
        beta_weights=HMPBetaWeights(
            joint=_require_float(beta_weights, "joint", "refinement.beta.weights"),
            prior=_require_float(beta_weights, "prior", "refinement.beta.weights"),
        ),
    )


def _parse_simple_config(path: Path, payload: dict[str, Any]) -> StrideSimpleConfig:
    refinement = _require_mapping(payload, "refinement", "root")
    weights = _require_mapping(refinement, "weights", "refinement")

    return StrideSimpleConfig(
        config_path=path,
        backend="simple",
        runtime=_parse_runtime(payload),
        iters=_require_positive_int(refinement, "iters", "refinement"),
        lr=_require_positive_float(refinement, "lr", "refinement"),
        weights=SimpleWeights(
            obs=_require_float(weights, "obs", "refinement.weights"),
            reproj=_require_float(weights, "reproj", "refinement.weights"),
            shape=_require_float(weights, "shape", "refinement.weights"),
            cam_smooth=_require_float(weights, "cam_smooth", "refinement.weights"),
            pose_smooth=_require_float(weights, "pose_smooth", "refinement.weights"),
            joint_smooth=_require_float(weights, "joint_smooth", "refinement.weights"),
            anchor=_require_float(weights, "anchor", "refinement.weights"),
            fft=_require_float(weights, "fft", "refinement.weights"),
        ),
        fft_band_low_hz=_optional_float(refinement.get("fft_band_low_hz"), "refinement.fft_band_low_hz"),
        fft_band_high_hz=_optional_float(refinement.get("fft_band_high_hz"), "refinement.fft_band_high_hz"),
        fps=_require_float(refinement, "fps", "refinement"),
        pose_rank=_require_positive_int(refinement, "pose_rank", "refinement"),
        cam_rank=_require_positive_int(refinement, "cam_rank", "refinement"),
    )


def _require_mapping(mapping: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required mapping '{context}.{key}'.")
    return value


def _require_str(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string '{context}.{key}'.")
    return value


def _require_bool(mapping: dict[str, Any], key: str, context: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Missing required boolean '{context}.{key}'.")
    return value


def _require_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if not _is_int(value):
        raise ValueError(f"Missing required integer '{context}.{key}'.")
    return int(value)


def _require_float(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if not _is_number(value):
        raise ValueError(f"Missing required numeric value '{context}.{key}'.")
    return float(value)


def _require_positive_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = _require_int(mapping, key, context)
    if value <= 0:
        raise ValueError(f"{context}.{key} must be > 0.")
    return value


def _require_non_negative_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = _require_int(mapping, key, context)
    if value < 0:
        raise ValueError(f"{context}.{key} must be >= 0.")
    return value


def _require_positive_float(mapping: dict[str, Any], key: str, context: str) -> float:
    value = _require_float(mapping, key, context)
    if value <= 0:
        raise ValueError(f"{context}.{key} must be > 0.")
    return value


def _optional_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if not _is_number(value):
        raise ValueError(f"{context} must be numeric or null.")
    return float(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

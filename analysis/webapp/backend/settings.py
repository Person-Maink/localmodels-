from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


PRIMARY_FAMILIES = (
    "wilor",
    "wilor_finetune",
    "hamba",
    "dynhamr",
    "stride",
    "mediapipe",
)
CAMERA_COMPATIBLE_FAMILIES = ("wilor", "wilor_finetune", "hamba", "dynhamr", "stride")
BETA_COMPATIBLE_FAMILIES = ("wilor", "wilor_finetune", "dynhamr", "stride")
MODEL_FAMILIES = ("wilor", "wilor_finetune", "hamba", "dynhamr", "stride")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
VARIANT_SUFFIXES = ("_amplified_modified", "_modified", "_amplified")
SOURCE_COLORS = (
    "#0f4c81",
    "#c04f15",
    "#1f7a53",
    "#7a1f4a",
    "#8c6a10",
    "#345995",
    "#b33951",
    "#1b998b",
)


@dataclass(frozen=True)
class CameraDisplaySettings:
    fov_deg: float = 60.0
    aspect: float = 16.0 / 9.0
    frustum_scale: float = 0.5
    trajectory_stride: int = 5
    center_radius: float = 0.02
    center_to_first_frame: bool = True
    translation_scale: float = 1.0
    invert_cam_t: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "fov_deg": self.fov_deg,
            "aspect": self.aspect,
            "frustum_scale": self.frustum_scale,
            "trajectory_stride": self.trajectory_stride,
            "center_radius": self.center_radius,
            "center_to_first_frame": self.center_to_first_frame,
            "translation_scale": self.translation_scale,
            "invert_cam_t": self.invert_cam_t,
        }


@dataclass(frozen=True)
class AppSettings:
    repo_root: Path
    workspace_root: Path
    data_root: Path
    outputs_root: Path
    dynhamr_logs_root: Path
    dynhamr_output_root: Path
    mediapipe_root: Path
    vipe_pose_root: Path
    vipe_rgb_root: Path
    mano_right_path: Path
    temp_root: Path
    default_fps: float = 30.0
    default_hand: str = "right"
    default_wrist_joint_index: int = 0
    default_neighbor_count: int = 25
    max_cache_items: int = 24
    camera_display: CameraDisplaySettings = field(default_factory=CameraDisplaySettings)

    def source_color(self, index: int) -> str:
        return SOURCE_COLORS[index % len(SOURCE_COLORS)]


def _resolve_path(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default.resolve()


def load_settings() -> AppSettings:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    outputs_root = _resolve_path("TREMOR_OUTPUTS_ROOT", workspace_root / "outputs")
    temp_root = _resolve_path("TREMOR_WEBAPP_TEMP_ROOT", Path(tempfile.gettempdir()) / "tremor-webapp")
    temp_root.mkdir(parents=True, exist_ok=True)

    return AppSettings(
        repo_root=repo_root,
        workspace_root=workspace_root,
        data_root=_resolve_path("TREMOR_DATA_ROOT", workspace_root / "data"),
        outputs_root=outputs_root,
        dynhamr_logs_root=_resolve_path("TREMOR_DYNHAMR_LOGS_ROOT", outputs_root / "dynhamr" / "logs" / "video-custom"),
        dynhamr_output_root=_resolve_path("TREMOR_DYNHAMR_OUTPUT_ROOT", outputs_root / "dynhamr"),
        mediapipe_root=_resolve_path("TREMOR_MEDIAPIPE_ROOT", outputs_root / "mediapipe"),
        vipe_pose_root=_resolve_path("TREMOR_VIPE_POSE_ROOT", outputs_root / "vipe" / "pose"),
        vipe_rgb_root=_resolve_path("TREMOR_VIPE_RGB_ROOT", outputs_root / "vipe" / "rgb"),
        mano_right_path=_resolve_path("TREMOR_MANO_RIGHT_PATH", repo_root / "mano_data" / "MANO_RIGHT.pkl"),
        temp_root=temp_root,
        default_fps=float(os.environ.get("TREMOR_DEFAULT_FPS", "30.0")),
        default_hand=os.environ.get("TREMOR_DEFAULT_HAND", "right").strip().lower(),
        default_wrist_joint_index=int(os.environ.get("TREMOR_DEFAULT_WRIST_JOINT", "0")),
        default_neighbor_count=int(os.environ.get("TREMOR_DEFAULT_NEIGHBORS", "25")),
        max_cache_items=int(os.environ.get("TREMOR_WEBAPP_MAX_CACHE_ITEMS", "24")),
    )


def strip_variant_suffix(clip_id: str) -> str:
    current = str(clip_id)
    while True:
        next_value = current
        for suffix in VARIANT_SUFFIXES:
            if current.endswith(suffix):
                next_value = current[: -len(suffix)]
                break
        if next_value == current:
            return current
        current = next_value


def hand_value_to_label(hand_value: int) -> str:
    if int(hand_value) == 1:
        return "right"
    if int(hand_value) == 0:
        return "left"
    return "unknown"


def hand_label_to_value(hand_label: str) -> int:
    normalized = str(hand_label).strip().lower()
    if normalized in {"right", "r", "1"}:
        return 1
    if normalized in {"left", "l", "0"}:
        return 0
    return -1


def parse_pair_text(value: object) -> Tuple[Tuple[int, int], ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                result.append((int(item[0]), int(item[1])))
                continue
            if isinstance(item, dict):
                result.append((int(item["a"]), int(item["b"])))
                continue
            text = str(item).strip()
            if not text:
                continue
            left, right = text.replace(":", "-").split("-", 1)
            result.append((int(left.strip()), int(right.strip())))
        return tuple(result)

    text = str(value).strip()
    if not text:
        return tuple()

    pairs = []
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" not in part and ":" not in part:
            raise ValueError(f"Invalid pair specification '{part}'. Use '4-8, 9-13'.")
        left, right = part.replace(":", "-").split("-", 1)
        pairs.append((int(left.strip()), int(right.strip())))
    return tuple(pairs)


def parse_number_list(value: object) -> Tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if not text:
        return tuple()
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())

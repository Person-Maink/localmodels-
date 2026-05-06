import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from ultralytics import YOLO

from frame_store import FrameRecord, FrameStore
from vipe_artifacts import (
    load_vipe_intrinsics_artifact,
    load_vipe_pose_artifact,
    pick_value_for_frame,
)
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils import recursive_to
from wilor.models.lora import unfreeze_lora_parameters


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FRAME_FILE_PATTERNS = ("*.jpg", "*.png", "*.jpeg")
SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mts", ".mov")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(use_gpu: bool = True) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_frame_store(
    image_folder: str,
    frame_cache_root: Optional[str] = None,
) -> FrameStore:
    return FrameStore(image_folder, cache_root=frame_cache_root)


def list_existing_frame_paths(
    image_folder: str,
    file_types: Sequence[str] = FRAME_FILE_PATTERNS,
    video_exts: Sequence[str] = SUPPORTED_VIDEO_EXTS,
    frame_cache_root: Optional[str] = None,
) -> tuple[List[Path], List[Path]]:
    image_root = Path(image_folder)
    if not image_root.exists():
        raise FileNotFoundError(f"Image folder not found: {image_root}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image folder is not a directory: {image_root}")

    frame_store = make_frame_store(image_folder, frame_cache_root=frame_cache_root)
    frame_paths: List[Path] = []
    for ext in file_types:
        frame_paths.extend(sorted(image_root.glob(ext)))

    for video_name in frame_store.list_videos():
        if video_name == "single_images":
            continue
        source_path = frame_store.get_source_path(video_name)
        if source_path is None or not source_path.is_dir():
            continue
        for ext in file_types:
            frame_paths.extend(sorted(source_path.glob(ext)))

    raw_video_paths = [
        path
        for path in sorted(image_root.iterdir())
        if path.is_file() and path.suffix.lower() in video_exts
    ]
    return sorted(frame_paths), raw_video_paths


def discover_videos(
    image_folder: str,
    frame_cache_root: Optional[str] = None,
) -> List[str]:
    frame_store = make_frame_store(image_folder, frame_cache_root=frame_cache_root)
    return [video_name for video_name in frame_store.list_videos() if video_name != "single_images"]


def filter_videos_with_vipe_artifacts(
    video_names: Sequence[str],
    pose_dir: str,
    intrinsics_dir: str,
) -> tuple[List[str], Dict[str, List[str]]]:
    pose_root = Path(pose_dir)
    intrinsics_root = Path(intrinsics_dir)
    valid_videos: List[str] = []
    missing_artifacts: Dict[str, List[str]] = {}

    for video_name in sorted(set(video_names)):
        missing_paths = []
        pose_path = pose_root / f"{video_name}.npz"
        intrinsics_path = intrinsics_root / f"{video_name}.npz"
        if not pose_path.exists():
            missing_paths.append(str(pose_path))
        if not intrinsics_path.exists():
            missing_paths.append(str(intrinsics_path))

        if missing_paths:
            missing_artifacts[video_name] = missing_paths
        else:
            valid_videos.append(video_name)

    return valid_videos, missing_artifacts


def parse_frame_index(frame_name: str, pattern: str = r"(\d+)$") -> int:
    match = re.search(pattern, frame_name)
    if match is None:
        raise ValueError(
            f"Could not parse frame index from '{frame_name}' using pattern '{pattern}'."
        )
    return int(match.group(1))


def infer_video_name_from_path(path_like: str, override_video_name: Optional[str] = None) -> str:
    if override_video_name:
        return override_video_name

    path = Path(path_like)
    parents = [path.parent.name]
    parents.extend(part for part in path.parts[:-1] if part)
    for candidate in parents:
        if candidate.endswith("_frames"):
            return candidate[: -len("_frames")]
    if path.parent.name:
        return path.parent.name
    raise ValueError(f"Could not infer video name from path '{path_like}'.")


def _frame_record_display_paths(
    frame_store: FrameStore,
    record: FrameRecord,
) -> tuple[str, str]:
    if record.source_kind in {"loose_frames", "single_image"}:
        path = Path(record.source_key)
        return str(path), str(path)

    source_path = frame_store.get_source_path(record.video_name)
    source_display = str(source_path) if source_path is not None else record.video_name
    synthetic_name = f"{record.video_name}/{record.frame_name}.jpg"
    return f"{source_display}::{record.source_key}", synthetic_name


class ViPECameraIndex:
    def __init__(self, pose_dir: str, intrinsics_dir: str):
        self.pose_dir = Path(pose_dir)
        self.intrinsics_dir = Path(intrinsics_dir)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _load_video(self, video_name: str) -> Dict[str, np.ndarray]:
        if video_name not in self._cache:
            pose_inds, poses = load_vipe_pose_artifact(self.pose_dir / f"{video_name}.npz")
            intr_inds, intrinsics = load_vipe_intrinsics_artifact(
                self.intrinsics_dir / f"{video_name}.npz"
            )
            self._cache[video_name] = {
                "pose_inds": pose_inds,
                "poses": poses,
                "intr_inds": intr_inds,
                "intrinsics": intrinsics,
            }
        return self._cache[video_name]

    def resolve(self, video_name: str, frame_idx: int) -> Dict[str, np.ndarray]:
        video_data = self._load_video(video_name)
        pose_c2w, pose_frame_idx, pose_fallback = pick_value_for_frame(
            frame_idx,
            video_data["pose_inds"],
            video_data["poses"],
        )
        intrinsics, intr_frame_idx, intr_fallback = pick_value_for_frame(
            frame_idx,
            video_data["intr_inds"],
            video_data["intrinsics"],
        )
        pose_w2c = np.linalg.inv(pose_c2w)
        return {
            "camera_target_t_full": pose_w2c[:3, 3].astype(np.float32),
            "camera_target_valid": np.float32(1.0),
            "camera_target_pose_frame_idx": np.int64(pose_frame_idx),
            "camera_target_intrinsics_frame_idx": np.int64(intr_frame_idx),
            "camera_target_focal": intrinsics[:2].astype(np.float32),
            "camera_target_center": intrinsics[2:].astype(np.float32),
            "camera_target_used_fallback": np.float32(pose_fallback or intr_fallback),
        }


def build_detection_samples(
    image_folder: str,
    video_names: List[str],
    detector: YOLO,
    camera_index: ViPECameraIndex,
    detection_conf: float = 0.3,
    detection_cache_path: Optional[str] = None,
    sample_limit: int = 0,
    frame_cache_root: Optional[str] = None,
) -> List[Dict]:
    frame_store = make_frame_store(image_folder, frame_cache_root=frame_cache_root)
    if detection_cache_path and Path(detection_cache_path).exists():
        print(
            f"[progress] Loading detection cache from {detection_cache_path}",
            flush=True,
        )
        with open(detection_cache_path, "r", encoding="utf-8") as handle:
            cached_samples = json.load(handle)
        video_name_to_idx = {name: idx for idx, name in enumerate(sorted(set(video_names)))}
        for sample in cached_samples:
            img_path_rel = sample.get("img_path_rel")
            if img_path_rel:
                sample["img_path"] = str(img_path_rel)
            if "video_name" in sample and sample["video_name"] in video_name_to_idx:
                sample["video_idx"] = video_name_to_idx[str(sample["video_name"])]
        selected_videos = set(video_names)
        if not selected_videos:
            limited_samples = _limit_samples_by_frame(cached_samples, sample_limit)
            print(
                f"[progress] Loaded {len(cached_samples)} cached detection samples; "
                f"using {len(limited_samples)} after frame limiting.",
                flush=True,
            )
            return limited_samples
        filtered_samples = [
            sample
            for sample in cached_samples
            if sample.get("video_name") in selected_videos
        ]
        filtered_samples = _limit_samples_by_frame(filtered_samples, sample_limit)
        print(
            "[progress] Detection cache contains "
            f"{len(cached_samples)} samples; using {len(filtered_samples)} "
            f"samples across {len(selected_videos)} selected video(s).",
            flush=True,
        )
        return filtered_samples

    selected_videos = set(video_names)
    unavailable_messages = {
        video_name: frame_store.explain_unavailable(video_name)
        for video_name in sorted(selected_videos)
        if not frame_store.has_video(video_name)
    }
    if unavailable_messages:
        for video_name, reason in unavailable_messages.items():
            print(f"[progress] Skipping unavailable video '{video_name}': {reason}", flush=True)

    selected_frame_records: List[FrameRecord] = []
    for video_name in sorted(selected_videos):
        if not frame_store.has_video(video_name):
            continue
        selected_frame_records.extend(frame_store.iter_video_frames(video_name))
    selected_frame_records.sort(key=lambda record: (record.video_name, int(record.frame_id)))
    if sample_limit > 0:
        selected_frame_records = selected_frame_records[:sample_limit]

    print(
        "[progress] Prepared "
        f"{len(selected_frame_records)} frame(s) for detection across "
        f"{len(selected_videos) - len(unavailable_messages)} selected video(s).",
        flush=True,
    )
    if not selected_frame_records:
        print(
            "[progress] No frame records matched the selected video(s). "
            "Build sidecar ZIP caches or provide legacy *_frames directories.",
            flush=True,
        )

    samples: List[Dict] = []
    video_name_to_idx = {name: idx for idx, name in enumerate(sorted(set(video_names)))}
    total_frames = len(selected_frame_records)
    progress_interval = 25

    for frame_number, frame_record in enumerate(selected_frame_records, start=1):
        if frame_number == 1 or frame_number % progress_interval == 0 or frame_number == total_frames:
            print(
                "[progress] Running detector on frame "
                f"{frame_number}/{total_frames}: {frame_record.video_name}:{frame_record.frame_name}",
                flush=True,
            )
        video_name = frame_record.video_name
        frame_idx = int(frame_record.frame_id)
        camera_target = camera_index.resolve(video_name, frame_idx)

        img_cv2 = frame_store.get_frame(video_name, frame_idx)
        if img_cv2 is None:
            continue

        detections = detector(img_cv2, conf=detection_conf, verbose=False)[0]
        img_path, img_path_rel = _frame_record_display_paths(frame_store, frame_record)
        frame_samples = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().reshape(-1, det.boxes.data.shape[-1])[0].numpy()
            right = float(det.boxes.cls.cpu().detach().reshape(-1)[0].item())
            x_center = float((bbox[0] + bbox[2]) * 0.5)
            frame_samples.append(
                {
                    "img_path": img_path,
                    "img_path_rel": img_path_rel,
                    "video_name": video_name,
                    "video_idx": video_name_to_idx[video_name],
                    "frame_idx": frame_idx,
                    "frame_name": frame_record.frame_name,
                    "bbox": [float(v) for v in bbox[:4]],
                    "right": right,
                    "x_center": x_center,
                    **{
                        key: value.tolist() if isinstance(value, np.ndarray) else value.item()
                        if isinstance(value, np.generic)
                        else value
                        for key, value in camera_target.items()
                    },
                }
            )
        frame_samples.sort(key=lambda item: (item["right"], item["x_center"]))
        for det_idx, sample in enumerate(frame_samples):
            sample["det_idx"] = det_idx
            samples.append(sample)

    if detection_cache_path:
        cache_path = Path(detection_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(samples, handle)
        print(
            f"[progress] Saved {len(samples)} detection samples to {cache_path}",
            flush=True,
        )

    print(
        f"[progress] Finished detection sampling with {len(samples)} sample(s).",
        flush=True,
    )

    return samples


def get_sample_frame_key(sample: Dict[str, Any]) -> Hashable:
    if "video_name" in sample and "frame_idx" in sample:
        return str(sample["video_name"]), int(sample["frame_idx"])

    frame_path = sample.get("img_path_rel") or sample.get("imgname_rel")
    if frame_path is None:
        frame_path = sample.get("img_path") or sample.get("imgname")
    if frame_path is None:
        raise KeyError("Sample does not contain a usable frame identifier.")
    return str(frame_path)


def _limit_samples_by_frame(
    samples: Sequence[Dict[str, Any]],
    sample_limit: int,
) -> List[Dict[str, Any]]:
    if sample_limit <= 0:
        return list(samples)

    kept_frame_keys: set[Hashable] = set()
    limited_samples: List[Dict[str, Any]] = []
    for sample in samples:
        frame_key = get_sample_frame_key(sample)
        if frame_key not in kept_frame_keys:
            if len(kept_frame_keys) >= sample_limit:
                continue
            kept_frame_keys.add(frame_key)
        limited_samples.append(sample)
    return limited_samples


def split_samples_by_frame(
    samples: Sequence[Dict[str, Any]],
    validation_split: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int | float]]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError(
            f"validation_split must be in [0.0, 1.0). Received {validation_split}."
        )

    if not samples:
        return [], [], {
            "validation_split": validation_split,
            "total_frames": 0,
            "train_frames": 0,
            "val_frames": 0,
            "total_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
        }

    unique_frame_keys = sorted({get_sample_frame_key(sample) for sample in samples}, key=str)
    val_frame_count = int(round(len(unique_frame_keys) * validation_split))
    if validation_split > 0.0 and len(unique_frame_keys) > 1:
        val_frame_count = max(1, val_frame_count)
        val_frame_count = min(val_frame_count, len(unique_frame_keys) - 1)
    else:
        val_frame_count = 0

    shuffled_frame_keys = list(unique_frame_keys)
    random.Random(seed).shuffle(shuffled_frame_keys)
    val_frame_keys = set(shuffled_frame_keys[:val_frame_count])

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    for sample in samples:
        target = val_samples if get_sample_frame_key(sample) in val_frame_keys else train_samples
        target.append(sample)

    return train_samples, val_samples, {
        "validation_split": validation_split,
        "total_frames": len(unique_frame_keys),
        "train_frames": len(unique_frame_keys) - len(val_frame_keys),
        "val_frames": len(val_frame_keys),
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }


class DetectedVideoHandDataset(Dataset):
    def __init__(
        self,
        model_cfg,
        samples: List[Dict],
        rescale_factor: float = 2.0,
        include_path_metadata: bool = True,
        frame_store: Optional[FrameStore] = None,
    ):
        self.model_cfg = model_cfg
        self.samples = samples
        self.rescale_factor = rescale_factor
        self.include_path_metadata = include_path_metadata
        self.frame_store = frame_store
        self._sample_local_index_by_global_idx: Dict[int, int] = {}
        self._samples_by_frame_key: Dict[Hashable, List[Dict[str, Any]]] = {}
        self._frame_cache: Dict[Hashable, List[Dict[str, Any]]] = {}

        for idx, sample in enumerate(self.samples):
            frame_key = self._sample_frame_key(sample)
            frame_samples = self._samples_by_frame_key.setdefault(frame_key, [])
            self._sample_local_index_by_global_idx[idx] = len(frame_samples)
            frame_samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frame_key(self, sample: Dict[str, Any]) -> Hashable:
        if self.frame_store is not None and "video_name" in sample and "frame_idx" in sample:
            return str(sample["video_name"]), int(sample["frame_idx"])
        return str(sample["img_path"])

    def _get_frame_items(self, frame_key: Hashable) -> List[Dict[str, Any]]:
        cached_items = self._frame_cache.get(frame_key)
        if cached_items is not None:
            return cached_items

        frame_samples = self._samples_by_frame_key[frame_key]
        exemplar_sample = frame_samples[0]
        if self.frame_store is not None and isinstance(frame_key, tuple):
            img_cv2 = self.frame_store.get_frame(str(frame_key[0]), int(frame_key[1]))
        else:
            img_cv2 = cv2.imread(str(exemplar_sample["img_path"]))
        if img_cv2 is None:
            raise RuntimeError(f"Failed to read image/frame: {frame_key}")

        crop_dataset = ViTDetDataset(
            self.model_cfg,
            img_cv2,
            np.asarray([sample["bbox"] for sample in frame_samples], dtype=np.float32),
            np.asarray([sample["right"] for sample in frame_samples], dtype=np.float32),
            rescale_factor=self.rescale_factor,
        )
        cached_items = [dict(crop_dataset[item_idx]) for item_idx in range(len(frame_samples))]
        self._frame_cache[frame_key] = cached_items
        return cached_items

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        frame_key = self._sample_frame_key(sample)
        frame_items = self._get_frame_items(frame_key)
        frame_item_index = self._sample_local_index_by_global_idx[idx]
        item = dict(frame_items[frame_item_index])
        if self.include_path_metadata:
            item["imgname"] = sample["img_path"]
            item["imgname_rel"] = sample["img_path_rel"]
            item["video_name"] = sample["video_name"]
        item["video_idx"] = np.int64(sample["video_idx"])
        item["frame_idx"] = np.int64(sample["frame_idx"])
        item["det_idx"] = np.int64(sample["det_idx"])
        item["camera_target_t_full"] = np.asarray(sample["camera_target_t_full"], dtype=np.float32)
        item["camera_target_valid"] = np.float32(sample["camera_target_valid"])
        item["camera_target_pose_frame_idx"] = np.int64(sample["camera_target_pose_frame_idx"])
        item["camera_target_intrinsics_frame_idx"] = np.int64(
            sample["camera_target_intrinsics_frame_idx"]
        )
        item["camera_target_focal"] = np.asarray(sample["camera_target_focal"], dtype=np.float32)
        item["camera_target_center"] = np.asarray(sample["camera_target_center"], dtype=np.float32)
        item["camera_target_used_fallback"] = np.float32(sample["camera_target_used_fallback"])
        return item


class SupervisedCameraDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        camera_index: ViPECameraIndex,
        frame_index_pattern: str = r"(\d+)$",
        override_video_name: Optional[str] = None,
    ):
        self.base_dataset = base_dataset
        self.camera_index = camera_index
        self.frame_index_pattern = frame_index_pattern
        self.override_video_name = override_video_name
        self.video_name_to_idx: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _get_video_idx(self, video_name: str) -> int:
        if video_name not in self.video_name_to_idx:
            self.video_name_to_idx[video_name] = len(self.video_name_to_idx)
        return self.video_name_to_idx[video_name]

    def __getitem__(self, idx: int) -> Dict:
        item = self.base_dataset[idx]
        rel_path = item.get("imgname_rel", item.get("imgname"))
        video_name = infer_video_name_from_path(rel_path, self.override_video_name)
        frame_idx = parse_frame_index(Path(rel_path).stem, self.frame_index_pattern)
        camera_target = self.camera_index.resolve(video_name, frame_idx)

        item["video_name"] = video_name
        item["video_idx"] = np.int64(self._get_video_idx(video_name))
        item["frame_idx"] = np.int64(frame_idx)
        item.update(camera_target)
        return item


def build_teacher_supervision_batch(batch: Dict, teacher_output: Dict) -> Dict:
    pred_keypoints_2d = teacher_output["pred_keypoints_2d"].detach()
    pred_keypoints_3d = teacher_output["pred_keypoints_3d"].detach()
    batch_size, num_keypoints, _ = pred_keypoints_2d.shape
    device = pred_keypoints_2d.device
    dtype = pred_keypoints_2d.dtype

    gt_keypoints_2d = torch.cat(
        [
            pred_keypoints_2d,
            torch.ones(batch_size, num_keypoints, 1, device=device, dtype=dtype),
        ],
        dim=-1,
    )
    gt_keypoints_3d = torch.cat(
        [
            pred_keypoints_3d,
            torch.ones(batch_size, pred_keypoints_3d.shape[1], 1, device=device, dtype=dtype),
        ],
        dim=-1,
    )

    gt_mano_params = {
        key: value.detach().clone()
        for key, value in teacher_output["pred_mano_params"].items()
    }
    has_mano_params = {
        key: torch.ones(batch_size, device=device, dtype=dtype) for key in gt_mano_params
    }
    mano_params_is_axis_angle = {
        key: torch.zeros(batch_size, device=device, dtype=torch.bool) for key in gt_mano_params
    }

    supervision_batch = dict(batch)
    supervision_batch["keypoints_2d"] = gt_keypoints_2d
    supervision_batch["keypoints_3d"] = gt_keypoints_3d
    supervision_batch["mano_params"] = gt_mano_params
    supervision_batch["has_mano_params"] = has_mano_params
    supervision_batch["mano_params_is_axis_angle"] = mano_params_is_axis_angle
    return supervision_batch


def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def configure_trainable_scope(
    model: torch.nn.Module,
    scope: str,
    *,
    include_lora: bool = False,
) -> List[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    if scope == "temporal_only":
        modules = []
    elif scope == "camera_head":
        modules = [model.backbone.cam_emb, model.backbone.deccam, model.refine_net.dec_cam]
    elif scope == "refine_net":
        modules = [model.refine_net]
    elif scope == "full":
        modules = [model]
    else:
        raise ValueError(f"Unsupported train scope '{scope}'.")

    for module in modules:
        unfreeze_module(module)

    if include_lora:
        unfreeze_lora_parameters(model)

    if hasattr(model, "discriminator"):
        freeze_module(model.discriminator)

    return [param for param in model.parameters() if param.requires_grad]


def apply_train_mode_for_scope(
    model: torch.nn.Module,
    scope: str,
    *,
    lora_enabled: bool = False,
) -> None:
    if scope == "temporal_only" and not lora_enabled:
        model.eval()
    else:
        model.train()
    if scope in {"camera_head", "refine_net", "temporal_only"} and not lora_enabled:
        model.backbone.eval()
    if hasattr(model, "discriminator"):
        model.discriminator.eval()


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def set_optional_loss_weight(cfg, key: str, value: float) -> None:
    cfg.defrost()
    cfg.LOSS_WEIGHTS[key] = value
    cfg.freeze()


def load_detector(detector_path: str, device: torch.device) -> YOLO:
    print(
        f"[progress] Loading YOLO detector from {detector_path} onto {device}",
        flush=True,
    )
    old_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return old_load(*args, **kwargs)

    torch.load = unsafe_load
    try:
        detector = YOLO(detector_path)
    finally:
        torch.load = old_load

    detector = detector.to(str(device))
    print("[progress] Detector ready.", flush=True)
    return detector


def append_metrics(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_wilor_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    epoch: int,
    extra: Optional[Dict] = None,
    extra_modules: Optional[Dict[str, torch.nn.Module]] = None,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "pytorch-lightning_version": pl.__version__,
        "hyper_parameters": {"cfg": model.cfg},
    }
    if optimizer is not None:
        checkpoint["optimizer_states"] = [optimizer.state_dict()]
    if extra_modules:
        checkpoint["extra_state_dicts"] = {
            name: module.state_dict() for name, module in extra_modules.items()
        }
    if extra:
        checkpoint.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def infinite_loader(dataloader: Iterable):
    while True:
        for batch in dataloader:
            yield batch


def format_loss_dict(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {
        key: float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
        for key, value in losses.items()
    }


def evaluate_distillation(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    amp: bool = True,
) -> Dict[str, float]:
    autocast_enabled = amp and device.type == "cuda"
    was_student_training = student.training
    was_teacher_training = teacher.training
    student.eval()
    teacher.eval()

    total_samples = 0
    total_batches = 0
    metric_sums: Dict[str, float] = {}

    with torch.no_grad():
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                teacher_output = teacher.forward_step(batch, train=False)
                supervision_batch = build_teacher_supervision_batch(batch, teacher_output)
                student_output = student.forward_step(batch, train=False)
                loss = student.compute_loss(supervision_batch, student_output, train=False)

            batch_size = int(batch["img"].shape[0])
            batch_metrics = format_loss_dict(student_output["losses"])
            batch_metrics["loss_total"] = float(loss.detach().item())
            if "camera_target_used_fallback" in batch:
                batch_metrics["camera_target_fallback_rate"] = float(
                    batch["camera_target_used_fallback"].float().mean().item()
                )

            total_samples += batch_size
            total_batches += 1
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size

    student.train(was_student_training)
    teacher.train(was_teacher_training)

    if total_samples == 0:
        return {"num_samples": 0.0, "num_batches": 0.0}

    averaged_metrics = {
        key: value / total_samples for key, value in metric_sums.items()
    }
    averaged_metrics["num_samples"] = float(total_samples)
    averaged_metrics["num_batches"] = float(total_batches)
    return averaged_metrics

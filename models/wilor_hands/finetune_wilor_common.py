import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from ultralytics import YOLO

from loader import load_images_from_folder
from wilor.datasets.vitdet_dataset import ViTDetDataset


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def load_vipe_pose_artifact(pose_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not pose_path.exists():
        raise FileNotFoundError(f"ViPE pose file not found: {pose_path}")

    data = np.load(pose_path)
    if "inds" not in data or "data" not in data:
        raise ValueError(f"{pose_path} must contain 'inds' and 'data'.")
    inds = np.asarray(data["inds"], dtype=np.int64)
    poses = np.asarray(data["data"], dtype=np.float32)
    if inds.ndim != 1 or poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"Unexpected ViPE pose shapes in {pose_path}: inds={inds.shape}, data={poses.shape}"
        )
    order = np.argsort(inds)
    return inds[order], poses[order]


def load_vipe_intrinsics_artifact(intrinsics_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"ViPE intrinsics file not found: {intrinsics_path}")

    data = np.load(intrinsics_path)
    if "inds" not in data or "data" not in data:
        raise ValueError(f"{intrinsics_path} must contain 'inds' and 'data'.")
    inds = np.asarray(data["inds"], dtype=np.int64)
    intrinsics = np.asarray(data["data"], dtype=np.float32)
    if inds.ndim != 1 or intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
        raise ValueError(
            f"Unexpected ViPE intrinsics shapes in {intrinsics_path}: inds={inds.shape}, data={intrinsics.shape}"
        )
    order = np.argsort(inds)
    return inds[order], intrinsics[order]


def pick_value_for_frame(
    frame_idx: int,
    inds: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    pos = np.searchsorted(inds, frame_idx, side="right") - 1
    if pos >= 0:
        source_idx = int(inds[pos])
        return values[pos], source_idx, frame_idx != source_idx
    return values[0], int(inds[0]), True


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
) -> List[Dict]:
    if detection_cache_path and Path(detection_cache_path).exists():
        with open(detection_cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    all_frame_paths = load_images_from_folder(image_folder)
    selected_frame_paths = [
        Path(img_path)
        for img_path in all_frame_paths
        if Path(img_path).parent.name.replace("_frames", "") in set(video_names)
    ]
    selected_frame_paths = sorted(selected_frame_paths)
    if sample_limit > 0:
        selected_frame_paths = selected_frame_paths[:sample_limit]

    samples: List[Dict] = []
    video_name_to_idx = {name: idx for idx, name in enumerate(sorted(set(video_names)))}

    for frame_path in selected_frame_paths:
        video_name = frame_path.parent.name.replace("_frames", "")
        frame_idx = parse_frame_index(frame_path.stem)
        camera_target = camera_index.resolve(video_name, frame_idx)

        img_cv2 = cv2.imread(str(frame_path))
        if img_cv2 is None:
            continue

        detections = detector(img_cv2, conf=detection_conf, verbose=False)[0]
        frame_samples = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().reshape(-1, det.boxes.data.shape[-1])[0].numpy()
            right = float(det.boxes.cls.cpu().detach().reshape(-1)[0].item())
            x_center = float((bbox[0] + bbox[2]) * 0.5)
            frame_samples.append(
                {
                    "img_path": str(frame_path),
                    "img_path_rel": str(frame_path.relative_to(Path(image_folder))),
                    "video_name": video_name,
                    "video_idx": video_name_to_idx[video_name],
                    "frame_idx": frame_idx,
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

    return samples


class DetectedVideoHandDataset(Dataset):
    def __init__(self, model_cfg, samples: List[Dict], rescale_factor: float = 2.0):
        self.model_cfg = model_cfg
        self.samples = samples
        self.rescale_factor = rescale_factor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img_cv2 = cv2.imread(sample["img_path"])
        if img_cv2 is None:
            raise RuntimeError(f"Failed to read image: {sample['img_path']}")

        crop_dataset = ViTDetDataset(
            self.model_cfg,
            img_cv2,
            np.asarray([sample["bbox"]], dtype=np.float32),
            np.asarray([sample["right"]], dtype=np.float32),
            rescale_factor=self.rescale_factor,
        )
        item = crop_dataset[0]
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


def configure_trainable_scope(model: torch.nn.Module, scope: str) -> List[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    if scope == "camera_head":
        modules = [model.backbone.cam_emb, model.backbone.deccam, model.refine_net.dec_cam]
    elif scope == "refine_net":
        modules = [model.refine_net]
    elif scope == "full":
        modules = [model]
    else:
        raise ValueError(f"Unsupported train scope '{scope}'.")

    for module in modules:
        unfreeze_module(module)

    if hasattr(model, "discriminator"):
        freeze_module(model.discriminator)

    return [param for param in model.parameters() if param.requires_grad]


def apply_train_mode_for_scope(model: torch.nn.Module, scope: str) -> None:
    model.train()
    if scope in {"camera_head", "refine_net"}:
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
    old_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return old_load(*args, **kwargs)

    torch.load = unsafe_load
    try:
        detector = YOLO(detector_path)
    finally:
        torch.load = old_load

    return detector.to(str(device))


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

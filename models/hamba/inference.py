import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from hamba.models import load_hamba
from hamba.utils import recursive_to
from loader import make_dataloader
from utils_new import cam_crop_to_full
from vitpose_model import ViTPoseModel


DEFAULT_HAMBA_CHECKPOINT = "ckpts/hamba/checkpoints/hamba.ckpt"
DEFAULT_DETECTRON2_CHECKPOINT = "~/.cache/torchhub/detectron2/model_final_f05665.pkl"

PERSON_DET_THRESHOLD = 0.5
HAND_KEYPOINT_THRESHOLD = 0.1


@dataclass
class Runtime:
    model: Any
    model_cfg: Any
    detector: Any
    vitpose: Any
    device: torch.device


def _require_file(path_str: str, description: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at: {path}")
    return path


def _empty_hand_outputs() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0, 21, 3), dtype=np.float32),
    )


def init_runtime(device: str = "cuda") -> Runtime:
    """Initialize Hamba model, detectron2 detector, and ViTPose keypoint model."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CPU-only runtime is not supported by the current detectron2 wrapper "
            "(DefaultPredictor_Lazy force-initializes CUDA)."
        )

    _require_file(DEFAULT_HAMBA_CHECKPOINT, "Hamba checkpoint")
    model, model_cfg = load_hamba(DEFAULT_HAMBA_CHECKPOINT)
    model = model.to(torch_device).eval()

    _require_file(DEFAULT_DETECTRON2_CHECKPOINT, "Detectron2 checkpoint")
    from detectron2.config import LazyConfig
    import hamba

    from hamba.utils.utils_detectron2 import DefaultPredictor_Lazy

    cfg_path = Path(hamba.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = str(DEFAULT_DETECTRON2_CHECKPOINT)
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.1
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    detector.model.to(torch_device)

    vitpose_name = "ViTPose+-G (multi-task train, COCO)"
    vitpose_ckpt = ViTPoseModel.MODEL_DICT[vitpose_name]["model"]
    _require_file(vitpose_ckpt, "ViTPose checkpoint")
    vitpose = ViTPoseModel(torch_device)

    return Runtime(
        model=model,
        model_cfg=model_cfg,
        detector=detector,
        vitpose=vitpose,
        device=torch_device,
    )


def detect_hands_and_keypoints(runtime: Runtime, img_cv2: np.ndarray):
    """Run person detection + ViTPose and return hand boxes, handedness, and hand keypoints."""
    det_out = runtime.detector(img_cv2)
    det_instances = det_out["instances"]
    if len(det_instances) == 0:
        return _empty_hand_outputs()

    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > PERSON_DET_THRESHOLD)
    if int(valid_idx.sum().item()) == 0:
        return _empty_hand_outputs()

    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    img_rgb = img_cv2[:, :, ::-1]
    det_for_pose = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
    vitposes_out = runtime.vitpose.predict_pose(img_rgb, det_for_pose)

    bboxes = []
    is_right = []
    keypoints_2d_list = []
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        for keyp, handedness in ((left_hand_keyp, 0), (right_hand_keyp, 1)):
            valid = keyp[:, 2] > HAND_KEYPOINT_THRESHOLD
            if int(valid.sum()) <= 3:
                continue
            bbox = [
                float(keyp[valid, 0].min()),
                float(keyp[valid, 1].min()),
                float(keyp[valid, 0].max()),
                float(keyp[valid, 1].max()),
            ]
            bboxes.append(bbox)
            is_right.append(handedness)
            keypoints_2d_list.append(keyp)

    if not bboxes:
        return _empty_hand_outputs()

    return (
        np.asarray(bboxes, dtype=np.float32),
        np.asarray(is_right, dtype=np.float32),
        np.asarray(keypoints_2d_list, dtype=np.float32),
    )


def run_hamba_inference(
    runtime: Runtime,
    img_cv2: np.ndarray,
    boxes: np.ndarray,
    is_right: np.ndarray,
    keypoints_2d_arr: np.ndarray,
    rescale_factor: float,
    batch_size: int,
    out_folder=None,
    img_fn=None,
    save_mesh=True,
):
    """Run Hamba on detected hands and return per-hand output dictionaries."""
    if len(boxes) == 0:
        return []

    dataloader = make_dataloader(
        runtime.model_cfg,
        img_cv2,
        boxes,
        is_right,
        keypoints_2d_arr,
        rescale_factor=rescale_factor,
        batch_size=batch_size,
    )

    all_results = []
    save_idx = 0

    for batch in dataloader:
        if "is_valid" in batch:
            valid_mask = batch["is_valid"] != -1
            if not bool(valid_mask.any().item()):
                continue
            if not bool(valid_mask.all().item()):
                for key, value in list(batch.items()):
                    if torch.is_tensor(value) and value.shape[0] == valid_mask.shape[0]:
                        batch[key] = value[valid_mask]

        batch = recursive_to(batch, runtime.device)

        with torch.no_grad():
            out = runtime.model(batch)

        pred_cam = out["pred_cam"].clone()
        multiplier = 2 * batch["right"] - 1
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        focal_scale = runtime.model_cfg.EXTRA.FOCAL_LENGTH / runtime.model_cfg.MODEL.IMAGE_SIZE
        scaled_focal_length = focal_scale * img_size.max(dim=1).values
        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            scaled_focal_length,
        ).detach().cpu().numpy()

        batch_count = batch["img"].shape[0]
        for n in range(batch_count):
            pred_vertices = out["pred_vertices"][n].detach().cpu().numpy()
            pred_keypoints_3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()

            right_flag = float(batch["right"][n].detach().cpu().item())
            mirror_sign = 2 * right_flag - 1
            pred_vertices[:, 0] = mirror_sign * pred_vertices[:, 0]
            pred_keypoints_3d[:, 0] = mirror_sign * pred_keypoints_3d[:, 0]

            cam_t = pred_cam_t_full[n]
            focal_length = float(scaled_focal_length[n].detach().cpu().item())
            img_res = batch["img_size"][n].detach().cpu().numpy().astype(np.int32)

            result = {
                "pred_vertices": pred_vertices,
                "pred_keypoints_3d": pred_keypoints_3d,
                "pred_cam_t_full": cam_t,
                "right": right_flag,
                "focal_length": focal_length,
                "img_res": img_res,
                "verts": pred_vertices,
                "joints": pred_keypoints_3d,
                "cam_t": cam_t,
            }
            all_results.append(result)

            if save_mesh and out_folder is not None and img_fn is not None:
                os.makedirs(out_folder, exist_ok=True)
                out_path = os.path.join(out_folder, f"{img_fn}_{save_idx:03d}_{int(right_flag)}.npy")
                np.save(out_path, result)
                save_idx += 1

    return all_results

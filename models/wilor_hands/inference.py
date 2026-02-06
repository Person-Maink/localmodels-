import torch
import numpy as np
import cv2
import os
from ultralytics import YOLO
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full, Renderer

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

def setup_models(device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load WiLoR
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )

    old_load = torch.load
    def unsafe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return old_load(*args, **kwargs)
    torch.load = unsafe_load

    detector = YOLO("./pretrained_models/detector.pt")

    torch.load = old_load  # restore safe default

    model = model.to(device).eval()
    detector = detector.to(device)
    return model, model_cfg, detector


def run_wilor_inference(model, model_cfg, detector, dataloader, img_cv2, device="cpu", out_folder=None, img_fn=None, save_mesh=True):
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    all_results = []

    for batch in dataloader:
        batch = recursive_to(batch, device)

        with torch.no_grad():
            out = model(batch)

        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        pred_cam_t_full = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, scaled_focal_length
        ).detach().cpu().numpy()

        for n in range(batch["img"].shape[0]):
            verts = out["pred_vertices"][n].detach().cpu().numpy()
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()
            is_right = batch["right"][n].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
            cam_t = pred_cam_t_full[n]

            img_res = batch["img_size"][n].detach().cpu().numpy().astype(int)

            all_results.append(dict(
                verts=verts,
                joints=joints,
                cam_t=cam_t,
                right=is_right,
                focal_length=float(scaled_focal_length),  # <- store the scaled value actually used
                img_res=img_res                           # <- store render resolution used in math
            ))
            if save_mesh and out_folder:
                os.makedirs(out_folder, exist_ok=True)
                tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
                tmesh.export(os.path.join(out_folder, f"{img_fn}_{n}_{is_right}.obj"))

    return all_results


# def run_wilor_inference(model, model_cfg, detector, dataloader, img_cv2, device="cpu", out_folder=None, img_fn=None, save_mesh=True):
#     renderer = Renderer(model_cfg, faces=model.mano.faces)

#     all_results = []

#     for batch in dataloader:
#         batch = recursive_to(batch, device)

#         with torch.no_grad():
#             out = model(batch)

#         multiplier = (2 * batch['right'] - 1)
#         pred_cam = out['pred_cam']
#         pred_cam[:, 1] = multiplier * pred_cam[:, 1]
#         box_center = batch["box_center"].float()
#         box_size = batch["box_size"].float()
#         img_size = batch["img_size"].float()
#         scaled_focal_length = (
#             model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
#         )
#         pred_cam_t_full = cam_crop_to_full(
#             pred_cam, box_center, box_size, img_size, scaled_focal_length
#         ).detach().cpu().numpy()

#         camera_records = []
#         for n in range(batch["img"].shape[0]):
#             verts = out["pred_vertices"][n].detach().cpu().numpy()
#             joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()
#             is_right = batch["right"][n].cpu().numpy()
#             verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
#             joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
#             cam_t = pred_cam_t_full[n]

#             img_res = batch["img_size"][n].detach().cpu().numpy().astype(int)

#             camera_records.append({
#                 "cam_t": cam_t.astype(np.float32),                # (3,)
#                 "focal_length": float(scaled_focal_length),       # scalar
#                 "img_res": img_res.astype(np.int32),               # (H, W)
#                 "right": bool(is_right),                           # handedness
#             })

#             all_results.append(dict(
#                 verts=verts,
#                 joints=joints,
#                 cam_t=cam_t,
#                 right=is_right,
#                 focal_length=float(scaled_focal_length),  # <- store the scaled value actually used
#                 img_res=img_res                           # <- store render resolution used in math
#             ))
#             if save_mesh and out_folder:
#                 os.makedirs(out_folder, exist_ok=True)
#                 tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
#                 tmesh.export(os.path.join(out_folder, f"{img_fn}_{n}_{is_right}.obj"))
            
#             if out_folder:
#                 np.savez(
#                     os.path.join(out_folder, "camera_params.npz"),
#                     cam_t=np.stack([c["cam_t"] for c in camera_records]),
#                     focal_length=np.array([c["focal_length"] for c in camera_records]),
#                     img_res=np.stack([c["img_res"] for c in camera_records]),
#                     right=np.array([c["right"] for c in camera_records]),
#                 )


#     return all_results
import argparse
import cv2
import numpy as np
import os
from pathlib import Path

from inference import * 
from loader import * 
from stride_refine import StrideConfig, run_stride_refinement
from visualize import *

from utils_new import *

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


def _resolve_image_paths(image_folder, video_name=None):
    img_paths = load_images_from_folder(image_folder)
    if not video_name:
        return img_paths

    target_parent = f"{video_name}_frames"
    selected = [img_path for img_path in img_paths if Path(img_path).parent.name == target_parent]
    if selected:
        return selected

    available_videos = sorted(
        {
            Path(img_path).parent.name.replace("_frames", "")
            for img_path in load_images_from_folder(image_folder)
            if Path(img_path).parent.name.endswith("_frames")
        }
    )
    print(f"[ERROR] No frames found for video '{video_name}' in {image_folder}")
    if available_videos:
        print("Available videos:", ", ".join(available_videos))
    return []


def _run_wilor_pass(args):
    device = "cuda" if args.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    model, model_cfg, detector = setup_models(
        device=device,
        checkpoint_path=args.checkpoint_path,
        cfg_path=args.cfg_path,
        detector_path=args.detector_path,
    )

    img_paths = _resolve_image_paths(args.image_folder, args.video)
    if not img_paths:
        return []

    processed_videos = []

    for img_path in img_paths:
        parent_name = Path(img_path).parent.name
        if parent_name.endswith("_frames"):
            video_name = parent_name.replace("_frames", "")
        else:
            video_name = "single_images"
        processed_videos.append(video_name)

        base_out_dir = Path(args.output_folder) / video_name

        vis_dir = base_out_dir / "visualizations"
        mesh_dir = base_out_dir / "meshes"

        for d in [vis_dir, mesh_dir]:
            d.mkdir(parents=True, exist_ok=True)

        base = Path(img_path).stem
        out_vis = vis_dir / f"{base}.jpg"
        out_mesh = mesh_dir / f"{base}/"

        print(f"\nProcessing: {img_path}")
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(img_cv2, conf=0.3, verbose=False)[0]

        if len(detections) == 0:
            print("No detections, skipping.")
            continue

        bboxes = []
        is_right = []
        detection_scores = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            bboxes.append(bbox[:4].tolist())
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            detection_scores.append(float(det.boxes.conf.cpu().detach().squeeze().item()))

        dataloader = make_dataloader(model_cfg, img_cv2, np.array(bboxes), np.array(is_right), args.rescale_factor)

        frame_id = int(base.split("_")[-1]) if base.startswith("frame_") else None
        results = run_wilor_inference(model, model_cfg, detector, dataloader, img_cv2,
                                      device=device, out_folder=out_mesh, img_fn=base, save_mesh=args.save_mesh,
                                      frame_id=frame_id, detection_scores=detection_scores)


        if args.visualize and results:
            all_verts = [r["verts"] for r in results]
            all_cam_t = [r["cam_t"] for r in results]
            all_right = [r["right"] for r in results]

            focal_length = results[0]["focal_length"]
            render_res   = tuple(results[0]["img_res"])  # (W, H)

            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                # focal_length=model_cfg.EXTRA.FOCAL_LENGTH,
                focal_length=focal_length,
            )

            cam_view = render_rgba_multiple(
                all_verts, model.mano.faces, cam_t=all_cam_t, render_res=render_res, is_right=all_right, **misc_args
            )

            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            cv2.imwrite(out_vis, (255 * overlay[:, :, ::-1]).astype(np.uint8))
            print(f"Saved visualization: {out_vis}")


    if args.visualize:
        video_out_root = Path(args.output_folder) / "videos"
        single_out_root = Path(args.output_folder) / "single_images"
        video_out_root.mkdir(exist_ok=True)

        if args.video:
            subdir = Path(args.output_folder) / args.video
            if subdir.is_dir():
                video_name = f"{subdir.name}.mp4"
                output_path = video_out_root / video_name
                print(f"\n[INFO] Creating video from {subdir.name} ...")
                images_to_video(subdir / "visualizations", output_path, fps=30)
        else:
            for subdir in Path(args.output_folder).iterdir():
                if not subdir.is_dir() or subdir == video_out_root or subdir == single_out_root:
                    continue

                video_name = f"{subdir.name}.mp4"
                output_path = video_out_root / video_name
                print(f"\n[INFO] Creating video from {subdir.name} ...")
                images_to_video(subdir / "visualizations", output_path, fps=30)

    print("\nAll images processed.")
    return sorted(set(processed_videos))


def _run_stride_pass(args):
    processed_videos = [args.video] if args.video else []
    if not args.stride_from_cache:
        processed_videos = _run_wilor_pass(args)
    elif not processed_videos:
        processed_videos = [
            path.name for path in Path(args.output_folder).iterdir()
            if path.is_dir() and path.name not in {"videos", "single_images"}
        ]

    if not processed_videos:
        raise RuntimeError("No videos available for STRIDE refinement.")

    summaries = []
    for video_name in processed_videos:
        print(f"\n[INFO] Running STRIDE refinement for {video_name}")
        summaries.append(
            run_stride_refinement(
                source_root=args.output_folder,
                output_root=args.stride_output_folder,
                video_name=video_name,
                image_folder=args.image_folder,
                visualize=args.visualize,
                target_hand=args.target_hand,
                mano_model_path=args.mano_model_path,
                use_gpu=args.use_gpu,
                stride_config=StrideConfig(
                    iters=args.stride_iters,
                    lr=args.stride_lr,
                    obs_weight=args.stride_obs_weight,
                    reproj_weight=args.stride_reproj_weight,
                    shape_weight=args.stride_shape_weight,
                    cam_smooth_weight=args.stride_cam_smooth_weight,
                    pose_smooth_weight=args.stride_pose_smooth_weight,
                    joint_smooth_weight=args.stride_joint_smooth_weight,
                    anchor_weight=args.stride_anchor_weight,
                    fft_weight=args.stride_fft_weight,
                    fft_band_low_hz=args.stride_fft_band_low_hz,
                    fft_band_high_hz=args.stride_fft_band_high_hz,
                    fps=args.stride_fps,
                    pose_rank=args.stride_pose_rank,
                    cam_rank=args.stride_cam_rank,
                ),
            )
        )
    print(f"\nRefined {len(summaries)} video(s) into {args.stride_output_folder}.")


def main(args):
    if args.mode == "stride":
        _run_stride_pass(args)
        return
    _run_wilor_pass(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WiLoR inference on image folder.")
    parser.add_argument("--mode", choices=["wilor", "stride"], default="wilor", help="Inference mode to run.")
    parser.add_argument("--image_folder", type=str, default="../../data/images/", help="Folder with input images.")
    parser.add_argument("--output_folder", type=str, default="../../outputs/wilor/", help="Folder for results.")
    parser.add_argument("--stride_output_folder", type=str, default="../../outputs/stride/", help="Folder for STRIDE-refined results.")
    parser.add_argument("--rescale_factor", type=float, default=2.0, help="BBox padding scale.")
    parser.add_argument("--video", type=str, default=None, help="Video name to process (expects <video>_frames folder).")
    parser.add_argument("--target_hand", default="auto", help="Target hand for STRIDE refinement: auto, 0, or 1.")
    parser.add_argument(
        "--stride_from_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip WiLoR inference and refine an existing WiLoR cache from output_folder.",
    )
    parser.add_argument(
        "--mano_model_path",
        type=str,
        default="./mano_data",
        help="Path to MANO assets used for STRIDE reconstruction.",
    )
    parser.add_argument("--stride_iters", type=int, default=300, help="Number of STRIDE optimization steps.")
    parser.add_argument("--stride_lr", type=float, default=0.05, help="Learning rate for STRIDE optimization.")
    parser.add_argument("--stride_obs_weight", type=float, default=10.0, help="3D observation loss weight.")
    parser.add_argument("--stride_reproj_weight", type=float, default=2.0, help="2D reprojection loss weight.")
    parser.add_argument("--stride_shape_weight", type=float, default=5.0, help="Shared shape consistency loss weight.")
    parser.add_argument("--stride_cam_smooth_weight", type=float, default=25.0, help="Camera and bbox smoothness weight.")
    parser.add_argument("--stride_pose_smooth_weight", type=float, default=1.5, help="Pose smoothness weight.")
    parser.add_argument("--stride_joint_smooth_weight", type=float, default=0.5, help="Joint trajectory smoothness weight.")
    parser.add_argument("--stride_anchor_weight", type=float, default=0.5, help="Anchor-to-WiLoR pose weight.")
    parser.add_argument("--stride_fft_weight", type=float, default=0.0, help="FFT preservation loss weight.")
    parser.add_argument("--stride_fft_band_low_hz", type=float, default=None, help="Lower FFT preservation band edge.")
    parser.add_argument("--stride_fft_band_high_hz", type=float, default=None, help="Upper FFT preservation band edge.")
    parser.add_argument("--stride_fps", type=float, default=30.0, help="Sequence FPS used for FFT preservation.")
    parser.add_argument("--stride_pose_rank", type=int, default=32, help="Temporal latent rank for pose refinement.")
    parser.add_argument("--stride_cam_rank", type=int, default=16, help="Temporal latent rank for camera refinement.")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate visualization overlays.",
    )
    parser.add_argument(
        "--save_mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save mesh reconstructions (.npy).",
    )
    parser.add_argument(
        "--use_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA if available.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./pretrained_models/wilor_final.ckpt",
        help="WiLoR checkpoint to load for inference.",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="./pretrained_models/model_config.yaml",
        help="WiLoR model config to load for inference.",
    )
    parser.add_argument(
        "--detector_path",
        type=str,
        default="./pretrained_models/detector.pt",
        help="Detector checkpoint to use during inference.",
    )
    args = parser.parse_args()
    main(args)

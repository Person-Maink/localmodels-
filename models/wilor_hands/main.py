import argparse
import cv2
import numpy as np
from inference import * 
from loader import * 
from visualize import *

from utils_new import *

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


def main(args):
    device = "cuda" if args.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    model, model_cfg, detector = setup_models(
        device=device,
        checkpoint_path=args.checkpoint_path,
        cfg_path=args.cfg_path,
        detector_path=args.detector_path,
    )

    img_paths = load_images_from_folder(args.image_folder)
    if args.video:
        target_parent = f"{args.video}_frames"
        img_paths = [img_path for img_path in img_paths if Path(img_path).parent.name == target_parent]

        if not img_paths:
            available_videos = sorted(
                {
                    Path(img_path).parent.name.replace("_frames", "")
                    for img_path in load_images_from_folder(args.image_folder)
                    if Path(img_path).parent.name.endswith("_frames")
                }
            )
            print(f"[ERROR] No frames found for video '{args.video}' in {args.image_folder}")
            if available_videos:
                print("Available videos:", ", ".join(available_videos))
            return

    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]

        parent_name = Path(img_path).parent.name
        if parent_name.endswith("_frames"):
            video_name = parent_name.replace("_frames", "")
        else:
            video_name = "single_images"

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
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            bboxes.append(bbox[:4].tolist())
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())

        dataloader = make_dataloader(model_cfg, img_cv2, np.array(bboxes), np.array(is_right), args.rescale_factor)

        results = run_wilor_inference(model, model_cfg, detector, dataloader, img_cv2,
                                      device=device, out_folder=out_mesh, img_fn=base, save_mesh=args.save_mesh)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WiLoR inference on image folder.")
    parser.add_argument("--image_folder", type=str, default="../../data/images/", help="Folder with input images.")
    parser.add_argument("--output_folder", type=str, default="../../outputs/wilor/", help="Folder for results.")
    parser.add_argument("--rescale_factor", type=float, default=2.0, help="BBox padding scale.")
    parser.add_argument("--video", type=str, default=None, help="Video name to process (expects <video>_frames folder).")
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

import argparse
from pathlib import Path

import cv2
import torch

from inference import detect_hands_and_keypoints, init_runtime, run_hamba_inference
from loader import load_images_from_folder
from utils_new import overlay_rgba_on_bgr, render_rgba_multiple
from visualize import images_to_video


LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


def main(args):
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    runtime = init_runtime(device=device)

    image_folder = Path(args.image_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    img_paths = load_images_from_folder(image_folder)

    if args.video:
        target_parent = f"{args.video}_frames"
        img_paths = [img_path for img_path in img_paths if img_path.parent.name == target_parent]
        if not img_paths:
            all_images = load_images_from_folder(image_folder)
            available_videos = sorted(
                {
                    img_path.parent.name.replace("_frames", "")
                    for img_path in all_images
                    if img_path.parent.name.endswith("_frames")
                }
            )
            print(f"[ERROR] No frames found for video '{args.video}' in {image_folder}")
            if available_videos:
                print("Available videos:", ", ".join(available_videos))
            return

    for img_path in img_paths:
        print(f"\nProcessing: {img_path}")
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print("Failed to read image, skipping.")
            continue

        parent_name = img_path.parent.name
        if parent_name.endswith("_frames"):
            video_name = parent_name.replace("_frames", "")
        else:
            video_name = "single_images"

        base_out_dir = output_folder / video_name
        vis_dir = base_out_dir / "visualizations"
        mesh_dir = base_out_dir / "meshes"
        vis_dir.mkdir(parents=True, exist_ok=True)
        mesh_dir.mkdir(parents=True, exist_ok=True)

        frame_stem = img_path.stem
        out_vis = vis_dir / f"{frame_stem}.jpg"
        out_mesh = mesh_dir / frame_stem

        boxes, is_right, keypoints_2d_arr = detect_hands_and_keypoints(runtime, img_cv2)
        if len(boxes) == 0:
            print("No detections, skipping.")
            continue

        results = run_hamba_inference(
            runtime=runtime,
            img_cv2=img_cv2,
            boxes=boxes,
            is_right=is_right,
            keypoints_2d_arr=keypoints_2d_arr,
            rescale_factor=args.rescale_factor,
            batch_size=16,
            out_folder=out_mesh,
            img_fn=frame_stem,
            save_mesh=args.save_mesh,
        )

        if args.visualize and results:
            all_verts = [result["verts"] for result in results]
            all_cam_t = [result["cam_t"] for result in results]
            all_right = [result["right"] for result in results]

            focal_length = results[0]["focal_length"]
            render_res = tuple(int(v) for v in results[0]["img_res"])
            cam_view = render_rgba_multiple(
                vertices=all_verts,
                faces_in=runtime.model.mano.faces,
                cam_t=all_cam_t,
                render_res=render_res,
                focal_length=focal_length,
                is_right=all_right,
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1.0, 1.0, 1.0),
                device=runtime.device,
            )

            overlay_bgr = overlay_rgba_on_bgr(img_cv2, cam_view)
            cv2.imwrite(str(out_vis), overlay_bgr)
            print(f"Saved visualization: {out_vis}")

    if args.visualize:
        video_out_root = output_folder / "videos"
        single_out_root = output_folder / "single_images"
        video_out_root.mkdir(parents=True, exist_ok=True)

        if args.video:
            subdir = output_folder / args.video
            vis_dir = subdir / "visualizations"
            if vis_dir.is_dir():
                out_video = video_out_root / f"{subdir.name}.mp4"
                print(f"\n[INFO] Creating video from {subdir.name} ...")
                images_to_video(vis_dir, out_video, fps=30)
        else:
            for subdir in sorted(output_folder.iterdir()):
                if not subdir.is_dir():
                    continue
                if subdir == video_out_root or subdir == single_out_root:
                    continue

                vis_dir = subdir / "visualizations"
                if not vis_dir.is_dir():
                    continue

                out_video = video_out_root / f"{subdir.name}.mp4"
                print(f"\n[INFO] Creating video from {subdir.name} ...")
                images_to_video(vis_dir, out_video, fps=30)

    print("\nAll images processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hamba inference with WiLoR-style IO.")
    parser.add_argument("--image_folder", type=str, default="/scratch/mthakur/manifold/data/images/", help="Folder with input images/videos.")
    parser.add_argument("--output_folder", type=str, default="/scratch/mthakur/manifold/outputs/hamba/", help="Folder for results.")
    parser.add_argument("--rescale_factor", type=float, default=2.0, help="BBox padding scale.")
    parser.add_argument("--video", type=str, default=None, help="Video name to process (<video>_frames).")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate visualization overlays.",
    )
    parser.add_argument(
        "--save_mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-hand .npy outputs.",
    )
    parser.add_argument(
        "--use_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA if available.",
    )
    main(parser.parse_args())

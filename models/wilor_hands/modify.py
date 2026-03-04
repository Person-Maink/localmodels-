import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from visualize import images_to_video
from utils_new import render_rgba_multiple
from wilor.models import load_wilor


LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


def parse_frame_index(frame_name: str) -> int:
    match = re.search(r"(\d+)$", frame_name)
    if match is None:
        raise ValueError(
            f"Could not parse frame index from '{frame_name}'. "
            "Expected names like 'frame_000123'."
        )
    return int(match.group(1))


def load_vipe_pose_artifact(pose_path: Path):
    if not pose_path.exists():
        raise FileNotFoundError(f"ViPE pose file not found: {pose_path}")

    if pose_path.suffix == ".npz":
        data = np.load(pose_path)
        if "inds" not in data or "data" not in data:
            raise ValueError(f"{pose_path} must contain 'inds' and 'data'.")
        inds = np.asarray(data["inds"])
        poses = np.asarray(data["data"])
    elif pose_path.suffix == ".npy":
        raise ValueError(f"Unsupported pose file type '{pose_path.suffix}'. Use or .npy. (Why are you using this bro, the output was different only!)")
        obj = np.load(pose_path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == ():
            obj = obj.item()
        if not isinstance(obj, dict):
            raise ValueError(f"{pose_path} must be a dict-like .npy with keys 'inds' and 'data'.")
        if "inds" not in obj or "data" not in obj:
            raise ValueError(f"{pose_path} must contain keys 'inds' and 'data'.")
        inds = np.asarray(obj["inds"])
        poses = np.asarray(obj["data"])
    else:
        raise ValueError(f"Unsupported pose file type '{pose_path.suffix}'. Use .npz or .npy.")

    if inds.ndim != 1:
        raise ValueError(f"Expected 'inds' to be 1D, got shape {inds.shape} in {pose_path}.")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected 'data' to have shape (N,4,4), got {poses.shape} in {pose_path}."
        )
    if len(inds) != len(poses):
        raise ValueError(
            f"Length mismatch in {pose_path}: len(inds)={len(inds)} vs len(data)={len(poses)}."
        )
    if len(inds) == 0:
        raise ValueError(f"No poses found in {pose_path}.")

    inds = inds.astype(np.int64)
    order = np.argsort(inds)
    inds = inds[order]
    poses = poses[order].astype(np.float32)
    return inds, poses


def pick_pose_for_frame(frame_idx: int, pose_inds: np.ndarray, poses_c2w: np.ndarray):
    pos = np.searchsorted(pose_inds, frame_idx, side="right") - 1
    if pos >= 0:
        return poses_c2w[pos], int(pose_inds[pos]), frame_idx != int(pose_inds[pos])

    # No prior pose exists. Use the first available pose.
    return poses_c2w[0], int(pose_inds[0]), True


def transform_raw_verts_with_pose(raw_verts: np.ndarray, pose_c2w: np.ndarray):
    w2c = np.linalg.inv(pose_c2w)
    rot = w2c[:3, :3]
    trans = w2c[:3, 3]
    verts_new = (rot @ raw_verts.T).T
    return verts_new.astype(np.float32), trans.astype(np.float32)


def load_mesh_payload(mesh_file: Path):
    payload = np.load(mesh_file, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()

    if not isinstance(payload, dict):
        raise ValueError(f"Mesh file is not a dict payload: {mesh_file}")
    if "verts" not in payload:
        raise ValueError(f"Missing 'verts' key in mesh file: {mesh_file}")

    verts = np.asarray(payload["verts"], dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"'verts' must be shaped (V,3) in {mesh_file}, got {verts.shape}")

    right = payload.get("right", 1)
    right = int(round(float(np.asarray(right).squeeze())))
    return payload, verts, right


def build_frame_image_map(image_folder: Path, video_name: str):
    frame_dir = image_folder / f"{video_name}_frames"
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame folder not found: {frame_dir}")

    image_map = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in frame_dir.glob(pattern):
            image_map[img_path.stem] = img_path

    if not image_map:
        raise FileNotFoundError(f"No frame images found in {frame_dir}")
    return image_map


def main(args):
    if not args.video:
        raise ValueError("--video is required for modify.py")

    source_video_dir = Path(args.output_folder) / args.video
    source_mesh_root = source_video_dir / "meshes"
    if not source_mesh_root.is_dir():
        raise FileNotFoundError(f"Source mesh directory not found: {source_mesh_root}")

    modified_video_name = f"{args.video}_modified"
    modified_base_dir = Path(args.output_folder) / modified_video_name
    modified_vis_dir = modified_base_dir / "visualizations"
    modified_mesh_root = modified_base_dir / "meshes"
    modified_vis_dir.mkdir(parents=True, exist_ok=True)
    modified_mesh_root.mkdir(parents=True, exist_ok=True)

    pose_path = Path(args.vipe_pose_path) if args.vipe_pose_path else Path(args.output_folder) /".." / "vipe" / "pose" / f"{args.video}.npz"
    pose_inds, poses_c2w = load_vipe_pose_artifact(pose_path)
    image_map = build_frame_image_map(Path(args.image_folder), args.video)

    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    faces = model.mano.faces

    frame_dirs = sorted([p for p in source_mesh_root.iterdir() if p.is_dir()])
    if not frame_dirs:
        raise FileNotFoundError(f"No frame mesh directories found in: {source_mesh_root}")

    vis_written = 0
    mesh_written = 0

    for frame_dir in frame_dirs:
        frame_name = frame_dir.name
        frame_idx = parse_frame_index(frame_name)
        pose_c2w, pose_frame_idx, used_fallback = pick_pose_for_frame(frame_idx, pose_inds, poses_c2w)

        if used_fallback:
            print(
                f"[WARN] No exact pose for frame {frame_name} ({frame_idx}). "
                f"Using pose from frame index {pose_frame_idx}."
            )

        mesh_files = sorted(frame_dir.glob("*.npy"))
        if not mesh_files:
            print(f"[WARN] No mesh files in {frame_dir}, skipping frame.")
            continue

        out_frame_mesh_dir = modified_mesh_root / frame_name
        out_frame_mesh_dir.mkdir(parents=True, exist_ok=True)

        all_verts = []
        all_cam_t = []
        all_right = []

        for mesh_file in mesh_files:
            payload, verts_raw, right = load_mesh_payload(mesh_file)
            verts_new, cam_t_new = transform_raw_verts_with_pose(verts_raw, pose_c2w)

            out_payload = dict(payload)
            out_payload["verts"] = verts_new
            out_payload["cam_t"] = cam_t_new
            out_payload["right"] = right
            out_payload["pose_frame_idx"] = pose_frame_idx

            np.save(out_frame_mesh_dir / mesh_file.name, out_payload)
            mesh_written += 1

            all_verts.append(verts_new)
            all_cam_t.append(cam_t_new)
            all_right.append(right)

        if args.visualize:
            frame_img_path = image_map.get(frame_name)
            if frame_img_path is None:
                raise FileNotFoundError(
                    f"No source frame image found for '{frame_name}' in {Path(args.image_folder) / (args.video + '_frames')}"
                )

            img_cv2 = cv2.imread(str(frame_img_path))
            if img_cv2 is None:
                raise RuntimeError(f"Failed to read image: {frame_img_path}")

            height, width = img_cv2.shape[:2]
            render_res = (width, height)
            focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(width, height)

            cam_view = render_rgba_multiple(
                all_verts,
                faces,
                cam_t=all_cam_t,
                render_res=render_res,
                is_right=all_right,
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=float(focal_length),
            )

            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            out_vis_path = modified_vis_dir / f"{frame_name}.jpg"
            cv2.imwrite(str(out_vis_path), (255 * overlay[:, :, ::-1]).astype(np.uint8))
            vis_written += 1

    if args.visualize and vis_written > 0:
        videos_dir = Path(args.output_folder) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_path = videos_dir / f"{modified_video_name}.mp4"
        images_to_video(modified_vis_dir, video_path, fps=30)
        print(f"[INFO] Saved modified video: {video_path}")

    print(f"[INFO] Done. Mesh files written: {mesh_written}, visualizations written: {vis_written}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify WiLoR mesh visualizations using ViPE camera poses.")
    parser.add_argument("--image_folder", type=str, default="../../data/images/", help="Folder with input images.")
    parser.add_argument("--output_folder", type=str, default="../../outputs/wilor/", help="Folder for results.")
    parser.add_argument("--rescale_factor", type=float, default=2.0, help="Unused here, kept for compatibility.")
    parser.add_argument("--video", type=str, default=None, help="Video name to process (expects <video>_frames folder).")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate visualization overlays.")
    parser.add_argument("--save_mesh", action="store_true", default=True, help="Save modified mesh reconstructions (.npy).")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Kept for compatibility.")
    parser.add_argument(
        "--vipe_pose_path",
        type=str,
        default=None,
        help="Path to ViPE pose artifact (.npz/.npy). Defaults to <output_folder>/pose/<video>.npz",
    )
    main(parser.parse_args())

import argparse
import pickle
import re
from pathlib import Path

import numpy as np

from _path_setup import PROJECT_ROOT  # ensures root imports work
from FILENAME import *

try:
    from vedo import Lines, Plotter, Sphere
except ModuleNotFoundError:
    Lines = Plotter = Sphere = None

VIPE_POSE_FILE_CFG = None
PRIMARY_MODEL_FRAMES_ROOT_CFG = None
ALT_MODEL_FRAMES_ROOT_CFG = None

VIPE_POSE_FILE_CFG = VIPE_POSE_FILE
PRIMARY_MODEL_FRAMES_ROOT_CFG = MODEL_ROOT
if PRIMARY_MODEL_FRAMES_ROOT_CFG is None:
    PRIMARY_MODEL_FRAMES_ROOT_CFG = WILOR_ROOT
ALT_MODEL_FRAMES_ROOT_CFG = HAMBA_ROOT

def _normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return Path(text)


DEFAULT_VIPE_POSE_FILE = _normalize_optional_path(VIPE_POSE_FILE_CFG)
DEFAULT_MODEL_FRAMES_ROOT = _normalize_optional_path(PRIMARY_MODEL_FRAMES_ROOT_CFG)
if DEFAULT_MODEL_FRAMES_ROOT is None:
    DEFAULT_MODEL_FRAMES_ROOT = _normalize_optional_path(ALT_MODEL_FRAMES_ROOT_CFG)


def extract_camera_params_from_results(results):
    cam_t_list = []
    right_list = []
    focal_list = []
    img_res_list = []

    for i, item in enumerate(results):
        if "cam_t" not in item:
            raise KeyError(f"Missing 'cam_t' in results[{i}]")

        cam_t_list.append(np.asarray(item["cam_t"], dtype=np.float32).reshape(3))
        right_list.append(int(np.asarray(item.get("right", -1)).item()))
        focal_list.append(float(np.asarray(item.get("focal_length", np.nan)).item()))
        img_res_list.append(np.asarray(item.get("img_res", [-1, -1]), dtype=np.int32).reshape(2))

    return {
        "cam_t": np.stack(cam_t_list, axis=0),
        "right": np.asarray(right_list, dtype=np.int32),
        "focal_length": np.asarray(focal_list, dtype=np.float32),
        "img_res": np.stack(img_res_list, axis=0),
    }


def save_camera_params_npy(results, out_npy):
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    cam_data = extract_camera_params_from_results(results)
    np.save(out_npy, cam_data, allow_pickle=True)
    return out_npy


def load_results_file(results_file):
    path = Path(results_file)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            data = arr.item() if arr.shape == () else arr.tolist()
        else:
            raise ValueError("Expected object array in .npy results file.")
    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError("Unsupported results file format. Use .npy or .pkl/.pickle")

    if not isinstance(data, list):
        raise ValueError("Loaded results must be a list of dictionaries.")
    return data


def _normalize_cam_t_shape(cam_t, source_name):
    cam_t = np.asarray(cam_t, dtype=np.float32)
    if cam_t.ndim == 1 and cam_t.shape[0] == 3:
        return cam_t.reshape(1, 3)
    if cam_t.ndim == 2 and cam_t.shape[1] == 3:
        if cam_t.shape[0] > 10:
            raise ValueError(
                f"{source_name} shape {cam_t.shape} looks like vertices, not camera translations."
            )
        return cam_t
    raise ValueError(f"Invalid cam_t shape in {source_name}: {cam_t.shape}")


def _parse_right_from_filename(path):
    tokens = path.stem.split("_")
    for tok in reversed(tokens):
        if tok in {"1", "1.0"}:
            return 1
        if tok in {"0", "0.0"}:
            return 0
    return None


def load_camera_file(camera_file):
    camera_file = Path(camera_file)
    data = np.load(camera_file, allow_pickle=True)

    if isinstance(data, np.lib.npyio.NpzFile):
        if "cam_t" not in data:
            data.close()
            raise KeyError(f"'cam_t' not found in {camera_file}")
        cam_t = _normalize_cam_t_shape(data["cam_t"], camera_file)
        right = np.asarray(data["right"], dtype=np.int32) if "right" in data else None
        data.close()
        return cam_t, right

    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        obj = data.item()
        if isinstance(obj, dict):
            if "cam_t" not in obj:
                raise KeyError(f"'cam_t' not found in dict stored in {camera_file}")
            cam_t = _normalize_cam_t_shape(obj["cam_t"], camera_file)
            right = np.asarray(obj["right"], dtype=np.int32) if "right" in obj else None
            return cam_t, right

    cam_t = _normalize_cam_t_shape(data, camera_file)
    right_guess = _parse_right_from_filename(camera_file)
    right = None if right_guess is None else np.full((len(cam_t),), right_guess, dtype=np.int32)
    return cam_t, right


def load_camera_poses_file(camera_poses_file):
    camera_poses_file = Path(camera_poses_file)
    data = np.load(camera_poses_file, allow_pickle=True)
    if "poses_wc" not in data:
        raise KeyError(f"'poses_wc' not found in {camera_poses_file}")

    poses_wc = np.asarray(data["poses_wc"], dtype=np.float32)
    if poses_wc.ndim != 3 or poses_wc.shape[1:] != (4, 4):
        raise ValueError(f"Invalid poses_wc shape in {camera_poses_file}: {poses_wc.shape}")

    right = np.asarray(data["right"], dtype=np.int32).reshape(-1) if "right" in data else None
    frame_id = np.asarray(data["frame_id"], dtype=np.int32).reshape(-1) if "frame_id" in data else None
    track_id = np.asarray(data["track_id"], dtype=np.int32).reshape(-1) if "track_id" in data else None
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float32) if "intrinsics" in data else None
    data.close()

    n = len(poses_wc)
    if right is not None and len(right) != n:
        raise ValueError(f"right length {len(right)} does not match poses_wc length {n} in {camera_poses_file}")
    if frame_id is not None and len(frame_id) != n:
        raise ValueError(f"frame_id length {len(frame_id)} does not match poses_wc length {n} in {camera_poses_file}")
    if track_id is not None and len(track_id) != n:
        raise ValueError(f"track_id length {len(track_id)} does not match poses_wc length {n} in {camera_poses_file}")
    if intrinsics is not None and intrinsics.ndim not in {1, 2}:
        raise ValueError(f"Invalid intrinsics shape in {camera_poses_file}: {intrinsics.shape}")

    return {
        "poses_wc": poses_wc,
        "right": right,
        "frame_id": frame_id,
        "track_id": track_id,
        "intrinsics": intrinsics,
    }


def parse_frame_index(path):
    for name in [path.name, path.parent.name]:
        m = re.match(r"frame_(\d+)$", name)
        if m:
            return int(m.group(1))
    return -1


def discover_frame_files(frames_root, frame_dirs_glob="frame_*", file_glob="*.npy"):
    root = Path(frames_root)
    if not root.exists():
        raise FileNotFoundError(f"frames_root does not exist: {root}")

    frame_dirs = [p for p in root.glob(frame_dirs_glob) if p.is_dir()]
    frame_dirs = sorted(frame_dirs, key=lambda p: (parse_frame_index(p), p.name))

    discovered = []
    if frame_dirs:
        for frame_dir in frame_dirs:
            frame_idx = parse_frame_index(frame_dir)
            for file_path in sorted(frame_dir.glob(file_glob)):
                if file_path.is_file():
                    discovered.append((frame_idx, file_path))
    else:
        for file_path in sorted(root.glob(file_glob)):
            if file_path.is_file():
                discovered.append((parse_frame_index(file_path), file_path))

    return discovered


def normalize_right(right, n):
    if right is None:
        return np.full((n,), -1, dtype=np.int32)

    arr = np.asarray(right, dtype=np.int32).reshape(-1)
    if arr.size == 1 and n > 1:
        arr = np.full((n,), int(arr.item()), dtype=np.int32)
    if arr.size != n:
        raise ValueError(f"right length {arr.size} does not match cam_t length {n}")
    return arr


def load_cameras_from_frames_root(frames_root, frame_dirs_glob="frame_*", file_glob="*.npy"):
    discovered = discover_frame_files(
        frames_root=frames_root,
        frame_dirs_glob=frame_dirs_glob,
        file_glob=file_glob,
    )

    cam_t_rows = []
    right_rows = []
    frame_rows = []
    loaded_files = 0
    skipped_files = 0

    for frame_idx, file_path in discovered:
        try:
            cam_t, right = load_camera_file(file_path)
        except (KeyError, ValueError):
            skipped_files += 1
            continue

        loaded_files += 1
        right = normalize_right(right, len(cam_t))
        frame_ids = np.full((len(cam_t),), frame_idx, dtype=np.int32)

        cam_t_rows.append(cam_t)
        right_rows.append(right)
        frame_rows.append(frame_ids)

    if not cam_t_rows:
        raise ValueError(
            f"No camera entries found under {frames_root}. "
            f"Checked {len(discovered)} files, skipped {skipped_files}."
        )

    cam_t = np.concatenate(cam_t_rows, axis=0)
    right = np.concatenate(right_rows, axis=0)
    frame_id = np.concatenate(frame_rows, axis=0)
    order = np.argsort(frame_id, kind="stable")

    return {
        "cam_t": cam_t[order],
        "right": right[order],
        "frame_id": frame_id[order],
        "loaded_files": loaded_files,
        "skipped_files": skipped_files,
        "total_files": len(discovered),
    }


def read_vipe_pose_artifacts(npz_file_path):
    data = np.load(npz_file_path)
    inds = data["inds"]
    poses = data["data"]  # (T, 4, 4), camera-to-world
    return inds, poses


def cam_t_to_pose_wc(cam_t, invert_cam_t=True):
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, 3] = -cam_t if invert_cam_t else cam_t
    return T_wc


def apply_translation_transform(poses, shift, scale):
    out = poses.copy()
    out[:, :3, 3] = (out[:, :3, 3] - shift.reshape(1, 3)) * float(scale)
    return out


def make_camera_frustum(T_wc, fov_deg=60.0, aspect=16 / 9, scale=0.2, color="red"):
    cam_center = T_wc[:3, 3]
    fov = np.deg2rad(fov_deg)
    h = np.tan(fov / 2.0)
    w = h * aspect

    corners_cam = np.array(
        [[-w, -h, 1.0], [w, -h, 1.0], [w, h, 1.0], [-w, h, 1.0]], dtype=np.float32
    ) * scale

    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    corners_world = (R @ corners_cam.T).T + t

    lines = []
    for c in corners_world:
        lines.append([cam_center.reshape(3), c.reshape(3)])
    for i in range(4):
        lines.append([corners_world[i].reshape(3), corners_world[(i + 1) % 4].reshape(3)])

    return Lines(lines, c=color)


def _add_pose_track(plotter, poses, stride, frustum_scale, center_radius, fov_deg, aspect, color):
    centers = poses[:, :3, 3]

    if len(centers) > 1:
        segments = [[centers[i], centers[i + 1]] for i in range(len(centers) - 1)]
        plotter += Lines(segments, c=color, lw=2)

    for i in range(0, len(poses), max(1, int(stride))):
        plotter += make_camera_frustum(
            poses[i],
            fov_deg=fov_deg,
            aspect=aspect,
            scale=frustum_scale,
            color=color,
        )

    for c in centers:
        plotter += Sphere(c, r=center_radius, c=color)


def visualize_sources(
    vipe_poses=None,
    model_cam_t=None,
    model_poses=None,
    model_right=None,
    model_hand="all",
    model_stride=5,
    vipe_stride=5,
    invert_cam_t=True,
    fov_deg=60.0,
    aspect=16 / 9,
    model_frustum_scale=0.5,
    vipe_frustum_scale=0.5,
    center_radius=0.01,
    center_to_first_frame=True,
    model_translation_scale=1.0,
    vipe_translation_scale=1.0,
):
    if Plotter is None:
        raise ModuleNotFoundError("vedo is required for visualization. Install it with: pip install vedo")

    has_vipe = vipe_poses is not None and len(vipe_poses) > 0
    has_model_cam_t = model_cam_t is not None and len(model_cam_t) > 0
    has_model_poses = model_poses is not None and len(model_poses) > 0
    has_model = has_model_cam_t or has_model_poses

    if not has_vipe and not has_model:
        raise ValueError("Both sources are empty/None. Provide at least one source.")

    plt = Plotter(bg="white", axes=1)

    vipe_poses_local = None
    if has_vipe:
        vipe_poses_local = np.asarray(vipe_poses, dtype=np.float32)
        if vipe_poses_local.ndim != 3 or vipe_poses_local.shape[1:] != (4, 4):
            raise ValueError(f"Invalid ViPE pose shape: {vipe_poses_local.shape}, expected (N, 4, 4).")

    model_poses_local = None
    model_right_local = None
    if has_model:
        if has_model_poses:
            model_poses_local = np.asarray(model_poses, dtype=np.float32)
            if model_poses_local.ndim != 3 or model_poses_local.shape[1:] != (4, 4):
                raise ValueError(
                    f"Invalid model pose shape: {model_poses_local.shape}, expected (N, 4, 4)."
                )
        else:
            model_cam_t = np.asarray(model_cam_t, dtype=np.float32).reshape(-1, 3)
            model_poses_local = np.stack(
                [cam_t_to_pose_wc(c, invert_cam_t=invert_cam_t) for c in model_cam_t],
                axis=0,
            )

        if model_right is None:
            model_right_local = np.full((len(model_poses_local),), -1, dtype=np.int32)
        else:
            model_right_local = np.asarray(model_right, dtype=np.int32).reshape(-1)
            if len(model_right_local) != len(model_poses_local):
                raise ValueError("Length mismatch between model poses and model_right.")

    if center_to_first_frame:
        if has_vipe:
            vipe_shift = vipe_poses_local[0, :3, 3]
            vipe_poses_local = apply_translation_transform(
                vipe_poses_local,
                vipe_shift,
                vipe_translation_scale,
            )
        if has_model:
            model_shift = model_poses_local[0, :3, 3]
            model_poses_local = apply_translation_transform(
                model_poses_local,
                model_shift,
                model_translation_scale,
            )

    if has_vipe:
        _add_pose_track(
            plotter=plt,
            poses=vipe_poses_local,
            stride=vipe_stride,
            frustum_scale=vipe_frustum_scale,
            center_radius=center_radius,
            fov_deg=fov_deg,
            aspect=aspect,
            color="forestgreen",
        )

    if has_model:
        if model_hand == "right":
            keep = model_right_local == 1
        elif model_hand == "left":
            keep = model_right_local == 0
        else:
            keep = np.ones(len(model_poses_local), dtype=bool)

        if not np.any(keep):
            raise ValueError(f"No model camera entries found for hand='{model_hand}'.")

        hand_styles = {1: "crimson", 0: "royalblue", -1: "gray"}
        for hand_id in [0, 1, -1]:
            idx = np.where(keep & (model_right_local == hand_id))[0]
            if len(idx) == 0:
                continue
            _add_pose_track(
                plotter=plt,
                poses=model_poses_local[idx],
                stride=model_stride,
                frustum_scale=model_frustum_scale,
                center_radius=center_radius,
                fov_deg=fov_deg,
                aspect=aspect,
                color=hand_styles[hand_id],
            )

    if has_vipe and has_model:
        print("Color key: ViPE=forestgreen, model right=crimson, model left=royalblue, unknown=gray")
    elif has_vipe:
        print("Color key: ViPE=forestgreen")
    else:
        print("Color key: model right=crimson, model left=royalblue, unknown=gray")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize ViPE and frame-camera trajectories in one scene.")
    parser.add_argument(
        "--vipe_pose_file",
        type=str,
        default=str(DEFAULT_VIPE_POSE_FILE) if DEFAULT_VIPE_POSE_FILE is not None else None,
        help="ViPE pose .npz file path. Use 'None' to disable ViPE source.",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default=str(DEFAULT_MODEL_FRAMES_ROOT) if DEFAULT_MODEL_FRAMES_ROOT is not None else None,
        help="Frame-camera folder (e.g. model meshes/frame_*/...). Use 'None' to disable this source.",
    )
    parser.add_argument(
        "--camera_poses_file",
        type=str,
        default=None,
        help="Explicit model camera-poses .npz file with poses_wc/right/frame_id arrays. Use 'None' to disable it.",
    )
    parser.add_argument("--frame_dirs_glob", type=str, default="frame_*", help="Glob for frame folders.")
    parser.add_argument(
        "--file_glob",
        type=str,
        default="*.npy",
        help="Glob for camera files inside each frame folder (e.g. '*cam*.npy').",
    )
    parser.add_argument("--hand", choices=["all", "right", "left"], default="all")
    parser.add_argument("--model_stride", type=int, default=5)
    parser.add_argument("--vipe_stride", type=int, default=5)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--aspect", type=float, default=16 / 9)
    parser.add_argument("--model_frustum_scale", type=float, default=0.5)
    parser.add_argument("--vipe_frustum_scale", type=float, default=0.5)
    parser.add_argument("--center_radius", type=float, default=0.01)
    parser.add_argument("--center_to_first_frame", action="store_true", default=True)
    parser.add_argument("--no_center_to_first_frame", dest="center_to_first_frame", action="store_false")
    parser.add_argument("--model_translation_scale", type=float, default=1.0)
    parser.add_argument("--vipe_translation_scale", type=float, default=1.0)
    parser.add_argument("--invert_cam_t", dest="invert_cam_t", action="store_true", default=True)
    parser.add_argument("--no_invert_cam_t", dest="invert_cam_t", action="store_false")
    args = parser.parse_args()

    vipe_pose_path = _normalize_optional_path(args.vipe_pose_file)
    model_frames_root = _normalize_optional_path(args.frames_root)
    camera_poses_path = _normalize_optional_path(args.camera_poses_file)

    if vipe_pose_path is None and model_frames_root is None and camera_poses_path is None:
        raise ValueError(
            "All configured sources are None. Set at least one of VIPE_POSE_FILE, model frame root, or camera poses file."
        )

    vipe_poses = None
    if vipe_pose_path is not None:
        _, vipe_poses = read_vipe_pose_artifacts(vipe_pose_path)
        print(f"Loaded {len(vipe_poses)} ViPE poses from: {vipe_pose_path}")
    else:
        print("ViPE source is None. Skipping ViPE visualization.")

    model_cam_t = None
    model_poses = None
    model_right = None
    if camera_poses_path is not None:
        pose_data = load_camera_poses_file(camera_poses_path)
        model_poses = pose_data["poses_wc"]
        model_right = pose_data["right"]
        print(f"Loaded {len(model_poses)} explicit model poses from: {camera_poses_path}")
    elif model_frames_root is not None:
        cam_data = load_cameras_from_frames_root(
            frames_root=model_frames_root,
            frame_dirs_glob=args.frame_dirs_glob,
            file_glob=args.file_glob,
        )
        model_cam_t = cam_data["cam_t"]
        model_right = cam_data["right"]
        print(
            f"Loaded {len(model_cam_t)} model camera entries from "
            f"{cam_data['loaded_files']}/{cam_data['total_files']} files "
            f"(skipped {cam_data['skipped_files']})."
        )
    else:
        print("Model frame-camera source is None. Skipping frame-camera visualization.")

    visualize_sources(
        vipe_poses=vipe_poses,
        model_cam_t=model_cam_t,
        model_poses=model_poses,
        model_right=model_right,
        model_hand=args.hand,
        model_stride=args.model_stride,
        vipe_stride=args.vipe_stride,
        invert_cam_t=args.invert_cam_t,
        fov_deg=args.fov_deg,
        aspect=args.aspect,
        model_frustum_scale=args.model_frustum_scale,
        vipe_frustum_scale=args.vipe_frustum_scale,
        center_radius=args.center_radius,
        center_to_first_frame=args.center_to_first_frame,
        model_translation_scale=args.model_translation_scale,
        vipe_translation_scale=args.vipe_translation_scale,
    )


if __name__ == "__main__":
    main()

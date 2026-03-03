import argparse
import pickle
import re
from pathlib import Path

import numpy as np

try:
    from vedo import Lines, Plotter, Sphere
except ModuleNotFoundError:
    Lines = Plotter = Sphere = None


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
        # Avoid misreading vertex arrays (e.g. 778x3) as camera translation arrays.
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
    """
    Load camera data from .npy or .npz.

    Supported:
    - .npz with key 'cam_t' (+ optional 'right')
    - .npy scalar object dict with key 'cam_t'
    - .npy raw array shaped (3,) or (N,3)
    """
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


def cam_t_to_pose_wc(cam_t, invert_cam_t=True):
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, 3] = -cam_t if invert_cam_t else cam_t
    return T_wc


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


def visualize_wilor_cameras(
    cam_t,
    right=None,
    hand="all",
    stride=5,
    invert_cam_t=True,
    fov_deg=60.0,
    aspect=16 / 9,
    frustum_scale=0.5,
    center_radius=0.01,
):
    if Plotter is None:
        raise ModuleNotFoundError("vedo is required for visualization. Install it with: pip install vedo")

    cam_t = np.asarray(cam_t, dtype=np.float32).reshape(-1, 3)
    T_wcs = np.stack([cam_t_to_pose_wc(c, invert_cam_t=invert_cam_t) for c in cam_t], axis=0)
    centers = T_wcs[:, :3, 3]

    if right is None:
        right = np.full((len(cam_t),), -1, dtype=np.int32)
    else:
        right = np.asarray(right, dtype=np.int32).reshape(-1)
        if len(right) != len(cam_t):
            raise ValueError("Length mismatch between cam_t and right.")

    if hand == "right":
        keep = right == 1
    elif hand == "left":
        keep = right == 0
    else:
        keep = np.ones(len(cam_t), dtype=bool)

    if not np.any(keep):
        raise ValueError(f"No camera entries found for hand='{hand}'.")

    hand_styles = {1: "crimson", 0: "royalblue", -1: "gray"}
    plt = Plotter(bg="white", axes=1)

    for hand_id in [0, 1, -1]:
        idx = np.where(keep & (right == hand_id))[0]
        if len(idx) == 0:
            continue

        color = hand_styles[hand_id]
        points = centers[idx]

        if len(points) > 1:
            segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
            plt += Lines(segments, c=color, lw=2)

        for j in idx[:: max(1, int(stride))]:
            plt += make_camera_frustum(
                T_wcs[j],
                fov_deg=fov_deg,
                aspect=aspect,
                scale=frustum_scale,
                color=color,
            )

        for p in points:
            plt += Sphere(p, r=center_radius, c=color)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Extract and visualize WiLoR camera parameters in 3D space.")
    parser.add_argument("--camera_npy", type=Path, default=None, help="Single camera file (.npy or .npz).")
    parser.add_argument(
        "--frames_root",
        type=Path,
        default="../outputs/wilor/120-2_clip_1/meshes",
        help="Folder containing frame_000xxx subfolders with per-frame camera files.",
    )
    parser.add_argument("--frame_dirs_glob", type=str, default="frame_*", help="Glob for frame folders.")
    parser.add_argument(
        "--file_glob",
        type=str,
        default="*.npy",
        help="Glob for camera files inside each frame folder (e.g. '*cam*.npy').",
    )
    parser.add_argument("--results_file", type=Path, default=None, help="run_wilor_inference results (.npy/.pkl).")
    parser.add_argument("--save_camera_npy", type=Path, default=None, help="Optional output .npy path.")
    parser.add_argument("--hand", choices=["all", "right", "left"], default="all")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--aspect", type=float, default=16 / 9)
    parser.add_argument("--frustum_scale", type=float, default=0.5)
    parser.add_argument("--center_radius", type=float, default=0.01)
    parser.add_argument("--invert_cam_t", dest="invert_cam_t", action="store_true", default=True)
    parser.add_argument("--no_invert_cam_t", dest="invert_cam_t", action="store_false")
    args = parser.parse_args()

    provided_sources = int(args.camera_npy is not None) + int(args.frames_root is not None) + int(args.results_file is not None)
    if provided_sources == 0:
        raise ValueError("Provide one input source: --camera_npy or --frames_root or --results_file.")
    if provided_sources > 1:
        raise ValueError("Provide only one input source at a time.")

    if args.frames_root is not None:
        cam_data = load_cameras_from_frames_root(
            frames_root=args.frames_root,
            frame_dirs_glob=args.frame_dirs_glob,
            file_glob=args.file_glob,
        )
        print(
            f"Loaded {len(cam_data['cam_t'])} camera entries from "
            f"{cam_data['loaded_files']}/{cam_data['total_files']} files "
            f"(skipped {cam_data['skipped_files']})."
        )
    elif args.results_file is not None:
        results = load_results_file(args.results_file)
        cam_data = extract_camera_params_from_results(results)
        if args.save_camera_npy is not None:
            out_npy = save_camera_params_npy(results, args.save_camera_npy)
            print(f"Saved camera params to: {out_npy}")
        print(f"Loaded {len(cam_data['cam_t'])} WiLoR camera entries from results file.")
    else:
        cam_t, right = load_camera_file(args.camera_npy)
        cam_data = {"cam_t": cam_t, "right": right}
        print(f"Loaded {len(cam_data['cam_t'])} WiLoR camera entries from single camera file.")

    visualize_wilor_cameras(
        cam_data["cam_t"],
        right=cam_data.get("right"),
        hand=args.hand,
        stride=args.stride,
        invert_cam_t=args.invert_cam_t,
        fov_deg=args.fov_deg,
        aspect=args.aspect,
        frustum_scale=args.frustum_scale,
        center_radius=args.center_radius,
    )


if __name__ == "__main__":
    main()

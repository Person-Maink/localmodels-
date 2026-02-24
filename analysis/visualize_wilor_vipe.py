import re
from pathlib import Path

import numpy as np
try:
    from vedo import Lines, Plotter, Sphere
except ModuleNotFoundError:
    Lines = Plotter = Sphere = None


# -----------------------------
# HARD-CODED PATHS (as requested)
# -----------------------------
VIPE_POSE_FILE = Path(
    "/home/mayank/Documents/Uni/TUD/Thesis Extra/DELFTBLUE /vipe/output/pose/120-2_clip_1.npz"
)
FRAMES_ROOT = Path(
    "/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/wilor/120-2_clip_1/meshes"
)

# -----------------------------
# Visualization config
# -----------------------------
FRUSTUM_STRIDE_VIPE = 5
FRUSTUM_STRIDE_WILOR = 5
FRUSTUM_SCALE_VIPE = 0.5
FRUSTUM_SCALE_WILOR = 0.5
FOV_DEG = 60.0
ASPECT = 16 / 9
CENTER_RADIUS = 0.01
CENTER_TO_FIRST_FRAME = True

# Optional manual scaling if one trajectory is much larger.
VIPE_TRANSLATION_SCALE = 1.0
WILOR_TRANSLATION_SCALE = 1.0


# -----------------------------
# Shared frustum helper
# -----------------------------
def make_camera_frustum(
    T_wc: np.ndarray,
    fov_deg: float = 60.0,
    aspect: float = 16 / 9,
    scale: float = 0.2,
    color="red",
):
    assert T_wc.shape == (4, 4)

    cam_center = T_wc[:3, 3]
    fov = np.deg2rad(fov_deg)
    h = np.tan(fov / 2)
    w = h * aspect

    corners_cam = np.array(
        [
            [-w, -h, 1.0],
            [w, -h, 1.0],
            [w, h, 1.0],
            [-w, h, 1.0],
        ],
        dtype=np.float32,
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


# -----------------------------
# ViPE loading
# -----------------------------
def read_vipe_pose_artifacts(npz_file_path: Path):
    data = np.load(npz_file_path)
    inds = data["inds"]
    poses = data["data"]  # (T, 4, 4), camera-to-world
    return inds, poses


# -----------------------------
# WiLoR loading from frame_XXXXXX/*.npy
# -----------------------------
def parse_frame_index(path: Path) -> int:
    for name in [path.name, path.parent.name]:
        m = re.match(r"frame_(\d+)$", name)
        if m:
            return int(m.group(1))
    return -1


def parse_wilor_camera_file(file_path: Path):
    """
    Expected file content (current pipeline):
      dict with keys: verts, cam_t, right
    """
    data = np.load(file_path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        item = data.item()
        if isinstance(item, dict) and "cam_t" in item:
            cam_t = np.asarray(item["cam_t"], dtype=np.float32).reshape(3)
            right_val = item.get("right", -1)
            right = int(float(np.asarray(right_val).item()))
            return cam_t, right

    # Fallbacks if camera is saved directly as ndarray.
    arr = np.asarray(data)
    if arr.ndim == 1 and arr.shape[0] == 3:
        return arr.astype(np.float32), -1

    return None, None


def load_wilor_frame_cameras(frames_root: Path):
    frame_dirs = [p for p in frames_root.glob("frame_*") if p.is_dir()]
    frame_dirs = sorted(frame_dirs, key=lambda p: (parse_frame_index(p), p.name))

    frame_ids = []
    cam_t_list = []
    right_list = []

    for frame_dir in frame_dirs:
        frame_idx = parse_frame_index(frame_dir)
        for file_path in sorted(frame_dir.glob("*.npy")):
            cam_t, right = parse_wilor_camera_file(file_path)
            if cam_t is None:
                continue
            cam_t_list.append(cam_t)
            right_list.append(right)
            frame_ids.append(frame_idx)

    if not cam_t_list:
        raise ValueError(
            f"No WiLoR camera entries found in {frames_root}. "
            "Expected per-frame .npy files containing dicts with key 'cam_t'."
        )

    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    cam_t = np.stack(cam_t_list, axis=0)
    right = np.asarray(right_list, dtype=np.int32)

    order = np.argsort(frame_ids, kind="stable")
    return frame_ids[order], cam_t[order], right[order]


# -----------------------------
# Camera conversion
# -----------------------------
def wilor_cam_t_to_pose_wc(cam_t: np.ndarray, invert_cam_t: bool = True) -> np.ndarray:
    """
    WiLoR projection uses points_cam = points + cam_t.
    Camera center in world is C = -cam_t (default).
    """
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, 3] = -cam_t if invert_cam_t else cam_t
    return T_wc


def apply_translation_transform(poses: np.ndarray, shift: np.ndarray, scale: float) -> np.ndarray:
    out = poses.copy()
    out[:, :3, 3] = (out[:, :3, 3] - shift.reshape(1, 3)) * float(scale)
    return out


# -----------------------------
# Main visualization
# -----------------------------
def main():
    if Plotter is None:
        raise ModuleNotFoundError("vedo is required for visualization. Install it with: pip install vedo")

    _, vipe_poses = read_vipe_pose_artifacts(VIPE_POSE_FILE)
    _, wilor_cam_t, wilor_right = load_wilor_frame_cameras(FRAMES_ROOT)

    wilor_poses = np.stack([wilor_cam_t_to_pose_wc(c, invert_cam_t=True) for c in wilor_cam_t], axis=0)

    if CENTER_TO_FIRST_FRAME:
        vipe_shift = vipe_poses[0, :3, 3]
        wilor_shift = wilor_poses[0, :3, 3]
    else:
        vipe_shift = np.zeros(3, dtype=np.float32)
        wilor_shift = np.zeros(3, dtype=np.float32)

    vipe_poses = apply_translation_transform(vipe_poses, vipe_shift, VIPE_TRANSLATION_SCALE)
    wilor_poses = apply_translation_transform(wilor_poses, wilor_shift, WILOR_TRANSLATION_SCALE)

    plt = Plotter(bg="white", axes=1)

    # ViPE (green)
    vipe_centers = vipe_poses[:, :3, 3]
    if len(vipe_centers) > 1:
        vipe_segments = [[vipe_centers[i], vipe_centers[i + 1]] for i in range(len(vipe_centers) - 1)]
        plt += Lines(vipe_segments, c="forestgreen", lw=2)

    for i in range(0, len(vipe_poses), max(1, FRUSTUM_STRIDE_VIPE)):
        plt += make_camera_frustum(
            vipe_poses[i],
            fov_deg=FOV_DEG,
            aspect=ASPECT,
            scale=FRUSTUM_SCALE_VIPE,
            color="forestgreen",
        )

    for c in vipe_centers:
        plt += Sphere(c, r=CENTER_RADIUS, c="forestgreen")

    # WiLoR (left=royalblue, right=crimson, unknown=gray)
    color_map = {0: "royalblue", 1: "crimson"}

    for hand_val in [0, 1]:
        idx = np.where(wilor_right == hand_val)[0]
        if len(idx) == 0:
            continue

        color = color_map[hand_val]
        centers = wilor_poses[idx, :3, 3]

        if len(centers) > 1:
            segments = [[centers[i], centers[i + 1]] for i in range(len(centers) - 1)]
            plt += Lines(segments, c=color, lw=2)

        for j in idx[:: max(1, FRUSTUM_STRIDE_WILOR)]:
            plt += make_camera_frustum(
                wilor_poses[j],
                fov_deg=FOV_DEG,
                aspect=ASPECT,
                scale=FRUSTUM_SCALE_WILOR,
                color=color,
            )

        for c in centers:
            plt += Sphere(c, r=CENTER_RADIUS, c=color)

    unknown_idx = np.where((wilor_right != 0) & (wilor_right != 1))[0]
    if len(unknown_idx) > 0:
        unknown_centers = wilor_poses[unknown_idx, :3, 3]
        for j in unknown_idx[:: max(1, FRUSTUM_STRIDE_WILOR)]:
            plt += make_camera_frustum(
                wilor_poses[j],
                fov_deg=FOV_DEG,
                aspect=ASPECT,
                scale=FRUSTUM_SCALE_WILOR,
                color="gray",
            )
        for c in unknown_centers:
            plt += Sphere(c, r=CENTER_RADIUS, c="gray")

    print("Color key: ViPE=forestgreen, WiLoR right=crimson, WiLoR left=royalblue")
    plt.show()


if __name__ == "__main__":
    main()

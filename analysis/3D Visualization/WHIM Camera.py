import argparse
from pathlib import Path

import numpy as np

from whim_io import (
    DEFAULT_WHIM_TEST_VIDEO_DIR,
    iter_whim_npy_paths,
    load_whim_frame_items,
    normalize_optional_path,
)

try:
    from vedo import Lines, Plotter, Sphere
except ModuleNotFoundError:
    Lines = Plotter = Sphere = None


RIGHT_COLOR = "crimson"
LEFT_COLOR = "royalblue"
UNKNOWN_COLOR = "gray"
DEFAULT_VIDEO_DIR = DEFAULT_WHIM_TEST_VIDEO_DIR


def _color_for_hand(side):
    if side == 1:
        return RIGHT_COLOR
    if side == 0:
        return LEFT_COLOR
    return UNKNOWN_COLOR

def load_whim_camera_tracks(video_dir, frame_step=1, max_frames=None):
    video_dir = Path(video_dir)
    npy_paths = iter_whim_npy_paths(video_dir, frame_step=frame_step, max_frames=max_frames)

    entries = []
    skipped_missing_trans = 0

    for npy_path in npy_paths:
        hands = load_whim_frame_items(npy_path)
        if not hands:
            continue

        frame_id = int(npy_path.stem)
        for hand in hands:
            trans = hand["trans"]
            if trans is None or np.asarray(trans).size < 3:
                skipped_missing_trans += 1
                continue
            trans = np.asarray(trans, dtype=np.float32).reshape(3)
            if not np.all(np.isfinite(trans)):
                skipped_missing_trans += 1
                continue
            entries.append(
                {
                    "frame_id": frame_id,
                    "side": int(hand["side"]),
                    "cam_t": trans,
                    "path": str(npy_path),
                }
            )

    if not entries:
        raise ValueError(
            f"No usable camera entries found under {video_dir}. "
            f"Checked {len(npy_paths)} files and skipped {skipped_missing_trans} hands without valid trans."
        )

    entries.sort(key=lambda item: (item["frame_id"], Path(item["path"]).name))
    return {
        "entries": entries,
        "total_files": len(npy_paths),
        "skipped_missing_trans": skipped_missing_trans,
        "video_dir": video_dir,
    }


def cam_t_to_pose_wc(cam_t, invert_cam_t=True):
    pose_wc = np.eye(4, dtype=np.float32)
    pose_wc[:3, 3] = -cam_t if invert_cam_t else cam_t
    return pose_wc


def apply_translation_transform(poses, shift, scale):
    out = poses.copy()
    out[:, :3, 3] = (out[:, :3, 3] - shift.reshape(1, 3)) * float(scale)
    return out


def make_camera_frustum(pose_wc, fov_deg=60.0, aspect=16 / 9, scale=0.2, color="red"):
    cam_center = pose_wc[:3, 3]
    fov = np.deg2rad(fov_deg)
    h = np.tan(fov / 2.0)
    w = h * aspect

    corners_cam = np.array(
        [[-w, -h, 1.0], [w, -h, 1.0], [w, h, 1.0], [-w, h, 1.0]], dtype=np.float32
    ) * scale

    rotation = pose_wc[:3, :3]
    translation = pose_wc[:3, 3]
    corners_world = (rotation @ corners_cam.T).T + translation

    lines = []
    for corner in corners_world:
        lines.append([cam_center.reshape(3), corner.reshape(3)])
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

    for center in centers:
        plotter += Sphere(center, r=center_radius, c=color)


def visualize_whim_camera_tracks(
    entries,
    hand="all",
    skip=10,
    stride=5,
    invert_cam_t=True,
    fov_deg=60.0,
    aspect=16 / 9,
    frustum_scale=0.5,
    center_radius=0.01,
    center_to_first_frame=True,
    translation_scale=1.0,
):
    if Plotter is None:
        raise ModuleNotFoundError("vedo is required for visualization. Install it with: pip install vedo")

    if hand == "right":
        keep = {1}
    elif hand == "left":
        keep = {0}
    else:
        keep = {0, 1, -1}

    filtered = [entry for entry in entries if entry["side"] in keep]
    filtered = filtered[:: max(1, int(skip))]
    if not filtered:
        raise ValueError(f"No WHIM camera entries found for hand='{hand}'.")

    plotter = Plotter(bg="white", axes=1)
    grouped_entries = {}
    for entry in filtered:
        grouped_entries.setdefault(entry["side"], []).append(entry)

    for side, hand_entries in grouped_entries.items():
        poses = np.stack(
            [cam_t_to_pose_wc(entry["cam_t"], invert_cam_t=invert_cam_t) for entry in hand_entries],
            axis=0,
        )
        if center_to_first_frame:
            poses = apply_translation_transform(poses, poses[0, :3, 3], translation_scale)
        _add_pose_track(
            plotter=plotter,
            poses=poses,
            stride=stride,
            frustum_scale=frustum_scale,
            center_radius=center_radius,
            fov_deg=fov_deg,
            aspect=aspect,
            color=_color_for_hand(side),
        )

    print("Color key: right=crimson, left=royalblue, unknown=gray")
    plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize WHIM camera tracks from per-frame trans values.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(DEFAULT_VIDEO_DIR),
        help="WHIM per-video annotation directory containing *.npy files.",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Load every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on loaded frames.")
    parser.add_argument("--hand", choices=["all", "right", "left"], default="all")
    parser.add_argument("--skip", type=int, default=10, help="Use every Nth camera entry for visualization.")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--aspect", type=float, default=16 / 9)
    parser.add_argument("--frustum_scale", type=float, default=0.5)
    parser.add_argument("--center_radius", type=float, default=0.01)
    parser.add_argument("--center_to_first_frame", action="store_true", default=True)
    parser.add_argument("--no_center_to_first_frame", dest="center_to_first_frame", action="store_false")
    parser.add_argument("--translation_scale", type=float, default=1.0)
    parser.add_argument("--invert_cam_t", dest="invert_cam_t", action="store_true", default=True)
    parser.add_argument("--no_invert_cam_t", dest="invert_cam_t", action="store_false")
    args = parser.parse_args()

    video_dir = normalize_optional_path(args.video_dir)
    if video_dir is None:
        raise ValueError("No WHIM video directory configured.")

    cam_data = load_whim_camera_tracks(
        video_dir=video_dir,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    print(
        f"Loaded {len(cam_data['entries'])} WHIM camera entries from {cam_data['total_files']} frame files under "
        f"{cam_data['video_dir']}. Skipped hands without valid trans: {cam_data['skipped_missing_trans']}."
    )

    visualize_whim_camera_tracks(
        entries=cam_data["entries"],
        hand=args.hand,
        skip=args.skip,
        stride=args.stride,
        invert_cam_t=args.invert_cam_t,
        fov_deg=args.fov_deg,
        aspect=args.aspect,
        frustum_scale=args.frustum_scale,
        center_radius=args.center_radius,
        center_to_first_frame=args.center_to_first_frame,
        translation_scale=args.translation_scale,
    )


if __name__ == "__main__":
    main()

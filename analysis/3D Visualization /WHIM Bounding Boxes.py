import argparse
from pathlib import Path

import numpy as np

from _path_setup import PROJECT_ROOT  # ensures root imports work

try:
    from vedo import Lines, Plotter, Sphere
except ModuleNotFoundError:
    Lines = Plotter = Sphere = None


RIGHT_COLOR = "crimson"
LEFT_COLOR = "royalblue"
UNKNOWN_COLOR = "gray"
DEFAULT_VIDEO_DIR = PROJECT_ROOT.parent / "data" / "whim" / "test" / "anno" / "NXRHcCScubA"


def _normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return Path(text)


def _to_numpy(value, dtype=np.float32):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        arr = value.numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _to_scalar_int(value, default=-1):
    if value is None:
        return int(default)
    arr = _to_numpy(value, dtype=np.float32)
    if arr.size == 0:
        return int(default)
    return int(round(float(arr.reshape(-1)[0])))


def _color_for_hand(side):
    if side == 1:
        return RIGHT_COLOR
    if side == 0:
        return LEFT_COLOR
    return UNKNOWN_COLOR


def _scene_center(box_center, depth):
    center = np.asarray(box_center, dtype=np.float32).reshape(2)
    return np.array([center[0], -center[1], float(depth)], dtype=np.float32)


def _make_box_segments(box_center, box_size, depth, box_scale=1.0):
    cx, cy = np.asarray(box_center, dtype=np.float32).reshape(2)
    half = float(box_size) * float(box_scale) / 2.0
    corners = np.array(
        [
            [cx - half, -(cy - half), float(depth)],
            [cx + half, -(cy - half), float(depth)],
            [cx + half, -(cy + half), float(depth)],
            [cx - half, -(cy + half), float(depth)],
        ],
        dtype=np.float32,
    )
    return [[corners[i], corners[(i + 1) % 4]] for i in range(4)]


def _load_whim_frame_items(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    if not (isinstance(arr, np.ndarray) and arr.dtype == object):
        raise ValueError(f"Unsupported WHIM npy format: {npy_path}")

    items = arr.tolist()
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise ValueError(f"Unsupported WHIM object payload in: {npy_path}")

    parsed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parsed.append(
            {
                "bbox": _to_numpy(item.get("bbox"), dtype=np.float32),
                "side": _to_scalar_int(item.get("side"), default=-1),
            }
        )
    return parsed


def load_whim_bbox_tracks(video_dir, frame_step=1, max_frames=None):
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"WHIM video directory does not exist: {video_dir}")

    npy_paths = sorted(video_dir.glob("*.npy"))
    if not npy_paths:
        raise RuntimeError(f"No .npy files found under: {video_dir}")

    step = max(1, int(frame_step))
    if step > 1:
        npy_paths = npy_paths[::step]
    if max_frames is not None:
        npy_paths = npy_paths[: max(1, int(max_frames))]

    entries = []
    skipped_missing_bbox = 0

    for npy_path in npy_paths:
        hands = _load_whim_frame_items(npy_path)
        if not hands:
            continue

        frame_id = int(npy_path.stem)
        for hand in hands:
            bbox = hand["bbox"]
            if bbox is None or bbox.size < 4:
                skipped_missing_bbox += 1
                continue

            x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]]
            width = x2 - x1
            height = y2 - y1
            box_size = max(width, height)
            box_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

            entries.append(
                {
                    "frame_id": frame_id,
                    "side": int(hand["side"]),
                    "box_center": box_center,
                    "box_size": float(box_size),
                    "path": str(npy_path),
                }
            )

    if not entries:
        raise ValueError(
            f"No usable bbox entries found under {video_dir}. "
            f"Checked {len(npy_paths)} files and skipped {skipped_missing_bbox} hands without bbox metadata."
        )

    entries.sort(key=lambda item: (item["frame_id"], Path(item["path"]).name))
    return {
        "entries": entries,
        "total_files": len(npy_paths),
        "skipped_missing_bbox": skipped_missing_bbox,
        "video_dir": video_dir,
    }


def visualize_bbox_tracks(entries, hand="all", skip=10, box_stride=5, center_radius=4.0, line_width=2.0, time_spacing=10.0):
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
        raise ValueError(f"No bbox entries found for hand='{hand}'.")

    plt = Plotter(bg="white", axes=1)

    grouped_entries = {}
    for entry in filtered:
        grouped_entries.setdefault(entry["side"], []).append(entry)

    grouped_centers = {}
    grouped_box_segments = {}
    stride = max(1, int(box_stride))
    first_frame = min(entry["frame_id"] for entry in filtered)
    num_frames = len({entry["frame_id"] for entry in filtered})
    box_scale = max(1.0, float(np.sqrt(num_frames)))

    for side, hand_entries in grouped_entries.items():
        grouped_centers[side] = [
            _scene_center(
                entry["box_center"],
                depth=(entry["frame_id"] - first_frame) * float(time_spacing),
            )
            for entry in hand_entries
        ]
        grouped_box_segments[side] = []

        for index, entry in enumerate(hand_entries):
            if index % stride != 0:
                continue
            depth = (entry["frame_id"] - first_frame) * float(time_spacing)
            grouped_box_segments[side].extend(
                _make_box_segments(entry["box_center"], entry["box_size"], depth, box_scale=box_scale)
            )

    for side, centers in grouped_centers.items():
        color = _color_for_hand(side)
        centers = np.asarray(centers, dtype=np.float32)

        if len(centers) > 1:
            segments = [[centers[i], centers[i + 1]] for i in range(len(centers) - 1)]
            plt += Lines(segments, c=color, lw=line_width)

        for center in centers:
            plt += Sphere(center, r=center_radius, c=color)

    for side, segments in grouped_box_segments.items():
        plt += Lines(segments, c=_color_for_hand(side), lw=line_width)

    print("Color key: right=crimson, left=royalblue, unknown=gray")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize WHIM bounding-box tracks for one video.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(DEFAULT_VIDEO_DIR),
        help="WHIM per-video annotation directory containing *.npy files.",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Load every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on loaded frames.")
    parser.add_argument("--hand", choices=["all", "right", "left"], default="all")
    parser.add_argument("--skip", type=int, default=10, help="Use every Nth bbox entry for visualization.")
    parser.add_argument("--box_stride", type=int, default=5)
    parser.add_argument("--center_radius", type=float, default=4.0)
    parser.add_argument("--line_width", type=float, default=2.0)
    parser.add_argument("--time_spacing", type=float, default=10.0)
    args = parser.parse_args()

    video_dir = _normalize_optional_path(args.video_dir)
    if video_dir is None:
        raise ValueError("No WHIM video directory configured.")

    bbox_data = load_whim_bbox_tracks(
        video_dir=video_dir,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    print(
        f"Loaded {len(bbox_data['entries'])} bbox records from {bbox_data['total_files']} frame files under "
        f"{bbox_data['video_dir']}. Skipped hands without bbox metadata: {bbox_data['skipped_missing_bbox']}."
    )

    visualize_bbox_tracks(
        entries=bbox_data["entries"],
        hand=args.hand,
        skip=args.skip,
        box_stride=args.box_stride,
        center_radius=args.center_radius,
        line_width=args.line_width,
        time_spacing=args.time_spacing,
    )


if __name__ == "__main__":
    main()

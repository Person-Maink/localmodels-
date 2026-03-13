import argparse
from pathlib import Path

import numpy as np

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from npy_io import discover_frame_files, load_wilor_record

try:
    from vedo import Lines, Plotter, Sphere
except ModuleNotFoundError:
    Lines = Plotter = Sphere = None


RIGHT_COLOR = "crimson"
LEFT_COLOR = "royalblue"
UNKNOWN_COLOR = "gray"


def _normalize_optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return Path(text)


def _resolve_default_frames_root():
    for attr_name in ("BBOX_SOURCE", "MODEL_ROOT"):
        value = getattr(CONFIG, attr_name, None)
        path = _normalize_optional_path(value)
        if path is not None:
            return path
    return None


DEFAULT_FRAMES_ROOT = _resolve_default_frames_root()


def _color_for_hand(hand_id):
    if hand_id == 1:
        return RIGHT_COLOR
    if hand_id == 0:
        return LEFT_COLOR
    return UNKNOWN_COLOR


def _scene_center(box_center, depth):
    center = np.asarray(box_center, dtype=np.float32).reshape(2)
    return np.array([center[0], -center[1], float(depth)], dtype=np.float32)


def _make_box_segments(box_center, box_size, depth):
    cx, cy = np.asarray(box_center, dtype=np.float32).reshape(2)
    half = float(box_size) / 2.0
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


def load_bbox_tracks(frames_root, frame_dirs_glob="frame_*", file_glob="*.npy"):
    discovered = discover_frame_files(
        frames_root=frames_root,
        frame_dirs_glob=frame_dirs_glob,
        file_glob=file_glob,
    )

    entries = []
    loaded_files = 0
    skipped_files = 0
    skipped_missing_bbox = 0

    for frame_idx, file_path in discovered:
        try:
            record = load_wilor_record(file_path)
        except Exception:
            skipped_files += 1
            continue

        loaded_files += 1

        if record["box_center"] is None or record["box_size"] is None:
            skipped_missing_bbox += 1
            continue

        entries.append(
            {
                "frame_id": int(frame_idx),
                "right": int(record["right"]),
                "box_center": np.asarray(record["box_center"], dtype=np.float32).reshape(2),
                "box_size": float(record["box_size"]),
                "path": str(file_path),
            }
        )

    if not entries:
        raise ValueError(
            f"No usable bbox entries found under {frames_root}. "
            f"Checked {len(discovered)} files, loaded {loaded_files}, "
            f"skipped {skipped_files} invalid files, skipped {skipped_missing_bbox} files without bbox metadata."
        )

    entries.sort(key=lambda item: (item["frame_id"], Path(item["path"]).name))
    return {
        "entries": entries,
        "loaded_files": loaded_files,
        "skipped_files": skipped_files,
        "skipped_missing_bbox": skipped_missing_bbox,
        "total_files": len(discovered),
    }


def visualize_bbox_tracks(entries, hand="all", box_stride=5, center_radius=4.0, line_width=2.0, time_spacing=1.0):
    if Plotter is None:
        raise ModuleNotFoundError("vedo is required for visualization. Install it with: pip install vedo")

    if hand == "right":
        keep = {1}
    elif hand == "left":
        keep = {0}
    else:
        keep = {0, 1, -1}

    filtered = [entry for entry in entries if entry["right"] in keep]
    if not filtered:
        raise ValueError(f"No bbox entries found for hand='{hand}'.")

    plt = Plotter(bg="white", axes=1)

    grouped_entries = {}
    for entry in filtered:
        grouped_entries.setdefault(entry["right"], []).append(entry)

    grouped_centers = {}
    grouped_box_segments = {}
    stride = max(1, int(box_stride))
    first_frame = min(entry["frame_id"] for entry in filtered)

    for hand_id, hand_entries in grouped_entries.items():
        grouped_centers[hand_id] = [
            _scene_center(
                entry["box_center"],
                depth=(entry["frame_id"] - first_frame) * float(time_spacing),
            )
            for entry in hand_entries
        ]
        grouped_box_segments[hand_id] = []

        for index, entry in enumerate(hand_entries):
            if index % stride != 0:
                continue
            depth = (entry["frame_id"] - first_frame) * float(time_spacing)
            grouped_box_segments[hand_id].extend(
                _make_box_segments(entry["box_center"], entry["box_size"], depth)
            )

    for hand_id, centers in grouped_centers.items():
        color = _color_for_hand(hand_id)
        centers = np.asarray(centers, dtype=np.float32)

        if len(centers) > 1:
            segments = [[centers[i], centers[i + 1]] for i in range(len(centers) - 1)]
            plt += Lines(segments, c=color, lw=line_width)

        for center in centers:
            plt += Sphere(center, r=center_radius, c=color)

    for hand_id, segments in grouped_box_segments.items():
        plt += Lines(segments, c=_color_for_hand(hand_id), lw=line_width)

    print("Color key: right=crimson, left=royalblue, unknown=gray")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize WiLoR bounding-box tracks from per-frame npy outputs.")
    parser.add_argument(
        "--frames_root",
        type=str,
        default=str(DEFAULT_FRAMES_ROOT) if DEFAULT_FRAMES_ROOT is not None else None,
        help="Frame bbox folder (e.g. model meshes/frame_*/...).",
    )
    parser.add_argument("--frame_dirs_glob", type=str, default="frame_*", help="Glob for frame folders.")
    parser.add_argument(
        "--file_glob",
        type=str,
        default="*.npy",
        help="Glob for bbox files inside each frame folder.",
    )
    parser.add_argument("--hand", choices=["all", "right", "left"], default="all")
    parser.add_argument("--box_stride", type=int, default=5)
    parser.add_argument("--center_radius", type=float, default=4.0)
    parser.add_argument("--line_width", type=float, default=2.0)
    parser.add_argument("--time_spacing", type=float, default=10.0)
    args = parser.parse_args()

    frames_root = _normalize_optional_path(args.frames_root)
    if frames_root is None:
        raise ValueError("No bbox source configured. Set BBOX_SOURCE/MODEL_ROOT/WILOR_ROOT in FILENAME.py or pass --frames_root.")

    bbox_data = load_bbox_tracks(
        frames_root=frames_root,
        frame_dirs_glob=args.frame_dirs_glob,
        file_glob=args.file_glob,
    )
    print(
        f"Loaded {bbox_data['loaded_files']} files from {bbox_data['total_files']} discovered files. "
        f"Usable bbox records: {len(bbox_data['entries'])}. "
        f"Skipped invalid files: {bbox_data['skipped_files']}. "
        f"Skipped missing bbox metadata: {bbox_data['skipped_missing_bbox']}."
    )

    visualize_bbox_tracks(
        entries=bbox_data["entries"],
        hand=args.hand,
        box_stride=args.box_stride,
        center_radius=args.center_radius,
        line_width=args.line_width,
        time_spacing=args.time_spacing,
    )


if __name__ == "__main__":
    main()

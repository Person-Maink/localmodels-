import argparse
from collections import deque
from functools import lru_cache
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from mano_pickle import load_mano_pickle
from npy_io import discover_frame_files, resolve_model_record_root


# NumPy legacy aliases for old pickle compatibility.
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.unicode = str
np.nan = float("nan")
np.inf = float("inf")


FPS = 30.0
LOWPASS_CUTOFF = 6.0
FILTER_ORDER = 3
LEFT_COLOR = "royalblue"
RIGHT_COLOR = "crimson"
MESH_ALPHA = 0.60
_MISSING_GLOBAL_ORIENT_WARNED = False


@lru_cache(maxsize=1)
def _load_mano_assets(mano_right_path):
    mano = load_mano_pickle(mano_right_path)
    j_reg = mano["J_regressor"]
    faces = np.asarray(mano["f"], dtype=np.int32)
    return j_reg, faces


def _pair_tuple(pair):
    if len(pair) != 2:
        raise ValueError(f"Pair must contain exactly two indices, got: {pair}")
    return (int(pair[0]), int(pair[1]))


def _dedupe_pairs(pairs):
    seen = set()
    resolved = []
    for pair in pairs:
        item = _pair_tuple(pair)
        if item in seen:
            continue
        seen.add(item)
        resolved.append(item)
    return resolved


def _default_mano_pairs():
    raw = getattr(CONFIG, "MULTI_POINT_MANO_PAIRS", None)
    if raw:
        return [_pair_tuple(pair) for pair in raw]
    return [(
        int(CONFIG.MODEL_SPECIFIC_VERTEX_A),
        int(CONFIG.MODEL_SPECIFIC_VERTEX_B),
    )]


def _pair_label(pair):
    return f"v{pair[0]}-v{pair[1]}"


def _axis_angle_to_matrix(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _load_raw_record(path):
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    if hasattr(payload, "item"):
        return payload.item()
    raise ValueError(f"Unsupported WiLoR record format: {path}")


def _normalize_global_orient(record, path):
    global _MISSING_GLOBAL_ORIENT_WARNED
    pred_mano_params = record.get("pred_mano_params", {})
    if "global_orient" not in pred_mano_params:
        if not _MISSING_GLOBAL_ORIENT_WARNED:
            print(
                "Warning: some WiLoR records do not contain pred_mano_params/global_orient; "
                "skipping global-orient removal for those files."
            )
            _MISSING_GLOBAL_ORIENT_WARNED = True
        return np.eye(3, dtype=np.float32)
    global_orient = np.asarray(pred_mano_params.get("global_orient"), dtype=np.float32)
    if global_orient.shape == (1, 3, 3):
        return global_orient[0]
    if global_orient.shape == (3, 3):
        return global_orient
    if global_orient.shape == (1, 3):
        return _axis_angle_to_matrix(global_orient[0])
    if global_orient.shape == (3,):
        return _axis_angle_to_matrix(global_orient)
    raise ValueError(f"Unsupported global_orient shape in {path}: {global_orient.shape}")


def _build_vertex_adjacency(total_verts, tri_faces):
    adjacency = [set() for _ in range(total_verts)]
    for tri in tri_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adjacency[a].add(b)
        adjacency[a].add(c)
        adjacency[b].add(a)
        adjacency[b].add(c)
        adjacency[c].add(a)
        adjacency[c].add(b)
    return [sorted(list(nei)) for nei in adjacency]


def _select_graph_neighbors(seed, adjacency, n_neighbors):
    visited = {seed}
    queue = deque([(seed, 0)])
    ranked = []

    while queue:
        node, dist = queue.popleft()
        for nb in adjacency[node]:
            if nb in visited:
                continue
            visited.add(nb)
            next_dist = dist + 1
            ranked.append((next_dist, nb))
            queue.append((nb, next_dist))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [vertex_id for _, vertex_id in ranked[:n_neighbors]]


def _build_region_indices(seed, adjacency, n_neighbors):
    selected = _select_graph_neighbors(seed, adjacency, n_neighbors)
    if len(selected) < n_neighbors:
        print(
            f"[warn] seed={seed}: requested {n_neighbors} neighbors, got {len(selected)} available"
        )
    return np.asarray([seed] + selected, dtype=np.int32)


def _lowpass_filter(signal, fs, cutoff, order):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal, axis=0)


def _finish_analysis(trajectory):
    trajectory = np.stack(trajectory, axis=0)
    filtered = _lowpass_filter(
        trajectory,
        fs=FPS,
        cutoff=LOWPASS_CUTOFF,
        order=FILTER_ORDER,
    )

    magnitude = np.linalg.norm(filtered, axis=1)
    magnitude -= magnitude.mean()

    freqs, psd = welch(
        magnitude,
        fs=FPS,
        nperseg=min(256, len(magnitude)),
    )

    dominant_freq = float(freqs[np.argmax(psd)])
    rms_amplitude = float(np.sqrt(np.mean(magnitude**2)))

    return {
        "trajectory": trajectory,
        "filtered": filtered,
        "magnitude": magnitude,
        "freqs": freqs,
        "psd": psd,
        "dominant": dominant_freq,
        "rms": rms_amplitude,
    }


def _color_for_hand(is_right):
    return RIGHT_COLOR if int(is_right) == 1 else LEFT_COLOR


def _resolve_source(source, wilor_root, video):
    if source is not None:
        return Path(source).expanduser().resolve()

    if video is None:
        default_source = getattr(CONFIG, "WILOR_ROOT", None) or getattr(CONFIG, "MODEL_ROOT", None)
        if default_source is None:
            raise ValueError("No source resolved. Pass --source or set WILOR_ROOT in FILENAME.py.")
        return Path(default_source).expanduser().resolve()

    if wilor_root is None:
        wilor_root = Path(CONFIG.OUTPUTS_ROOT) / "wilor"
    else:
        wilor_root = Path(wilor_root).expanduser().resolve()

    candidate = wilor_root / video
    meshes_dir = candidate / "meshes"
    if meshes_dir.is_dir():
        return meshes_dir.resolve()
    raise FileNotFoundError(f"Could not find WiLoR meshes for video '{video}' under {wilor_root}")


def _resolve_record_root(source_path):
    record_root = resolve_model_record_root(source_path)
    if record_root is None:
        raise FileNotFoundError(
            "Could not find a compatible raw record root under "
            f"'{source_path}'. Expected frame records in the source path or its 'meshes' child."
        )
    return record_root


def _collect_frames(source_path, j_reg, hand_idx, wrist_joint_idx, n_verts):
    frame_map = {}
    record_root = _resolve_record_root(source_path)
    for discovered_frame_id, path in discover_frame_files(record_root, frame_dirs_glob="frame_*", file_glob="*.npy"):
        rec = _load_raw_record(path)
        frame_id = int(discovered_frame_id)
        if frame_id < 0:
            frame_id = int(np.asarray(rec.get("frame_id", 0)).reshape(()))

        verts = np.asarray(rec["verts"], dtype=np.float32)
        cam_t = np.asarray(rec.get("cam_t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        global_orient = _normalize_global_orient(rec, path)

        if verts.shape[0] != n_verts:
            raise ValueError(
                f"Vertex count mismatch in {path}: got {verts.shape[0]}, expected {n_verts}"
            )

        verts_world = verts + cam_t[None, :]
        if "joints" in rec:
            joints = np.asarray(rec["joints"], dtype=np.float32)
            joints_world = joints + cam_t[None, :]
        else:
            joints_world = np.asarray(j_reg @ verts_world, dtype=np.float32)
        wrist = joints_world[int(wrist_joint_idx)]
        inv_global_orient = global_orient.T
        verts_no_global = (verts_world - wrist[None, :]) @ inv_global_orient

        frame_map.setdefault(int(frame_id), []).append(
            {
                "frame_id": int(frame_id),
                "right": int(rec["right"]),
                "verts_world": verts_no_global + wrist[None, :],
                "verts_centered": verts_no_global,
            }
        )

    frames = [(frame_id, frame_map[frame_id]) for frame_id in sorted(frame_map)]

    if not frames:
        raise RuntimeError(f"No frames found under source '{source_path}'.")

    selected_count = sum(
        1 for _, frame_hands in frames for hand in frame_hands if int(hand["right"]) == int(hand_idx)
    )
    if selected_count == 0:
        raise RuntimeError(
            f"No hands with HAND_IDX={hand_idx} found under '{source_path}'."
        )

    return frames


def _analyze_frames(frames, hand_idx, region_a, region_b):
    trajectory = []
    for _, frame_hands in frames:
        selected = [
            hand["verts_centered"] for hand in frame_hands if int(hand["right"]) == int(hand_idx)
        ]
        if not selected:
            continue

        frame_diffs = []
        for verts in selected:
            centroid_a = verts[region_a].mean(axis=0)
            centroid_b = verts[region_b].mean(axis=0)
            frame_diffs.append(centroid_a - centroid_b)

        trajectory.append(np.mean(np.stack(frame_diffs, axis=0), axis=0))

    if not trajectory:
        raise RuntimeError("No usable frames remained after handedness filtering.")

    return _finish_analysis(trajectory)


def build_camera_space_figure(entries, source_label, figsize_inches=(14, 9), dpi=100):
    fig, axes = plt.subplots(2, 1, figsize=figsize_inches, dpi=dpi, sharex=False)

    for idx, entry in enumerate(entries):
        result = entry["result"]
        color = f"C{idx % 10}"
        t = np.arange(len(result["magnitude"])) / FPS

        axes[0].plot(
            t,
            result["magnitude"],
            color=color,
            lw=1.5,
            label=f"{source_label} {entry['pair_label']}",
        )

        axes[1].semilogy(
            result["freqs"],
            result["psd"],
            color=color,
            lw=1.5,
            label=f"{source_label} {entry['pair_label']} ({result['dominant']:.2f} Hz)",
        )
        axes[1].axvline(result["dominant"], color=color, ls=":", alpha=0.35)

    axes[0].set_title("Filtered camera-space point-to-point displacement magnitude")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Displacement magnitude")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=2, fontsize="small", frameon=True)
    axes[0].grid(True)

    axes[1].set_title("Frequency spectrum of camera-space point-to-point motion")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power spectral density")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=2, fontsize="small", frameon=True)
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def _make_frame_actor(frame_hands, faces, vedo, hand_idx):
    actors = []
    for hand in frame_hands:
        if int(hand["right"]) != int(hand_idx):
            continue
        mesh = vedo.Mesh([hand["verts_world"], faces])
        mesh.c(_color_for_hand(hand["right"]))
        mesh.alpha(MESH_ALPHA)
        mesh.lighting("default")
        actors.append(mesh)

    if not actors:
        return None
    if len(actors) == 1:
        return actors[0]
    return sum(actors[1:], actors[0])


def show_vedo_viewer(frames, faces, hand_idx):
    try:
        import vedo
        from vedo.applications import AnimationPlayer
    except ImportError as exc:
        raise ImportError(
            "This viewer needs vedo. Install it first, for example with `pip install vedo vtk`."
        ) from exc

    actor = _make_frame_actor(frames[0][1], faces, vedo, hand_idx)

    def update_scene(i):
        nonlocal actor
        if actor is not None:
            plt3d.remove(actor)

        frame_id, frame_hands = frames[i]
        actor = _make_frame_actor(frame_hands, faces, vedo, hand_idx)
        if actor is not None:
            plt3d.add(actor)

        if title_actor is not None:
            title_actor.text(f"WiLoR camera-space meshes | frame {frame_id:06d}")

        plt3d.render()

    plt3d = AnimationPlayer(update_scene, irange=[0, len(frames) - 1])
    if actor is not None:
        plt3d += actor

    title_actor = vedo.Text2D(
        f"WiLoR camera-space meshes | frame {frames[0][0]:06d}",
        pos="top-middle",
        c="black",
        s=0.9,
    )
    note_actor = vedo.Text2D(
        "camera-space only, not world-space",
        pos="bottom-left",
        c="dimgray",
        s=0.7,
    )
    plt3d += title_actor
    plt3d += note_actor
    plt3d.show(bg="white", axes=1, viewup="z")
    plt3d.close()


def run_camera_space_frequency_analysis(source_path, hand_idx, wrist_joint_idx, n_neighbors, pair_overrides=None):
    j_reg, faces = _load_mano_assets(str(CONFIG.MANO_RIGHT_PATH))
    n_verts = int(j_reg.shape[1])

    if n_neighbors <= 0:
        raise ValueError(f"n_neighbors must be > 0, got {n_neighbors}")

    mano_pairs = _dedupe_pairs(pair_overrides if pair_overrides is not None else _default_mano_pairs())
    if not mano_pairs:
        raise ValueError("No MANO pairs resolved.")

    adjacency = _build_vertex_adjacency(n_verts, faces)
    frames = _collect_frames(source_path, j_reg, hand_idx, wrist_joint_idx, n_verts)
    entries = []
    for pair in mano_pairs:
        vertex_a, vertex_b = pair
        if not (0 <= vertex_a < n_verts):
            raise ValueError(f"vertex_a={vertex_a} out of range [0, {n_verts - 1}]")
        if not (0 <= vertex_b < n_verts):
            raise ValueError(f"vertex_b={vertex_b} out of range [0, {n_verts - 1}]")
        if vertex_a == vertex_b:
            raise ValueError(f"Pair indices must differ, got {pair}")

        region_a = _build_region_indices(vertex_a, adjacency, n_neighbors)
        region_b = _build_region_indices(vertex_b, adjacency, n_neighbors)
        entries.append(
            {
                "pair": pair,
                "pair_label": _pair_label(pair),
                "region_a": region_a,
                "region_b": region_b,
                "result": _analyze_frames(frames, hand_idx, region_a, region_b),
            }
        )

    return {
        "source_path": str(source_path),
        "hand_idx": int(hand_idx),
        "wrist_joint_idx": int(wrist_joint_idx),
        "n_neighbors": int(n_neighbors),
        "mano_pairs": mano_pairs,
        "frames": frames,
        "faces": faces,
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a compatible saved model source in camera space and make the standard "
            "frequency-analysis graphs. The source is interpreted in camera space via "
            "verts_world = verts + cam_t."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Direct path to a compatible model source. This can be a mesh cache directory or a stride clip root.",
    )
    parser.add_argument("--wilor_root", type=str, default=None, help="Root containing WiLoR per-video output folders.")
    parser.add_argument("--video", type=str, default="120-2_clip_1", help="Video folder name under --wilor_root when --source is not provided.")
    # parser.add_argument("--hand_idx", type=int, default=int(CONFIG.HAND_IDX), help="1=right, 0=left.")
    parser.add_argument("--hand_idx", type=int, default=1, help="1=right, 0=left.")
    parser.add_argument("--wrist_joint_idx", type=int, default=int(CONFIG.WRIST_JOINT_IDX), help="Wrist joint index used for centering.")
    parser.add_argument("--n_neighbors", type=int, default=int(CONFIG.N_NEIGHBORS), help="Number of graph neighbors added around each seed vertex.")
    parser.add_argument("--show_3d", action="store_true", help="Open the camera-space hand meshes in vedo before showing graphs.")
    parser.add_argument("--save_png", type=str, default=None, help="Optional output .png path for the frequency figure.")
    args = parser.parse_args()

    source_path = _resolve_source(args.source, args.wilor_root, args.video)
    analysis = run_camera_space_frequency_analysis(
        source_path=source_path,
        hand_idx=args.hand_idx,
        wrist_joint_idx=args.wrist_joint_idx,
        n_neighbors=args.n_neighbors,
    )

    print(f"Using source: {analysis['source_path']}")
    print(f"Using {len(analysis['entries'])} MANO point pair(s) from FILENAME.py")
    for entry in analysis["entries"]:
        print(f"{entry['pair_label']}: region A={entry['region_a'].tolist()} | region B={entry['region_b'].tolist()}")
        print(
            f"{entry['pair_label']}: dominant={entry['result']['dominant']:.2f} Hz | "
            f"rms={entry['result']['rms']:.6f}"
        )

    if args.show_3d:
        show_vedo_viewer(analysis["frames"], analysis["faces"], analysis["hand_idx"])

    source_label = Path(analysis["source_path"]).parent.name if Path(analysis["source_path"]).name == "meshes" else Path(analysis["source_path"]).name
    fig = build_camera_space_figure(analysis["entries"], source_label=source_label)

    if args.save_png:
        out_path = Path(args.save_png).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=fig.dpi, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

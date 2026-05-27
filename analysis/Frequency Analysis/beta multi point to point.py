import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _path_setup import PROJECT_ROOT  # ensures root imports work
import FILENAME as CONFIG
from analysis_metrics import subset_neighbor_pairs


THIS_DIR = Path(__file__).resolve().parent
VARIANT_STYLES = {
    "Actual model": "-",
    "Beta average": "--",
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_beta_comparison_module():
    return _load_module("beta_comparison_module", THIS_DIR / "beta comparison.py")


def _load_multi_point_module():
    return _load_module("multi_point_module", THIS_DIR / "Multi Point to Point.py")


def _hand_name_from_idx(hand_idx):
    return "right" if int(hand_idx) == 1 else "left"


def run_beta_multi_point_analysis(config_overrides=None):
    overrides = config_overrides or {}
    beta_comparison_module = _load_beta_comparison_module()
    multi_point_module = _load_multi_point_module()

    hand_idx = int(overrides.get("hand_idx", CONFIG.HAND_IDX))
    n_neighbors = int(overrides.get("n_neighbors", CONFIG.N_NEIGHBORS))
    wrist_joint_idx = int(overrides.get("wrist_joint_idx", CONFIG.WRIST_JOINT_IDX))
    video_name = str(overrides.get("video", "me 1.mp4"))
    wilor_root = str(overrides.get("wilor_root", Path(CONFIG.OUTPUTS_ROOT) / "wilor"))
    source_path_override = overrides.get("source_path", None)
    mano_model_path = str(overrides.get("mano_model_path", CONFIG.MANO_RIGHT_PATH))
    mano_right_path = beta_comparison_module._resolve_mano_right_path(mano_model_path)

    source_path = beta_comparison_module._resolve_model_source_path(source_path_override, wilor_root, video_name)
    source_label = beta_comparison_module._source_label(source_path)

    j_reg, faces = beta_comparison_module._load_mano_assets(str(mano_right_path))
    n_verts = int(j_reg.shape[1])
    adjacency = beta_comparison_module._build_vertex_adjacency(n_verts, faces)

    extra_mano_pairs = list(overrides.get("extra_mano_pairs", []))
    mano_pairs = multi_point_module._dedupe_pairs(
        list(multi_point_module._default_mano_pairs()) + list(multi_point_module.EXTRA_MANO_PAIRS) + extra_mano_pairs
    )
    if not mano_pairs:
        raise ValueError("No MANO pairs resolved for beta multi-point analysis.")

    actual_frames, beta_average_frames, beta_average_bundle = beta_comparison_module._load_variant_frame_sets(
        str(source_path),
        mano_right_path=str(mano_right_path),
        hand_idx=hand_idx,
    )

    entries = []
    for pair in mano_pairs:
        multi_point_module._validate_mano_pair(n_verts, pair)
        region_a = beta_comparison_module._build_region_indices(pair[0], adjacency, n_neighbors)
        region_b = beta_comparison_module._build_region_indices(pair[1], adjacency, n_neighbors)
        region_vertices = np.asarray(sorted(set(region_a.tolist()) | set(region_b.tolist())), dtype=np.int32)
        coherence_pairs = subset_neighbor_pairs(region_vertices.tolist(), adjacency)
        pair_label = multi_point_module._pair_label("model", pair)

        for variant_label, frames in (
            ("Actual model", actual_frames),
            ("Beta average", beta_average_frames),
        ):
            result = beta_comparison_module._analyze_variant_frames(
                frames,
                hand_idx,
                region_a,
                region_b,
                f"{variant_label} {pair_label}",
                region_vertices=region_vertices,
                coherence_pairs=coherence_pairs,
            )
            entries.append(
                {
                    "slot": f"{variant_label}:{pair_label}",
                    "label": f"{variant_label} {pair_label}",
                    "variant": variant_label,
                    "pair": pair,
                    "pair_label": pair_label,
                    "source": str(source_path),
                    "kind": "model",
                    "result": result,
                }
            )

    return {
        "analysis": "beta_multi_point_to_point",
        "video": video_name,
        "source_path": str(source_path),
        "source_label": source_label,
        "mesh_dir": str(source_path),
        "hand_idx": hand_idx,
        "wrist_joint_idx": wrist_joint_idx,
        "n_neighbors": n_neighbors,
        "mano_pairs": mano_pairs,
        "record_root": str(beta_average_bundle["record_root"]),
        "entries": entries,
    }


def build_beta_multi_point_figure(analysis_data, figsize_inches=(14, 9), dpi=100):
    fig, axes = plt.subplots(2, 1, figsize=figsize_inches, dpi=dpi, sharex=False)

    pair_order = []
    for entry in analysis_data["entries"]:
        if entry["pair_label"] not in pair_order:
            pair_order.append(entry["pair_label"])
    color_map = {pair_label: f"C{index % 10}" for index, pair_label in enumerate(pair_order)}

    for entry in analysis_data["entries"]:
        result = entry["result"]
        color = color_map[entry["pair_label"]]
        linestyle = VARIANT_STYLES.get(entry["variant"], "-")
        label_with_peak = f"{entry['variant']} {entry['pair_label']} ({result['dominant']:.2f} Hz)"
        t = np.arange(len(result["magnitude"])) / 30.0

        axes[0].plot(
            t,
            result["magnitude"],
            color=color,
            linestyle=linestyle,
            lw=1.5,
            label=label_with_peak,
        )

        axes[1].semilogy(
            result["freqs"],
            result["psd"],
            color=color,
            linestyle=linestyle,
            lw=1.5,
            label=label_with_peak,
        )
        axes[1].axvline(result["dominant"], color=color, ls=":", alpha=0.35)

    axes[0].set_title("Filtered point-to-point displacement magnitude: actual vs beta average")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Displacement magnitude")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=2, fontsize="small", frameon=True)
    axes[0].grid(True)

    axes[1].set_title("Frequency spectrum: actual vs beta average")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power spectral density")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=2, fontsize="small", frameon=True)
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-point frequency analysis on a compatible saved model source for the raw "
            "model output and the sequence-beta-averaged reconstruction."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Direct path to a compatible model source. This can be a mesh cache directory or a stride clip root.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="clip_2.mp4",
        help="Video filename or stem used with --wilor_root when --source is not provided.",
    )
    parser.add_argument(
        "--wilor_root",
        type=str,
        default=str(Path(CONFIG.OUTPUTS_ROOT) / "wilor"),
        help="Root directory containing per-video WiLoR outputs when --source is not provided.",
    )
    parser.add_argument(
        "--mano_model_path",
        type=str,
        default=str(CONFIG.MANO_RIGHT_PATH),
        help="Path to MANO assets or directly to MANO_RIGHT.pkl for beta-averaged reconstruction.",
    )
    parser.add_argument(
        "--hand_idx",
        type=int,
        default=int(CONFIG.HAND_IDX),
        help="1=right, 0=left.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=int(CONFIG.N_NEIGHBORS),
        help="Number of graph neighbors added around each seed vertex.",
    )
    parser.add_argument(
        "--save_png",
        type=str,
        default=None,
        help="Optional output .png path for the beta multi-point figure.",
    )
    args = parser.parse_args()

    analysis_data = run_beta_multi_point_analysis(
        {
            "source_path": args.source,
            "video": args.video,
            "wilor_root": args.wilor_root,
            "mano_model_path": args.mano_model_path,
            "hand_idx": args.hand_idx,
            "n_neighbors": args.n_neighbors,
        }
    )

    print(f"Using source: {analysis_data['source_path']}")
    print(f"Using hand_idx={analysis_data['hand_idx']}")
    fig = build_beta_multi_point_figure(analysis_data)

    if args.save_png:
        out_path = Path(args.save_png).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=fig.dpi, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()
    plt.close(fig)

    for entry in analysis_data["entries"]:
        result = entry["result"]
        print(
            f"{entry['label']}: dominant={result['dominant']:.2f} Hz, "
            f"rms={result['rms']:.6f}"
        )


if __name__ == "__main__":
    main()

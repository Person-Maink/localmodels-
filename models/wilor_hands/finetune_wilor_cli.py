import argparse


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune WiLoR with teacher distillation, ViPE supervision, and temporal loss ablations."
    )
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg_path", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector_path", type=str, default="./pretrained_models/detector.pt")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="../../data/images/",
        help="Folder containing raw videos, sidecar ZIP frame caches, legacy *_frames folders, or loose images.",
    )
    parser.add_argument(
        "--frame_cache_root",
        type=str,
        default=None,
        help="Optional directory containing sidecar *.frames.zip / *.frames.index.json caches. Defaults to image_folder.",
    )
    parser.add_argument("--pose_dir", type=str, default="../../outputs/vipe/pose")
    parser.add_argument("--intrinsics_dir", type=str, default="../../outputs/vipe/intrinsics")
    parser.add_argument("--output_dir", type=str, default="./finetune_runs/distill_vipe")
    parser.add_argument("--loss_config", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--video", action="append", dest="videos", default=[])
    parser.add_argument(
        "--all_videos",
        action="store_true",
        help="Use every video that has a sidecar ZIP cache or legacy *_frames directory under image_folder.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=0,
        help="Limit the number of frames used to build the dataset.",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.15,
        help="Fraction of unique frames reserved for validation, e.g. 0.15 for 15%%.",
    )
    parser.add_argument("--detection_cache", type=str, default=None)
    parser.add_argument("--detection_conf", type=float, default=0.3)
    parser.add_argument(
        "--lazy_detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, run the detector only for frames needed by each batch instead of precomputing the whole dataset.",
    )
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument(
        "--train_scope",
        type=str,
        choices=["temporal_only", "refine_net"],
        default="refine_net",
    )
    parser.add_argument(
        "--camera_loss_weight",
        type=float,
        default=0.01,
        help="Legacy alias for the ViPE camera supervision weight.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of temporal windows per optimization step.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal_window_size", type=int, default=None)
    parser.add_argument("--temporal_window_stride", type=int, default=None)
    parser.add_argument("--temporal_max_frame_gap", type=int, default=None)
    parser.add_argument(
        "--temporal_reduction",
        type=str,
        default=None,
        choices=["l1", "l2", "smooth_l1"],
    )
    parser.add_argument("--temporal_scorer_hidden_dim", type=int, default=None)
    parser.add_argument("--temporal_scorer_layers", type=int, default=None)
    parser.add_argument("--temporal_scorer_dropout", type=float, default=None)
    parser.add_argument("--vipe_camera_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vipe_camera_weight", type=float, default=None)
    parser.add_argument("--temporal_camera_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--temporal_camera_formulation",
        type=str,
        default=None,
        choices=["static", "learnable"],
    )
    parser.add_argument("--temporal_camera_weight", type=float, default=None)
    parser.add_argument("--temporal_camera_scorer_weight", type=float, default=None)
    parser.add_argument(
        "--temporal_bbox_projected_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--temporal_bbox_projected_formulation",
        type=str,
        default=None,
        choices=["static", "learnable"],
    )
    parser.add_argument("--temporal_bbox_projected_weight", type=float, default=None)
    parser.add_argument("--temporal_bbox_projected_scorer_weight", type=float, default=None)
    parser.add_argument("--lora_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_block_start", type=int, default=None)
    parser.add_argument("--lora_block_end", type=int, default=None)
    parser.add_argument("--lora_target_modules", type=str, default=None)
    return parser

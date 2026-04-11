import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from finetune_wilor_common import (
    DetectedVideoHandDataset,
    ViPECameraIndex,
    build_detection_samples,
    choose_device,
    discover_videos,
    evaluate_distillation,
    filter_videos_with_vipe_artifacts,
    load_detector,
    seed_everything,
    set_optional_loss_weight,
    split_samples_by_frame,
)
from wilor.models import load_wilor


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a WiLoR checkpoint against distillation and ViPE camera targets."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Teacher checkpoint used to generate pseudo-labels. If omitted, try to infer it from the evaluated checkpoint.",
    )
    parser.add_argument("--cfg_path", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector_path", type=str, default="./pretrained_models/detector.pt")
    parser.add_argument("--image_folder", type=str, default="../../data/images/")
    parser.add_argument("--pose_dir", type=str, default="../../outputs/vipe/pose")
    parser.add_argument("--intrinsics_dir", type=str, default="../../outputs/vipe/intrinsics")
    parser.add_argument("--video", action="append", dest="videos", default=[])
    parser.add_argument(
        "--all_videos",
        action="store_true",
        help="Use every *_frames directory under image_folder.",
    )
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.15,
        help="Fraction of unique frames reserved for validation, e.g. 0.15 for 15%%.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default="val",
        help="Which split to evaluate after applying the frame-based validation split.",
    )
    parser.add_argument("--detection_cache", type=str, default=None)
    parser.add_argument("--detection_conf", type=float, default=0.3)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--camera_loss_weight", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", type=str, default=None)
    return parser


def infer_teacher_checkpoint(model_checkpoint: str) -> str | None:
    checkpoint_data = torch.load(model_checkpoint, map_location="cpu")
    finetune_args = checkpoint_data.get("finetune_args", {})
    teacher_checkpoint = finetune_args.get("checkpoint")
    if teacher_checkpoint:
        return str(teacher_checkpoint)
    return None


def resolve_video_names(args: argparse.Namespace) -> list[str]:
    if args.all_videos:
        video_names = discover_videos(args.image_folder)
    else:
        video_names = sorted(set(args.videos))
    if not video_names:
        raise ValueError("No videos selected. Pass --video ... or use --all_videos.")

    valid_video_names, missing_vipe_artifacts = filter_videos_with_vipe_artifacts(
        video_names,
        args.pose_dir,
        args.intrinsics_dir,
    )
    if missing_vipe_artifacts:
        if args.all_videos:
            print(
                f"Skipping {len(missing_vipe_artifacts)} discovered video(s) without matching "
                "ViPE pose/intrinsics artifacts:"
            )
            for video_name, missing_paths in sorted(missing_vipe_artifacts.items()):
                print(f"  - {video_name}: missing {', '.join(missing_paths)}")
            video_names = valid_video_names
        else:
            missing_lines = [
                f"{video_name}: missing {', '.join(missing_paths)}"
                for video_name, missing_paths in sorted(missing_vipe_artifacts.items())
            ]
            raise ValueError(
                "The following requested video(s) do not have matching ViPE pose/intrinsics artifacts:\n"
                + "\n".join(missing_lines)
            )

    if not video_names:
        raise ValueError(
            "No videos remain after filtering for matching ViPE pose/intrinsics artifacts."
        )
    return video_names


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.use_gpu)

    teacher_checkpoint = args.teacher_checkpoint or infer_teacher_checkpoint(args.checkpoint)
    if teacher_checkpoint is None:
        raise ValueError(
            "Could not infer the teacher checkpoint from the evaluated checkpoint. "
            "Pass --teacher_checkpoint explicitly."
        )

    video_names = resolve_video_names(args)

    student, _ = load_wilor(args.checkpoint, args.cfg_path)
    teacher, _ = load_wilor(teacher_checkpoint, args.cfg_path)
    set_optional_loss_weight(student.cfg, "ADVERSARIAL", 0.0)
    set_optional_loss_weight(student.cfg, "CAMERA_T_FULL", args.camera_loss_weight)

    student = student.to(device).eval()
    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    detector = load_detector(args.detector_path, device)
    camera_index = ViPECameraIndex(args.pose_dir, args.intrinsics_dir)

    detection_cache = args.detection_cache
    if detection_cache is None:
        video_slug = "all" if args.all_videos else "_".join(video_names)
        detection_cache = str(
            Path(args.checkpoint).resolve().parent / f"test_detections_{video_slug}.json"
        )

    samples = build_detection_samples(
        image_folder=args.image_folder,
        video_names=video_names,
        detector=detector,
        camera_index=camera_index,
        detection_conf=args.detection_conf,
        detection_cache_path=detection_cache,
        sample_limit=args.sample_limit,
    )
    if not samples:
        raise RuntimeError("No detector samples were generated for testing.")

    train_samples, val_samples, split_stats = split_samples_by_frame(
        samples,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    if args.split == "train":
        eval_samples = train_samples
    elif args.split == "val":
        eval_samples = val_samples
    else:
        eval_samples = list(samples)

    if not eval_samples:
        raise RuntimeError(
            f"No samples are available for split '{args.split}'. "
            "Use --split all, reduce --validation_split, or provide more frames."
        )

    dataset = DetectedVideoHandDataset(
        student.cfg,
        eval_samples,
        rescale_factor=args.rescale_factor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    metrics = evaluate_distillation(
        student,
        teacher,
        dataloader,
        device=device,
        amp=args.amp,
    )
    metrics.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "teacher_checkpoint": str(Path(teacher_checkpoint).resolve()),
            "split": args.split,
            "video_count": len(video_names),
            "frame_total": split_stats["total_frames"],
            "frame_train": split_stats["train_frames"],
            "frame_val": split_stats["val_frames"],
            "sample_total": split_stats["total_samples"],
            "sample_train": split_stats["train_samples"],
            "sample_val": split_stats["val_samples"],
        }
    )

    print(f"Device: {device}")
    print(f"Videos: {', '.join(video_names)}")
    print(
        f"Frames: total={split_stats['total_frames']} "
        f"train={split_stats['train_frames']} val={split_stats['val_frames']}"
    )
    print(
        f"Samples: total={split_stats['total_samples']} "
        f"train={split_stats['train_samples']} val={split_stats['val_samples']}"
    )
    print(
        f"Eval split={args.split} loss={metrics.get('loss_total', 0.0):.4f} "
        f"cam={metrics.get('loss_camera_t_full', 0.0):.4f} "
        f"kp2d={metrics.get('loss_keypoints_2d', 0.0):.4f} "
        f"kp3d={metrics.get('loss_keypoints_3d', 0.0):.4f}"
    )
    if "camera_target_fallback_rate" in metrics:
        print(
            f"Fallback frame rate: {metrics['camera_target_fallback_rate']:.4f}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main(make_argparser().parse_args())

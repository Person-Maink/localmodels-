from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import DataLoader

from finetune_wilor_common import (
    DetectedVideoHandDataset,
    ViPECameraIndex,
    build_detection_samples,
    discover_videos,
    filter_videos_with_vipe_artifacts,
    split_samples_by_frame,
)
from temporal_losses import (
    TemporalWindowDataset,
    build_temporal_windows,
    temporal_window_collate,
)


EMPTY_WINDOW_STATS = {
    "window_count": 0,
    "dropped_window_count": 0,
    "stream_count": 0,
    "broken_segment_count": 0,
}


@dataclass(frozen=True)
class WindowDataBundle:
    video_names: list[str]
    split_stats: dict[str, int | float]
    train_window_stats: dict[str, int | float]
    val_window_stats: dict[str, int | float]
    train_dataloader: DataLoader
    val_dataloader: DataLoader | None


def resolve_video_names(
    args,
    log_fn: Callable[[str], None],
) -> list[str]:
    if args.all_videos:
        log_fn("Discovering videos from image folder because --all_videos is enabled.")
        video_names = discover_videos(args.image_folder)
    else:
        video_names = sorted(set(args.videos))
        log_fn(
            "Using explicitly requested video selection: "
            f"{', '.join(video_names) if video_names else '<none>'}"
        )
    if not video_names:
        raise ValueError("No videos selected. Pass --video ... or use --all_videos.")

    log_fn(f"Filtering {len(video_names)} selected video(s) for matching ViPE artifacts.")
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

    log_fn("Proceeding with video(s): " + ", ".join(video_names))
    return video_names


def build_window_data(
    args,
    output_dir: Path,
    model_cfg,
    detector,
    camera_index: ViPECameraIndex,
    temporal_cfg: dict[str, Any],
    log_fn: Callable[[str], None],
) -> WindowDataBundle:
    video_names = resolve_video_names(args, log_fn)

    detection_cache = args.detection_cache
    if detection_cache is None:
        video_slug = "all" if args.all_videos else "_".join(video_names)
        detection_cache = str(output_dir / f"detections_{video_slug}.json")

    log_fn(f"Building detection samples (cache path: {detection_cache}).")
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
        raise RuntimeError("No detector samples were generated for fine-tuning.")
    log_fn(f"Collected {len(samples)} detector sample(s).")

    log_fn(f"Splitting samples into train/val with validation_split={args.validation_split}.")
    train_samples, val_samples, split_stats = split_samples_by_frame(
        samples,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    if not train_samples:
        raise RuntimeError(
            "Validation split left no training samples. Reduce --validation_split or provide more frames."
        )

    log_fn(
        f"Building temporal training windows with size={temporal_cfg['window_size']}, "
        f"stride={temporal_cfg['window_stride']}, gap={temporal_cfg['max_frame_gap']}."
    )
    train_windows, train_window_stats = build_temporal_windows(
        train_samples,
        window_size=int(temporal_cfg["window_size"]),
        window_stride=int(temporal_cfg["window_stride"]),
        max_frame_gap=int(temporal_cfg["max_frame_gap"]),
    )
    if not train_windows:
        raise RuntimeError(
            "No temporal training windows were produced. Increase sample coverage or reduce temporal window settings."
        )

    log_fn("Constructing training dataset and dataloader.")
    train_base_dataset = DetectedVideoHandDataset(
        model_cfg,
        train_samples,
        rescale_factor=args.rescale_factor,
        include_path_metadata=False,
    )
    train_dataset = TemporalWindowDataset(train_base_dataset, train_windows)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=temporal_window_collate,
    )

    val_window_stats = dict(EMPTY_WINDOW_STATS)
    val_dataloader = None
    if val_samples:
        log_fn("Building validation temporal windows and dataloader.")
        val_windows, val_window_stats = build_temporal_windows(
            val_samples,
            window_size=int(temporal_cfg["window_size"]),
            window_stride=int(temporal_cfg["window_stride"]),
            max_frame_gap=int(temporal_cfg["max_frame_gap"]),
        )
        if val_windows:
            val_base_dataset = DetectedVideoHandDataset(
                model_cfg,
                val_samples,
                rescale_factor=args.rescale_factor,
                include_path_metadata=False,
            )
            val_dataset = TemporalWindowDataset(val_base_dataset, val_windows)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                collate_fn=temporal_window_collate,
            )

    return WindowDataBundle(
        video_names=video_names,
        split_stats=split_stats,
        train_window_stats=train_window_stats,
        val_window_stats=val_window_stats,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

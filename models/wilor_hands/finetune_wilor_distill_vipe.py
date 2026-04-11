import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from finetune_wilor_common import (
    DetectedVideoHandDataset,
    ViPECameraIndex,
    append_metrics,
    apply_train_mode_for_scope,
    build_detection_samples,
    build_teacher_supervision_batch,
    choose_device,
    configure_trainable_scope,
    count_trainable_parameters,
    discover_videos,
    evaluate_distillation,
    filter_videos_with_vipe_artifacts,
    format_loss_dict,
    infinite_loader,
    load_detector,
    save_wilor_checkpoint,
    seed_everything,
    set_optional_loss_weight,
    split_samples_by_frame,
)
from wilor.models import load_wilor
from wilor.utils import recursive_to


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune WiLoR with teacher distillation and ViPE camera supervision."
    )
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg_path", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector_path", type=str, default="./pretrained_models/detector.pt")
    parser.add_argument("--image_folder", type=str, default="../../data/images/")
    parser.add_argument("--pose_dir", type=str, default="../../outputs/vipe/pose")
    parser.add_argument("--intrinsics_dir", type=str, default="../../outputs/vipe/intrinsics")
    parser.add_argument("--output_dir", type=str, default="./finetune_runs/distill_vipe")
    parser.add_argument("--video", action="append", dest="videos", default=[])
    parser.add_argument("--all_videos", action="store_true", help="Use every *_frames directory under image_folder.")
    parser.add_argument("--sample_limit", type=int, default=0, help="Limit the number of frames used to build the dataset.")
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.15,
        help="Fraction of unique frames reserved for validation, e.g. 0.15 for 15%%.",
    )
    parser.add_argument("--detection_cache", type=str, default=None)
    parser.add_argument("--detection_conf", type=float, default=0.3)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--train_scope", type=str, choices=["camera_head", "refine_net", "full"], default="refine_net")
    parser.add_argument("--camera_loss_weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.use_gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    student, _ = load_wilor(args.checkpoint, args.cfg_path)
    teacher = copy.deepcopy(student)

    set_optional_loss_weight(student.cfg, "ADVERSARIAL", 0.0)
    set_optional_loss_weight(student.cfg, "CAMERA_T_FULL", args.camera_loss_weight)

    student = student.to(device)
    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    detector = load_detector(args.detector_path, device)
    camera_index = ViPECameraIndex(args.pose_dir, args.intrinsics_dir)

    detection_cache = args.detection_cache
    if detection_cache is None:
        video_slug = "all" if args.all_videos else "_".join(video_names)
        detection_cache = str(output_dir / f"detections_{video_slug}.json")

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

    train_samples, val_samples, split_stats = split_samples_by_frame(
        samples,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    if not train_samples:
        raise RuntimeError(
            "Validation split left no training samples. Reduce --validation_split or provide more frames."
        )

    train_dataset = DetectedVideoHandDataset(
        student.cfg,
        train_samples,
        rescale_factor=args.rescale_factor,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_dataloader = None
    if val_samples:
        val_dataset = DetectedVideoHandDataset(
            student.cfg,
            val_samples,
            rescale_factor=args.rescale_factor,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    trainable_params = configure_trainable_scope(student, args.train_scope)
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters were selected for scope '{args.train_scope}'.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    batch_iter = infinite_loader(train_dataloader)
    metrics_path = output_dir / "metrics.jsonl"

    best_metric = float("inf")
    best_metric_name = "val_loss_total" if val_dataloader is not None else "train_loss_total"

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
    if args.validation_split > 0.0 and val_dataloader is None:
        print(
            "Validation split requested, but there were not enough frames to reserve a validation set."
        )
    print(f"Train scope: {args.train_scope}")
    print(f"Trainable params: {count_trainable_parameters(student):,}")

    for step in range(1, args.max_steps + 1):
        apply_train_mode_for_scope(student, args.train_scope)
        batch = recursive_to(next(batch_iter), device)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_output = teacher.forward_step(batch, train=False)
            supervision_batch = build_teacher_supervision_batch(batch, teacher_output)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            student_output = student.forward_step(batch, train=True)
            loss = student.compute_loss(supervision_batch, student_output, train=True)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_value = float(loss.detach().item())
        should_log = step % args.log_every == 0 or step == 1 or step == args.max_steps

        if val_dataloader is None and loss_value < best_metric:
            best_metric = loss_value
            save_wilor_checkpoint(
                output_dir / "best.ckpt",
                student,
                optimizer,
                step,
                epoch=0,
                extra={
                    "finetune_args": vars(args),
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                },
            )

        if should_log:
            train_metrics = format_loss_dict(student_output["losses"])
            train_metrics.update({"step": step, "split": "train", "loss_total": loss_value})
            append_metrics(metrics_path, train_metrics)
            print(
                f"[step {step:05d}] loss={loss_value:.4f} "
                f"cam={train_metrics.get('loss_camera_t_full', 0.0):.4f} "
                f"kp2d={train_metrics.get('loss_keypoints_2d', 0.0):.4f} "
                f"kp3d={train_metrics.get('loss_keypoints_3d', 0.0):.4f}"
            )
            if val_dataloader is not None:
                val_metrics = evaluate_distillation(
                    student,
                    teacher,
                    val_dataloader,
                    device=device,
                    amp=scaler.is_enabled(),
                )
                val_metrics.update({"step": step, "split": "val"})
                append_metrics(metrics_path, val_metrics)
                print(
                    f"             val_loss={val_metrics.get('loss_total', 0.0):.4f} "
                    f"val_cam={val_metrics.get('loss_camera_t_full', 0.0):.4f} "
                    f"val_kp2d={val_metrics.get('loss_keypoints_2d', 0.0):.4f} "
                    f"val_kp3d={val_metrics.get('loss_keypoints_3d', 0.0):.4f}"
                )
                if val_metrics["loss_total"] < best_metric:
                    best_metric = val_metrics["loss_total"]
                    save_wilor_checkpoint(
                        output_dir / "best.ckpt",
                        student,
                        optimizer,
                        step,
                        epoch=0,
                        extra={
                            "finetune_args": vars(args),
                            "best_metric": best_metric,
                            "best_metric_name": best_metric_name,
                        },
                    )

        if step % args.save_every == 0 or step == args.max_steps:
            save_wilor_checkpoint(
                output_dir / "latest.ckpt",
                student,
                optimizer,
                step,
                epoch=0,
                extra={
                    "finetune_args": vars(args),
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                },
            )

    print(f"Finished fine-tuning. Best {best_metric_name}: {best_metric:.4f}")
    print(f"Saved checkpoints under: {output_dir}")


if __name__ == "__main__":
    main(make_argparser().parse_args())

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
    format_loss_dict,
    infinite_loader,
    load_images_from_folder,
    load_detector,
    save_wilor_checkpoint,
    seed_everything,
    set_optional_loss_weight,
)
from wilor.models import load_wilor
from wilor.utils import recursive_to


def discover_videos(image_folder: str) -> list[str]:
    frame_paths = [Path(path) for path in load_images_from_folder(image_folder)]
    return sorted(
        {
            frame_path.parent.name[: -len("_frames")]
            for frame_path in frame_paths
            if frame_path.parent.name.endswith("_frames")
        }
    )


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

    dataset = DetectedVideoHandDataset(student.cfg, samples, rescale_factor=args.rescale_factor)
    dataloader = DataLoader(
        dataset,
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
    batch_iter = infinite_loader(dataloader)
    metrics_path = output_dir / "metrics.jsonl"

    best_loss = float("inf")

    print(f"Device: {device}")
    print(f"Videos: {', '.join(video_names)}")
    print(f"Training samples: {len(dataset)}")
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
        if loss_value < best_loss:
            best_loss = loss_value
            save_wilor_checkpoint(
                output_dir / "best.ckpt",
                student,
                optimizer,
                step,
                epoch=0,
                extra={"finetune_args": vars(args), "best_loss": best_loss},
            )

        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            metrics = format_loss_dict(student_output["losses"])
            metrics.update({"step": step, "loss_total": loss_value})
            append_metrics(metrics_path, metrics)
            print(
                f"[step {step:05d}] loss={loss_value:.4f} "
                f"cam={metrics.get('loss_camera_t_full', 0.0):.4f} "
                f"kp2d={metrics.get('loss_keypoints_2d', 0.0):.4f} "
                f"kp3d={metrics.get('loss_keypoints_3d', 0.0):.4f}"
            )

        if step % args.save_every == 0 or step == args.max_steps:
            save_wilor_checkpoint(
                output_dir / "latest.ckpt",
                student,
                optimizer,
                step,
                epoch=0,
                extra={"finetune_args": vars(args), "best_loss": best_loss},
            )

    print(f"Finished fine-tuning. Best loss: {best_loss:.4f}")
    print(f"Saved checkpoints under: {output_dir}")


if __name__ == "__main__":
    main(make_argparser().parse_args())

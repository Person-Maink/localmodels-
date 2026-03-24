import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from finetune_wilor_common import (
    SupervisedCameraDataset,
    ViPECameraIndex,
    append_metrics,
    apply_train_mode_for_scope,
    choose_device,
    configure_trainable_scope,
    count_trainable_parameters,
    format_loss_dict,
    infinite_loader,
    save_wilor_checkpoint,
    seed_everything,
    set_optional_loss_weight,
)
from wilor.models import load_wilor
from wilor.utils import recursive_to


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervised WiLoR fine-tuning with optional ViPE camera supervision."
    )
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg_path", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--pose_dir", type=str, default="../../outputs/vipe/pose")
    parser.add_argument("--intrinsics_dir", type=str, default="../../outputs/vipe/intrinsics")
    parser.add_argument("--output_dir", type=str, default="./finetune_runs/supervised_vipe")
    parser.add_argument("--video_name", type=str, default=None, help="Override the ViPE artifact video name for every sample.")
    parser.add_argument("--frame_index_pattern", type=str, default=r"(\d+)$")
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--train_scope", type=str, choices=["camera_head", "refine_net", "full"], default="refine_net")
    parser.add_argument("--camera_loss_weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mocap_file", type=str, default=None)
    parser.add_argument("--adversarial_weight", type=float, default=0.0)
    return parser


def main(args: argparse.Namespace) -> None:
    from hamba.hamba.datasets.image_dataset import ImageDataset
    from hamba.hamba.datasets.mocap_dataset import MoCapDataset

    seed_everything(args.seed)
    device = choose_device(args.use_gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student, _ = load_wilor(args.checkpoint, args.cfg_path)
    set_optional_loss_weight(student.cfg, "CAMERA_T_FULL", args.camera_loss_weight)
    set_optional_loss_weight(student.cfg, "ADVERSARIAL", args.adversarial_weight if args.mocap_file else 0.0)
    student = student.to(device)

    camera_index = ViPECameraIndex(args.pose_dir, args.intrinsics_dir)
    base_dataset = ImageDataset(
        student.cfg,
        dataset_file=args.dataset_file,
        img_dir=args.img_dir,
        train=True,
        rescale_factor=args.rescale_factor,
    )
    if args.sample_limit > 0:
        base_dataset = Subset(base_dataset, list(range(min(args.sample_limit, len(base_dataset)))))

    dataset = SupervisedCameraDataset(
        base_dataset=base_dataset,
        camera_index=camera_index,
        frame_index_pattern=args.frame_index_pattern,
        override_video_name=args.video_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        drop_last=False,
    )

    trainable_params = configure_trainable_scope(student, args.train_scope)
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters were selected for scope '{args.train_scope}'.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    mocap_iter = None
    optimizer_disc = None
    if args.mocap_file and args.adversarial_weight > 0.0:
        mocap_dataset = MoCapDataset(dataset_file=args.mocap_file)
        mocap_loader = DataLoader(
            mocap_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        mocap_iter = infinite_loader(mocap_loader)
        optimizer_disc = torch.optim.AdamW(
            student.discriminator.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    metrics_path = output_dir / "metrics.jsonl"
    batch_iter = infinite_loader(dataloader)
    best_loss = float("inf")

    print(f"Device: {device}")
    print(f"Training samples: {len(dataset)}")
    print(f"Train scope: {args.train_scope}")
    print(f"Trainable params: {count_trainable_parameters(student):,}")

    for step in range(1, args.max_steps + 1):
        apply_train_mode_for_scope(student, args.train_scope)
        batch = recursive_to(next(batch_iter), device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            student_output = student.forward_step(batch, train=True)
            loss = student.compute_loss(batch, student_output, train=True)

            if optimizer_disc is not None:
                pred_mano_params = student_output["pred_mano_params"]
                batch_size = pred_mano_params["hand_pose"].shape[0]
                disc_out = student.discriminator(
                    pred_mano_params["hand_pose"].reshape(batch_size, -1),
                    pred_mano_params["betas"].reshape(batch_size, -1),
                )
                loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
                loss = loss + args.adversarial_weight * loss_adv
                student_output["losses"]["loss_gen"] = loss_adv.detach()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if optimizer_disc is not None and mocap_iter is not None:
            pred_mano_params = student_output["pred_mano_params"]
            batch_size = pred_mano_params["hand_pose"].shape[0]
            mocap_batch = recursive_to(next(mocap_iter), device)
            loss_disc = student.training_step_discriminator(
                mocap_batch,
                pred_mano_params["hand_pose"].reshape(batch_size, -1),
                pred_mano_params["betas"].reshape(batch_size, -1),
                optimizer_disc,
            )
            student_output["losses"]["loss_disc"] = loss_disc

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

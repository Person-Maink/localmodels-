from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from finetune_wilor_common import (
    append_metrics,
    apply_train_mode_for_scope,
    build_teacher_supervision_batch,
    count_trainable_parameters,
    format_loss_dict,
    infinite_loader,
    save_wilor_checkpoint,
)
from temporal_losses import (
    TemporalViPECameraHead,
    TemporalWindowScorer,
    compute_temporal_loss_bundle,
    flatten_temporal_batch,
    reshape_temporal_output,
)
from wilor.models.lora import has_lora_modules
from wilor.utils import recursive_to


def compute_window_loss(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    window_batch: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    temporal_cfg: dict[str, Any],
    loss_cfg: dict[str, dict[str, Any]],
    temporal_scorer: TemporalWindowScorer | None,
    temporal_vipe_camera_head: TemporalViPECameraHead | None,
    train: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    window_batch = recursive_to(window_batch, device)
    flat_batch, (batch_size, window_size) = flatten_temporal_batch(window_batch)

    with torch.no_grad():
        teacher_output = teacher.forward_step(flat_batch, train=False)
        supervision_batch = build_teacher_supervision_batch(flat_batch, teacher_output)

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        student_output = student.forward_step(flat_batch, train=train)
        distill_loss = student.compute_loss(supervision_batch, student_output, train=train)
        student_output_seq = reshape_temporal_output(student_output, batch_size, window_size)
        temporal_loss, temporal_metrics = compute_temporal_loss_bundle(
            window_batch,
            student_output_seq,
            loss_cfg,
            temporal_cfg,
            temporal_scorer,
            temporal_vipe_camera_head,
        )
        total_loss = distill_loss + temporal_loss

    metrics = format_loss_dict(student_output["losses"])
    metrics["loss_total"] = float(total_loss.detach().item())
    metrics["window_size"] = float(window_size)
    for key, value in temporal_metrics.items():
        metrics[key] = (
            float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
        )
    return total_loss, metrics


def evaluate_window_dataloader(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader,
    device: torch.device,
    amp_enabled: bool,
    temporal_cfg: dict[str, Any],
    loss_cfg: dict[str, dict[str, Any]],
    temporal_scorer: TemporalWindowScorer | None,
    temporal_vipe_camera_head: TemporalViPECameraHead | None,
) -> dict[str, float]:
    was_student_training = student.training
    student.eval()
    scorer_was_training = temporal_scorer.training if temporal_scorer is not None else False
    if temporal_scorer is not None:
        temporal_scorer.eval()
    vipe_head_was_training = (
        temporal_vipe_camera_head.training
        if temporal_vipe_camera_head is not None
        else False
    )
    if temporal_vipe_camera_head is not None:
        temporal_vipe_camera_head.eval()

    metric_sums: dict[str, float] = {}
    total_windows = 0

    with torch.no_grad():
        for window_batch in dataloader:
            _, batch_metrics = compute_window_loss(
                student,
                teacher,
                window_batch,
                device=device,
                amp_enabled=amp_enabled,
                temporal_cfg=temporal_cfg,
                loss_cfg=loss_cfg,
                temporal_scorer=temporal_scorer,
                temporal_vipe_camera_head=temporal_vipe_camera_head,
                train=False,
            )
            batch_window_count = int(window_batch["img"].shape[0])
            total_windows += batch_window_count
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_window_count

    student.train(was_student_training)
    if temporal_scorer is not None:
        temporal_scorer.train(scorer_was_training)
    if temporal_vipe_camera_head is not None:
        temporal_vipe_camera_head.train(vipe_head_was_training)

    if total_windows == 0:
        return {"loss_total": 0.0, "window_count": 0.0}

    averaged = {key: value / total_windows for key, value in metric_sums.items()}
    averaged["window_count"] = float(total_windows)
    return averaged


def _active_temporal_families(loss_cfg: dict[str, dict[str, Any]]) -> list[str]:
    return [
        family_name
        for family_name in (
            "temporal_camera",
            "temporal_bbox_projected",
            "temporal_vipe_camera",
        )
        if loss_cfg[family_name]["enabled"]
    ]


def _collect_extra_modules(
    temporal_scorer: TemporalWindowScorer | None,
    temporal_vipe_camera_head: TemporalViPECameraHead | None,
) -> dict[str, torch.nn.Module] | None:
    extra_modules: dict[str, torch.nn.Module] = {}
    if temporal_scorer is not None:
        extra_modules["temporal_scorer"] = temporal_scorer
    if temporal_vipe_camera_head is not None:
        extra_modules["temporal_vipe_camera_head"] = temporal_vipe_camera_head
    return extra_modules or None


def print_run_summary(
    device: torch.device,
    resolved_experiment: dict[str, Any],
    video_names: list[str],
    split_stats: dict[str, Any],
    train_window_stats: dict[str, Any],
    val_window_stats: dict[str, Any],
    train_scope: str,
    lora_cfg: dict[str, Any],
    total_trainable_params: int,
    active_temporal_families: list[str],
    val_dataloader,
    window_mode: str,
    sample_count_mode: str,
) -> None:
    print(f"Device: {device}", flush=True)
    print(f"Experiment: {resolved_experiment['name']}", flush=True)
    print(f"Videos: {', '.join(video_names)}", flush=True)
    print(
        f"Frames: total={split_stats['total_frames']} "
        f"train={split_stats['train_frames']} val={split_stats['val_frames']}",
        flush=True,
    )
    if sample_count_mode == "lazy_detection":
        print(
            "Samples: computed lazily per batch from detector outputs; no global sample count is precomputed upfront.",
            flush=True,
        )
    else:
        print(
            f"Samples: total={split_stats['total_samples']} "
            f"train={split_stats['train_samples']} val={split_stats['val_samples']}",
            flush=True,
        )
    if window_mode == "lazy_frame_windows":
        print(
            "Temporal windows: candidate frame windows "
            f"train={train_window_stats['window_count']} "
            f"val={val_window_stats['window_count']} "
            f"dropped_train={train_window_stats['dropped_window_count']} "
            f"dropped_val={val_window_stats['dropped_window_count']}",
            flush=True,
        )
    else:
        print(
            f"Temporal windows: train={train_window_stats['window_count']} "
            f"val={val_window_stats['window_count']} "
            f"dropped_train={train_window_stats['dropped_window_count']} "
            f"dropped_val={val_window_stats['dropped_window_count']}",
            flush=True,
        )
    if split_stats["validation_split"] > 0.0 and val_dataloader is None:
        print(
            "Validation split requested, but there were not enough consecutive frames to reserve validation windows.",
            flush=True,
        )
    print(f"Train scope: {train_scope}", flush=True)
    print(
        "LoRA: "
        + (
            f"enabled rank={lora_cfg['rank']} alpha={lora_cfg['alpha']} "
            f"dropout={lora_cfg['dropout']} blocks=[{lora_cfg['block_start']}, {lora_cfg['block_end']}) "
            f"targets={','.join(lora_cfg['target_modules'])}"
            if lora_cfg["enabled"]
            else "disabled"
        ),
        flush=True,
    )
    print(f"Trainable params: {total_trainable_params:,}", flush=True)
    print(
        f"Temporal families: {', '.join(active_temporal_families) if active_temporal_families else 'none'}",
        flush=True,
    )


def run_training_loop(
    args,
    output_dir: Path,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    device: torch.device,
    resolved_experiment: dict[str, Any],
    temporal_cfg: dict[str, Any],
    loss_cfg: dict[str, dict[str, Any]],
    lora_cfg: dict[str, Any],
    data_bundle,
    log_fn: Callable[[str], None],
) -> None:
    log_fn(f"Configuring trainable scope: {args.train_scope}")
    trainable_params = apply_trainable_scope(
        student,
        args.train_scope,
        lora_cfg,
    )

    temporal_scorer = None
    if any(
        loss_cfg[family_name]["enabled"]
        and loss_cfg[family_name].get("formulation", "static") == "learnable"
        and loss_cfg[family_name]["scorer_weight"] > 0.0
        for family_name in ("temporal_camera", "temporal_bbox_projected")
    ):
        temporal_scorer = TemporalWindowScorer(
            hidden_dim=int(temporal_cfg["scorer_hidden_dim"]),
            layers=int(temporal_cfg["scorer_layers"]),
            dropout=float(temporal_cfg["scorer_dropout"]),
        ).to(device)
        log_fn("Initialized learnable temporal scorer.")

    temporal_vipe_camera_head = None
    if loss_cfg["temporal_vipe_camera"]["enabled"]:
        temporal_vipe_camera_head = TemporalViPECameraHead(
            hidden_dim=int(temporal_cfg["scorer_hidden_dim"]),
            layers=int(temporal_cfg["scorer_layers"]),
            dropout=float(temporal_cfg["scorer_dropout"]),
        ).to(device)
        log_fn("Initialized temporal ViPE camera refinement head.")

    if (
        args.train_scope == "temporal_only"
        and temporal_scorer is None
        and temporal_vipe_camera_head is None
    ):
        raise RuntimeError(
            "train_scope='temporal_only' requires at least one learnable temporal module "
            "so there are parameters to optimize while WiLoR stays frozen."
        )
    if not trainable_params and temporal_scorer is None and temporal_vipe_camera_head is None:
        raise RuntimeError(f"No trainable parameters were selected for scope '{args.train_scope}'.")

    optimizer_params: list[dict[str, Any]] = []
    if trainable_params:
        optimizer_params.append({"params": trainable_params})
    if temporal_scorer is not None:
        optimizer_params.append({"params": list(temporal_scorer.parameters())})
    if temporal_vipe_camera_head is not None:
        optimizer_params.append({"params": list(temporal_vipe_camera_head.parameters())})

    log_fn("Creating optimizer and AMP scaler.")
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    batch_iter = infinite_loader(data_bundle.train_dataloader)
    metrics_path = output_dir / "metrics.jsonl"
    log_fn(f"Metrics will be appended to {metrics_path}")

    best_metric = float("inf")
    best_metric_name = (
        "val_loss_total" if data_bundle.val_dataloader is not None else "train_loss_total"
    )
    active_temporal_families = _active_temporal_families(loss_cfg)
    total_trainable_params = count_trainable_parameters(student)
    if temporal_scorer is not None:
        total_trainable_params += sum(
            param.numel() for param in temporal_scorer.parameters() if param.requires_grad
        )
    if temporal_vipe_camera_head is not None:
        total_trainable_params += sum(
            param.numel()
            for param in temporal_vipe_camera_head.parameters()
            if param.requires_grad
        )

    print_run_summary(
        device=device,
        resolved_experiment=resolved_experiment,
        video_names=data_bundle.video_names,
        split_stats=data_bundle.split_stats,
        train_window_stats=data_bundle.train_window_stats,
        val_window_stats=data_bundle.val_window_stats,
        train_scope=args.train_scope,
        lora_cfg=lora_cfg,
        total_trainable_params=total_trainable_params,
        active_temporal_families=active_temporal_families,
        val_dataloader=data_bundle.val_dataloader,
        window_mode=data_bundle.window_mode,
        sample_count_mode=data_bundle.sample_count_mode,
    )
    log_fn("Entering optimization loop.")

    for step in range(1, args.max_steps + 1):
        if step == 1:
            log_fn("Fetching first training batch.")
        apply_train_mode_for_scope(
            student,
            args.train_scope,
            lora_enabled=lora_cfg["enabled"] and has_lora_modules(student),
        )
        if temporal_scorer is not None:
            temporal_scorer.train()
        if temporal_vipe_camera_head is not None:
            temporal_vipe_camera_head.train()

        window_batch = next(batch_iter)
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = compute_window_loss(
            student,
            teacher,
            window_batch,
            device=device,
            amp_enabled=scaler.is_enabled(),
            temporal_cfg=temporal_cfg,
            loss_cfg=loss_cfg,
            temporal_scorer=temporal_scorer,
            temporal_vipe_camera_head=temporal_vipe_camera_head,
            train=True,
        )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_value = float(loss.detach().item())
        should_log = step % args.log_every == 0 or step == 1 or step == args.max_steps

        if data_bundle.val_dataloader is None and loss_value < best_metric:
            best_metric = loss_value
            save_wilor_checkpoint(
                output_dir / "best.ckpt",
                student,
                optimizer,
                step,
                epoch=0,
                extra={
                    "finetune_args": vars(args),
                    "experiment_config": resolved_experiment,
                    "lora_config": lora_cfg,
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                },
                extra_modules=_collect_extra_modules(
                    temporal_scorer,
                    temporal_vipe_camera_head,
                ),
            )

        if should_log:
            train_metrics.update(
                {
                    "step": step,
                    "split": "train",
                    "train_window_count": float(data_bundle.train_window_stats["window_count"]),
                    "train_dropped_window_count": float(
                        data_bundle.train_window_stats["dropped_window_count"]
                    ),
                    "val_window_count": float(data_bundle.val_window_stats["window_count"]),
                    "val_dropped_window_count": float(
                        data_bundle.val_window_stats["dropped_window_count"]
                    ),
                }
            )
            append_metrics(metrics_path, train_metrics)
            print(
                f"[step {step:05d}] loss={train_metrics['loss_total']:.4f} "
                f"cam={train_metrics.get('loss_camera_t_full', 0.0):.4f} "
                f"temp_cam={train_metrics.get('loss_temporal_camera_base', 0.0):.4f} "
                f"bbox_p={train_metrics.get('loss_temporal_bbox_projected_base', 0.0):.4f} "
                f"vipe_t={train_metrics.get('loss_temporal_vipe_camera_align', 0.0):.4f}",
                flush=True,
            )
            if data_bundle.val_dataloader is not None:
                log_fn(f"Running validation at step {step}.")
                val_metrics = evaluate_window_dataloader(
                    student,
                    teacher,
                    data_bundle.val_dataloader,
                    device=device,
                    amp_enabled=scaler.is_enabled(),
                    temporal_cfg=temporal_cfg,
                    loss_cfg=loss_cfg,
                    temporal_scorer=temporal_scorer,
                    temporal_vipe_camera_head=temporal_vipe_camera_head,
                )
                val_metrics.update(
                    {
                        "step": step,
                        "split": "val",
                        "train_window_count": float(data_bundle.train_window_stats["window_count"]),
                        "train_dropped_window_count": float(
                            data_bundle.train_window_stats["dropped_window_count"]
                        ),
                        "val_window_count": float(data_bundle.val_window_stats["window_count"]),
                        "val_dropped_window_count": float(
                            data_bundle.val_window_stats["dropped_window_count"]
                        ),
                    }
                )
                append_metrics(metrics_path, val_metrics)
                print(
                    f"             val_loss={val_metrics.get('loss_total', 0.0):.4f} "
                    f"val_cam={val_metrics.get('loss_camera_t_full', 0.0):.4f} "
                    f"val_temp_cam={val_metrics.get('loss_temporal_camera_base', 0.0):.4f} "
                    f"val_bbox_p={val_metrics.get('loss_temporal_bbox_projected_base', 0.0):.4f} "
                    f"val_vipe_t={val_metrics.get('loss_temporal_vipe_camera_align', 0.0):.4f}",
                    flush=True,
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
                            "experiment_config": resolved_experiment,
                            "lora_config": lora_cfg,
                            "best_metric": best_metric,
                            "best_metric_name": best_metric_name,
                        },
                        extra_modules=_collect_extra_modules(
                            temporal_scorer,
                            temporal_vipe_camera_head,
                        ),
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
                    "experiment_config": resolved_experiment,
                    "lora_config": lora_cfg,
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                },
                extra_modules=_collect_extra_modules(
                    temporal_scorer,
                    temporal_vipe_camera_head,
                ),
            )
            log_fn(f"Saved latest checkpoint at step {step}.")

    print(f"Finished fine-tuning. Best {best_metric_name}: {best_metric:.4f}", flush=True)
    print(f"Saved checkpoints under: {output_dir}", flush=True)


def apply_trainable_scope(
    student: torch.nn.Module,
    train_scope: str,
    lora_cfg: dict[str, Any],
) -> list[torch.nn.Parameter]:
    from finetune_wilor_common import configure_trainable_scope

    return configure_trainable_scope(
        student,
        train_scope,
        include_lora=lora_cfg["enabled"],
    )

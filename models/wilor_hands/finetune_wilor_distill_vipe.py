import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from experiment_config import DEFAULT_EXPERIMENT_CONFIG, resolve_experiment_config
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
    filter_videos_with_vipe_artifacts,
    format_loss_dict,
    infinite_loader,
    load_detector,
    save_wilor_checkpoint,
    seed_everything,
    set_optional_loss_weight,
    split_samples_by_frame,
)
from temporal_losses import (
    TemporalWindowDataset,
    TemporalWindowScorer,
    build_temporal_windows,
    compute_temporal_loss_bundle,
    flatten_temporal_batch,
    reshape_temporal_output,
    temporal_window_collate,
)
from wilor.models import load_wilor
from wilor.utils import recursive_to


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune WiLoR with teacher distillation, ViPE supervision, and temporal loss ablations."
    )
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg_path", type=str, default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector_path", type=str, default="./pretrained_models/detector.pt")
    parser.add_argument("--image_folder", type=str, default="../../data/images/")
    parser.add_argument("--pose_dir", type=str, default="../../outputs/vipe/pose")
    parser.add_argument("--intrinsics_dir", type=str, default="../../outputs/vipe/intrinsics")
    parser.add_argument("--output_dir", type=str, default="./finetune_runs/distill_vipe")
    parser.add_argument("--loss_config", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
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
    parser.add_argument("--camera_loss_weight", type=float, default=0.01, help="Legacy alias for the ViPE camera supervision weight.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of temporal windows per optimization step.")
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
    parser.add_argument("--temporal_reduction", type=str, default=None, choices=["l1", "l2", "smooth_l1"])
    parser.add_argument("--temporal_scorer_hidden_dim", type=int, default=None)
    parser.add_argument("--temporal_scorer_layers", type=int, default=None)
    parser.add_argument("--temporal_scorer_dropout", type=float, default=None)
    parser.add_argument("--vipe_camera_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vipe_camera_weight", type=float, default=None)
    parser.add_argument("--temporal_camera_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temporal_camera_formulation", type=str, default=None, choices=["static", "learnable"])
    parser.add_argument("--temporal_camera_weight", type=float, default=None)
    parser.add_argument("--temporal_camera_scorer_weight", type=float, default=None)
    parser.add_argument("--temporal_bbox_projected_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temporal_bbox_projected_formulation", type=str, default=None, choices=["static", "learnable"])
    parser.add_argument("--temporal_bbox_projected_weight", type=float, default=None)
    parser.add_argument("--temporal_bbox_projected_scorer_weight", type=float, default=None)
    parser.add_argument("--temporal_bbox_input_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temporal_bbox_input_formulation", type=str, default=None, choices=["static", "learnable"])
    parser.add_argument("--temporal_bbox_input_weight", type=float, default=None)
    parser.add_argument("--temporal_bbox_input_scorer_weight", type=float, default=None)
    return parser


def _set_arg_from_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    dest: str,
    value: Any,
) -> None:
    current = getattr(args, dest)
    default = parser.get_default(dest)
    if isinstance(current, list):
        if not current or current == default:
            setattr(args, dest, copy.deepcopy(value))
        return
    if current is None or current == default:
        setattr(args, dest, copy.deepcopy(value))


def _apply_experiment_defaults(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    resolved_experiment: dict[str, Any],
) -> None:
    for dest, cfg_key in (
        ("train_scope", "train_scope"),
        ("validation_split", "validation_split"),
        ("sample_limit", "sample_limit"),
        ("detection_conf", "detection_conf"),
        ("rescale_factor", "rescale_factor"),
        ("batch_size", "batch_size"),
        ("num_workers", "num_workers"),
        ("max_steps", "max_steps"),
        ("log_every", "log_every"),
        ("save_every", "save_every"),
        ("seed", "seed"),
    ):
        _set_arg_from_config(args, parser, dest, resolved_experiment[cfg_key])

    _set_arg_from_config(args, parser, "lr", resolved_experiment["optimizer"]["lr"])
    _set_arg_from_config(
        args,
        parser,
        "weight_decay",
        resolved_experiment["optimizer"]["weight_decay"],
    )

    if not args.videos and args.all_videos == parser.get_default("all_videos"):
        if resolved_experiment["all_videos"]:
            args.all_videos = True
        elif resolved_experiment["videos"]:
            args.videos = list(resolved_experiment["videos"])

    temporal_cfg = resolved_experiment["temporal"]
    _set_arg_from_config(args, parser, "temporal_window_size", temporal_cfg["window_size"])
    _set_arg_from_config(args, parser, "temporal_window_stride", temporal_cfg["window_stride"])
    _set_arg_from_config(args, parser, "temporal_max_frame_gap", temporal_cfg["max_frame_gap"])
    _set_arg_from_config(args, parser, "temporal_reduction", temporal_cfg["reduction"])
    _set_arg_from_config(
        args,
        parser,
        "temporal_scorer_hidden_dim",
        temporal_cfg["scorer_hidden_dim"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_scorer_layers",
        temporal_cfg["scorer_layers"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_scorer_dropout",
        temporal_cfg["scorer_dropout"],
    )

    loss_cfg = resolved_experiment["losses"]
    _set_arg_from_config(
        args,
        parser,
        "vipe_camera_enabled",
        loss_cfg["vipe_camera"]["enabled"],
    )
    _set_arg_from_config(
        args,
        parser,
        "vipe_camera_weight",
        loss_cfg["vipe_camera"]["weight"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_camera_enabled",
        loss_cfg["temporal_camera"]["enabled"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_camera_formulation",
        loss_cfg["temporal_camera"]["formulation"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_camera_weight",
        loss_cfg["temporal_camera"]["weight"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_camera_scorer_weight",
        loss_cfg["temporal_camera"]["scorer_weight"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_bbox_projected_enabled",
        loss_cfg["temporal_bbox_projected"]["enabled"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_bbox_projected_formulation",
        loss_cfg["temporal_bbox_projected"]["formulation"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_bbox_projected_weight",
        loss_cfg["temporal_bbox_projected"]["weight"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_bbox_projected_scorer_weight",
        loss_cfg["temporal_bbox_projected"]["scorer_weight"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_bbox_input_enabled",
        loss_cfg["temporal_bbox_input"]["enabled"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_bbox_input_formulation",
        loss_cfg["temporal_bbox_input"]["formulation"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_bbox_input_weight",
        loss_cfg["temporal_bbox_input"]["weight"],
    )
    _set_arg_from_config(
        args,
        parser,
        "temporal_bbox_input_scorer_weight",
        loss_cfg["temporal_bbox_input"]["scorer_weight"],
    )


def _resolve_loss_settings(
    args: argparse.Namespace,
    loaded_experiment: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    base_loss_cfg = copy.deepcopy(
        loaded_experiment["losses"] if loaded_experiment else DEFAULT_EXPERIMENT_CONFIG["losses"]
    )

    vipe_weight = (
        float(args.vipe_camera_weight)
        if args.vipe_camera_weight is not None
        else float(args.camera_loss_weight)
    )
    base_loss_cfg["vipe_camera"]["enabled"] = (
        base_loss_cfg["vipe_camera"]["enabled"]
        if args.vipe_camera_enabled is None
        else bool(args.vipe_camera_enabled)
    )
    base_loss_cfg["vipe_camera"]["weight"] = vipe_weight
    base_loss_cfg["vipe_camera"]["scorer_weight"] = 0.0

    for family_name, enabled_attr, formulation_attr, weight_attr, scorer_attr in (
        (
            "temporal_camera",
            "temporal_camera_enabled",
            "temporal_camera_formulation",
            "temporal_camera_weight",
            "temporal_camera_scorer_weight",
        ),
        (
            "temporal_bbox_projected",
            "temporal_bbox_projected_enabled",
            "temporal_bbox_projected_formulation",
            "temporal_bbox_projected_weight",
            "temporal_bbox_projected_scorer_weight",
        ),
        (
            "temporal_bbox_input",
            "temporal_bbox_input_enabled",
            "temporal_bbox_input_formulation",
            "temporal_bbox_input_weight",
            "temporal_bbox_input_scorer_weight",
        ),
    ):
        base_loss_cfg[family_name]["formulation"] = str(
            base_loss_cfg[family_name].get("formulation", "static")
        )
        if getattr(args, enabled_attr) is not None:
            base_loss_cfg[family_name]["enabled"] = bool(getattr(args, enabled_attr))
        if getattr(args, formulation_attr) is not None:
            base_loss_cfg[family_name]["formulation"] = str(getattr(args, formulation_attr))
        if getattr(args, weight_attr) is not None:
            base_loss_cfg[family_name]["weight"] = float(getattr(args, weight_attr))
        if getattr(args, scorer_attr) is not None:
            base_loss_cfg[family_name]["scorer_weight"] = float(getattr(args, scorer_attr))

    return base_loss_cfg


def _resolve_temporal_settings(
    args: argparse.Namespace,
    loaded_experiment: dict[str, Any] | None,
) -> dict[str, Any]:
    temporal_cfg = copy.deepcopy(
        loaded_experiment["temporal"] if loaded_experiment else DEFAULT_EXPERIMENT_CONFIG["temporal"]
    )
    for attr, key in (
        ("temporal_window_size", "window_size"),
        ("temporal_window_stride", "window_stride"),
        ("temporal_max_frame_gap", "max_frame_gap"),
        ("temporal_reduction", "reduction"),
        ("temporal_scorer_hidden_dim", "scorer_hidden_dim"),
        ("temporal_scorer_layers", "scorer_layers"),
        ("temporal_scorer_dropout", "scorer_dropout"),
    ):
        value = getattr(args, attr)
        if value is not None:
            temporal_cfg[key] = value
    return temporal_cfg


def _build_resolved_experiment_snapshot(
    args: argparse.Namespace,
    loaded_experiment: dict[str, Any] | None,
    temporal_cfg: dict[str, Any],
    loss_cfg: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    snapshot = copy.deepcopy(loaded_experiment or DEFAULT_EXPERIMENT_CONFIG)
    snapshot["name"] = args.experiment_name or snapshot.get("name") or "manual_distill_run"
    snapshot["source_loss_config"] = (
        str(Path(args.loss_config).expanduser().resolve())
        if args.loss_config
        else None
    )
    snapshot["run_name_suffix"] = snapshot.get("run_name_suffix", "")
    snapshot["train_mode"] = snapshot.get("train_mode", "distill")
    snapshot["videos"] = [] if args.all_videos else sorted(set(args.videos))
    snapshot["all_videos"] = bool(args.all_videos)
    snapshot["train_scope"] = args.train_scope
    snapshot["validation_split"] = float(args.validation_split)
    snapshot["sample_limit"] = int(args.sample_limit)
    snapshot["detection_conf"] = float(args.detection_conf)
    snapshot["rescale_factor"] = float(args.rescale_factor)
    snapshot["batch_size"] = int(args.batch_size)
    snapshot["num_workers"] = int(args.num_workers)
    snapshot["max_steps"] = int(args.max_steps)
    snapshot["log_every"] = int(args.log_every)
    snapshot["save_every"] = int(args.save_every)
    snapshot["seed"] = int(args.seed)
    snapshot["optimizer"] = {
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
    }
    snapshot["temporal"] = copy.deepcopy(temporal_cfg)
    snapshot["losses"] = copy.deepcopy(loss_cfg)
    return snapshot


def _save_resolved_experiment(output_dir: Path, resolved_experiment: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_experiment.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_experiment, handle, sort_keys=False)


def _compute_window_loss(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    window_batch: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    temporal_cfg: dict[str, Any],
    loss_cfg: dict[str, dict[str, Any]],
    temporal_scorer: TemporalWindowScorer | None,
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
        )
        total_loss = distill_loss + temporal_loss

    metrics = format_loss_dict(student_output["losses"])
    metrics["loss_total"] = float(total_loss.detach().item())
    metrics["window_size"] = float(window_size)
    for key, value in temporal_metrics.items():
        metrics[key] = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    return total_loss, metrics


def _evaluate_window_dataloader(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    temporal_cfg: dict[str, Any],
    loss_cfg: dict[str, dict[str, Any]],
    temporal_scorer: TemporalWindowScorer | None,
) -> dict[str, float]:
    was_student_training = student.training
    student.eval()
    scorer_was_training = temporal_scorer.training if temporal_scorer is not None else False
    if temporal_scorer is not None:
        temporal_scorer.eval()

    metric_sums: dict[str, float] = {}
    total_windows = 0

    with torch.no_grad():
        for window_batch in dataloader:
            _, batch_metrics = _compute_window_loss(
                student,
                teacher,
                window_batch,
                device=device,
                amp_enabled=amp_enabled,
                temporal_cfg=temporal_cfg,
                loss_cfg=loss_cfg,
                temporal_scorer=temporal_scorer,
                train=False,
            )
            batch_window_count = int(window_batch["img"].shape[0])
            total_windows += batch_window_count
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_window_count

    student.train(was_student_training)
    if temporal_scorer is not None:
        temporal_scorer.train(scorer_was_training)

    if total_windows == 0:
        return {"loss_total": 0.0, "window_count": 0.0}

    averaged = {key: value / total_windows for key, value in metric_sums.items()}
    averaged["window_count"] = float(total_windows)
    return averaged


def _active_temporal_families(loss_cfg: dict[str, dict[str, Any]]) -> list[str]:
    return [
        family_name
        for family_name in ("temporal_camera", "temporal_bbox_projected", "temporal_bbox_input")
        if loss_cfg[family_name]["enabled"]
    ]


def _log_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _log_progress("Starting fine-tuning entrypoint.")
    loaded_experiment = None
    if args.loss_config:
        _log_progress(
            f"Resolving experiment config from {args.loss_config}"
            + (
                f" (experiment={args.experiment_name})"
                if args.experiment_name
                else ""
            )
        )
        loaded_experiment = resolve_experiment_config(args.loss_config, args.experiment_name)
        _apply_experiment_defaults(args, parser, loaded_experiment)
        args.experiment_name = args.experiment_name or loaded_experiment["name"]
        _log_progress(f"Resolved experiment '{args.experiment_name}'.")

    seed_everything(args.seed)
    device = choose_device(args.use_gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_progress(f"Using device {device}; output directory is {output_dir}")

    temporal_cfg = _resolve_temporal_settings(args, loaded_experiment)
    loss_cfg = _resolve_loss_settings(args, loaded_experiment)
    resolved_experiment = _build_resolved_experiment_snapshot(
        args,
        loaded_experiment,
        temporal_cfg,
        loss_cfg,
    )
    _save_resolved_experiment(output_dir, resolved_experiment)
    _log_progress("Saved resolved experiment snapshot.")

    if args.all_videos:
        _log_progress("Discovering videos from image folder because --all_videos is enabled.")
        video_names = discover_videos(args.image_folder)
    else:
        video_names = sorted(set(args.videos))
        _log_progress(
            "Using explicitly requested video selection: "
            f"{', '.join(video_names) if video_names else '<none>'}"
        )
    if not video_names:
        raise ValueError("No videos selected. Pass --video ... or use --all_videos.")

    _log_progress(
        f"Filtering {len(video_names)} selected video(s) for matching ViPE artifacts."
    )
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

    _log_progress(
        "Proceeding with video(s): "
        f"{', '.join(video_names)}"
    )
    _log_progress("Loading student checkpoint and cloning teacher model.")
    student, _ = load_wilor(args.checkpoint, args.cfg_path)
    teacher = copy.deepcopy(student)

    set_optional_loss_weight(student.cfg, "ADVERSARIAL", 0.0)
    vipe_weight = loss_cfg["vipe_camera"]["weight"] if loss_cfg["vipe_camera"]["enabled"] else 0.0
    set_optional_loss_weight(student.cfg, "CAMERA_T_FULL", vipe_weight)

    student = student.to(device)
    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    _log_progress("Loading detector and ViPE camera index.")
    detector = load_detector(args.detector_path, device)
    camera_index = ViPECameraIndex(args.pose_dir, args.intrinsics_dir)

    detection_cache = args.detection_cache
    if detection_cache is None:
        video_slug = "all" if args.all_videos else "_".join(video_names)
        detection_cache = str(output_dir / f"detections_{video_slug}.json")

    _log_progress(
        f"Building detection samples (cache path: {detection_cache})."
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
        raise RuntimeError("No detector samples were generated for fine-tuning.")
    _log_progress(f"Collected {len(samples)} detector sample(s).")

    _log_progress(
        f"Splitting samples into train/val with validation_split={args.validation_split}."
    )
    train_samples, val_samples, split_stats = split_samples_by_frame(
        samples,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    if not train_samples:
        raise RuntimeError(
            "Validation split left no training samples. Reduce --validation_split or provide more frames."
        )

    _log_progress(
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

    _log_progress("Constructing training dataset and dataloader.")
    train_base_dataset = DetectedVideoHandDataset(
        student.cfg,
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

    val_window_stats = {"window_count": 0, "dropped_window_count": 0, "stream_count": 0, "broken_segment_count": 0}
    val_dataloader = None
    if val_samples:
        _log_progress("Building validation temporal windows and dataloader.")
        val_windows, val_window_stats = build_temporal_windows(
            val_samples,
            window_size=int(temporal_cfg["window_size"]),
            window_stride=int(temporal_cfg["window_stride"]),
            max_frame_gap=int(temporal_cfg["max_frame_gap"]),
        )
        if val_windows:
            val_base_dataset = DetectedVideoHandDataset(
                student.cfg,
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

    _log_progress(f"Configuring trainable scope: {args.train_scope}")
    trainable_params = configure_trainable_scope(student, args.train_scope)
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters were selected for scope '{args.train_scope}'.")

    temporal_scorer = None
    if any(
        loss_cfg[family_name]["enabled"]
        and loss_cfg[family_name].get("formulation", "static") == "learnable"
        and loss_cfg[family_name]["scorer_weight"] > 0.0
        for family_name in _active_temporal_families(loss_cfg)
    ):
        temporal_scorer = TemporalWindowScorer(
            hidden_dim=int(temporal_cfg["scorer_hidden_dim"]),
            layers=int(temporal_cfg["scorer_layers"]),
            dropout=float(temporal_cfg["scorer_dropout"]),
        ).to(device)
        _log_progress("Initialized learnable temporal scorer.")

    optimizer_params: list[dict[str, Any]] = [{"params": trainable_params}]
    if temporal_scorer is not None:
        optimizer_params.append({"params": list(temporal_scorer.parameters())})

    _log_progress("Creating optimizer and AMP scaler.")
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    batch_iter = infinite_loader(train_dataloader)
    metrics_path = output_dir / "metrics.jsonl"
    _log_progress(f"Metrics will be appended to {metrics_path}")

    best_metric = float("inf")
    best_metric_name = "val_loss_total" if val_dataloader is not None else "train_loss_total"
    active_temporal_families = _active_temporal_families(loss_cfg)
    total_trainable_params = count_trainable_parameters(student)
    if temporal_scorer is not None:
        total_trainable_params += sum(
            param.numel() for param in temporal_scorer.parameters() if param.requires_grad
        )

    print(f"Device: {device}", flush=True)
    print(f"Experiment: {resolved_experiment['name']}", flush=True)
    print(f"Videos: {', '.join(video_names)}", flush=True)
    print(
        f"Frames: total={split_stats['total_frames']} "
        f"train={split_stats['train_frames']} val={split_stats['val_frames']}"
    , flush=True)
    print(
        f"Samples: total={split_stats['total_samples']} "
        f"train={split_stats['train_samples']} val={split_stats['val_samples']}"
    , flush=True)
    print(
        f"Temporal windows: train={train_window_stats['window_count']} "
        f"val={val_window_stats['window_count']} "
        f"dropped_train={train_window_stats['dropped_window_count']} "
        f"dropped_val={val_window_stats['dropped_window_count']}"
    , flush=True)
    if args.validation_split > 0.0 and val_dataloader is None:
        print(
            "Validation split requested, but there were not enough consecutive frames to reserve validation windows."
        , flush=True)
    print(f"Train scope: {args.train_scope}", flush=True)
    print(f"Trainable params: {total_trainable_params:,}", flush=True)
    print(
        f"Temporal families: {', '.join(active_temporal_families) if active_temporal_families else 'none'}",
        flush=True,
    )
    _log_progress("Entering optimization loop.")

    for step in range(1, args.max_steps + 1):
        if step == 1:
            _log_progress("Fetching first training batch.")
        apply_train_mode_for_scope(student, args.train_scope)
        if temporal_scorer is not None:
            temporal_scorer.train()

        window_batch = next(batch_iter)
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = _compute_window_loss(
            student,
            teacher,
            window_batch,
            device=device,
            amp_enabled=scaler.is_enabled(),
            temporal_cfg=temporal_cfg,
            loss_cfg=loss_cfg,
            temporal_scorer=temporal_scorer,
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
                    "experiment_config": resolved_experiment,
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                },
                extra_modules={"temporal_scorer": temporal_scorer} if temporal_scorer is not None else None,
            )

        if should_log:
            train_metrics.update(
                {
                    "step": step,
                    "split": "train",
                    "train_window_count": float(train_window_stats["window_count"]),
                    "train_dropped_window_count": float(train_window_stats["dropped_window_count"]),
                    "val_window_count": float(val_window_stats["window_count"]),
                    "val_dropped_window_count": float(val_window_stats["dropped_window_count"]),
                }
            )
            append_metrics(metrics_path, train_metrics)
            print(
                f"[step {step:05d}] loss={train_metrics['loss_total']:.4f} "
                f"cam={train_metrics.get('loss_camera_t_full', 0.0):.4f} "
                f"temp_cam={train_metrics.get('loss_temporal_camera_base', 0.0):.4f} "
                f"bbox_p={train_metrics.get('loss_temporal_bbox_projected_base', 0.0):.4f} "
                f"bbox_i={train_metrics.get('loss_temporal_bbox_input_base', 0.0):.4f}"
            , flush=True)
            if val_dataloader is not None:
                _log_progress(f"Running validation at step {step}.")
                val_metrics = _evaluate_window_dataloader(
                    student,
                    teacher,
                    val_dataloader,
                    device=device,
                    amp_enabled=scaler.is_enabled(),
                    temporal_cfg=temporal_cfg,
                    loss_cfg=loss_cfg,
                    temporal_scorer=temporal_scorer,
                )
                val_metrics.update(
                    {
                        "step": step,
                        "split": "val",
                        "train_window_count": float(train_window_stats["window_count"]),
                        "train_dropped_window_count": float(train_window_stats["dropped_window_count"]),
                        "val_window_count": float(val_window_stats["window_count"]),
                        "val_dropped_window_count": float(val_window_stats["dropped_window_count"]),
                    }
                )
                append_metrics(metrics_path, val_metrics)
                print(
                    f"             val_loss={val_metrics.get('loss_total', 0.0):.4f} "
                    f"val_cam={val_metrics.get('loss_camera_t_full', 0.0):.4f} "
                    f"val_temp_cam={val_metrics.get('loss_temporal_camera_base', 0.0):.4f} "
                    f"val_bbox_p={val_metrics.get('loss_temporal_bbox_projected_base', 0.0):.4f} "
                    f"val_bbox_i={val_metrics.get('loss_temporal_bbox_input_base', 0.0):.4f}"
                , flush=True)
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
                            "best_metric": best_metric,
                            "best_metric_name": best_metric_name,
                        },
                        extra_modules={"temporal_scorer": temporal_scorer} if temporal_scorer is not None else None,
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
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                },
                extra_modules={"temporal_scorer": temporal_scorer} if temporal_scorer is not None else None,
            )
            _log_progress(f"Saved latest checkpoint at step {step}.")

    print(f"Finished fine-tuning. Best {best_metric_name}: {best_metric:.4f}", flush=True)
    print(f"Saved checkpoints under: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = make_argparser()
    main(parser.parse_args(), parser)

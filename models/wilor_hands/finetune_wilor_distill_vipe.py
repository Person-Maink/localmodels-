import copy
from pathlib import Path

import yaml

from finetune_wilor_cli import make_argparser
from finetune_wilor_common import (
    ViPECameraIndex,
    choose_device,
    load_detector,
    seed_everything,
    set_optional_loss_weight,
)
from finetune_wilor_data import build_window_data
from finetune_wilor_experiment import resolve_runtime_experiment
from finetune_wilor_training import run_training_loop
from wilor.models import load_wilor
from wilor.models.lora import apply_lora_to_wilor


def _log_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _save_resolved_experiment(output_dir: Path, resolved_experiment: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_experiment.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_experiment, handle, sort_keys=False)


def main(args, parser) -> None:
    _log_progress("Starting fine-tuning entrypoint.")
    _, resolved_experiment = resolve_runtime_experiment(args, parser, log_fn=_log_progress)

    seed_everything(args.seed)
    device = choose_device(args.use_gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_progress(f"Using device {device}; output directory is {output_dir}")

    temporal_cfg = resolved_experiment["temporal"]
    lora_cfg = resolved_experiment["lora"]
    loss_cfg = resolved_experiment["losses"]
    _save_resolved_experiment(output_dir, resolved_experiment)
    _log_progress("Saved resolved experiment snapshot.")

    _log_progress("Loading student checkpoint and cloning teacher model.")
    student, _ = load_wilor(args.checkpoint, args.cfg_path)
    lora_wrapped_modules: list[str] = []
    if lora_cfg["enabled"]:
        lora_wrapped_modules = apply_lora_to_wilor(student, lora_cfg)
        _log_progress(
            "LoRA is enabled with targets="
            f"{','.join(lora_cfg['target_modules'])}, "
            f"blocks=[{lora_cfg['block_start']}, {lora_cfg['block_end']})."
        )
        if lora_wrapped_modules:
            _log_progress(
                f"Wrapped {len(lora_wrapped_modules)} attention linear(s) with LoRA adapters."
            )
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

    data_bundle = build_window_data(
        args=args,
        output_dir=output_dir,
        model_cfg=student.cfg,
        detector=detector,
        camera_index=camera_index,
        temporal_cfg=temporal_cfg,
        log_fn=_log_progress,
    )

    run_training_loop(
        args=args,
        output_dir=output_dir,
        student=student,
        teacher=teacher,
        device=device,
        resolved_experiment=resolved_experiment,
        temporal_cfg=temporal_cfg,
        loss_cfg=loss_cfg,
        lora_cfg=lora_cfg,
        data_bundle=data_bundle,
        log_fn=_log_progress,
    )


if __name__ == "__main__":
    parser = make_argparser()
    main(parser.parse_args(), parser)

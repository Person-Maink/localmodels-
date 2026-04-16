from __future__ import annotations

import math
import re
import shlex
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COMMAND_RE = re.compile(r"^Command: (.+)$", re.MULTILINE)
HOURS_RE = re.compile(r"Execution took ([0-9]+(?:\.[0-9]+)?) hours")


@dataclass(frozen=True)
class TrainingRun:
    elapsed_seconds: int
    train_scope: str
    max_steps: int
    batch_size: int
    sample_limit: int
    log_every: int
    all_videos: bool
    video_count: int


@dataclass(frozen=True)
class RuntimeEstimate:
    seconds: int
    time_str: str
    source: str
    matched_runs: int


def _time_to_seconds(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def _seconds_to_hms(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _round_up(total_seconds: int, quantum_seconds: int) -> int:
    if quantum_seconds <= 0:
        return total_seconds
    return int(math.ceil(total_seconds / quantum_seconds) * quantum_seconds)


def _get_arg(tokens: list[str], flag: str, default: str = "") -> str:
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return default


def _has_flag(tokens: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in tokens)


def _parse_training_log(path: Path) -> TrainingRun | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "Finished fine-tuning." not in text:
        return None

    duration_match = HOURS_RE.search(text)
    command_match = COMMAND_RE.search(text)
    if not duration_match or not command_match:
        return None

    try:
        command_tokens = shlex.split(command_match.group(1))
    except ValueError:
        return None

    try:
        elapsed_seconds = int(round(float(duration_match.group(1)) * 3600))
        max_steps = int(_get_arg(command_tokens, "--max_steps", "0"))
        batch_size = int(_get_arg(command_tokens, "--batch_size", "0"))
        sample_limit = int(_get_arg(command_tokens, "--sample_limit", "0"))
        log_every = int(_get_arg(command_tokens, "--log_every", "0"))
    except ValueError:
        return None

    video_count = sum(
        1
        for token in command_tokens
        if token == "--video" or token.startswith("--video=")
    )

    return TrainingRun(
        elapsed_seconds=elapsed_seconds,
        train_scope=_get_arg(command_tokens, "--train_scope", "refine_net"),
        max_steps=max_steps,
        batch_size=batch_size,
        sample_limit=sample_limit,
        log_every=log_every,
        all_videos=_has_flag(command_tokens, "--all_videos"),
        video_count=video_count,
    )


def load_training_runs(logs_dir: Path) -> list[TrainingRun]:
    if not logs_dir.is_dir():
        return []

    runs: list[TrainingRun] = []
    for path in sorted(logs_dir.glob("wilor-train_*.out")):
        run = _parse_training_log(path)
        if run is not None and run.max_steps > 0 and run.batch_size > 0:
            runs.append(run)
    return runs


def _target_video_mode(resolved_experiment: dict[str, Any]) -> tuple[bool, int]:
    all_videos = bool(resolved_experiment.get("all_videos", False))
    if all_videos:
        return True, 0
    videos = resolved_experiment.get("videos", []) or []
    return False, len(videos)


def estimate_runtime(
    resolved_experiment: dict[str, Any],
    logs_dir: Path,
    *,
    safety_factor: float = 1.35,
    rounding_minutes: int = 15,
    max_partition_time: str = "04:00:00",
    fallback_time: str = "01:00:00",
) -> RuntimeEstimate:
    runs = load_training_runs(logs_dir)
    if not runs:
        fallback_seconds = _time_to_seconds(fallback_time)
        return RuntimeEstimate(
            seconds=fallback_seconds,
            time_str=fallback_time,
            source="no historical finetune logs found; using fallback",
            matched_runs=0,
        )

    target_scope = str(resolved_experiment.get("train_scope", "refine_net"))
    target_batch_size = int(resolved_experiment.get("batch_size", 8))
    target_max_steps = int(resolved_experiment.get("max_steps", 1000))
    target_sample_limit = int(resolved_experiment.get("sample_limit", 0))
    target_all_videos, target_video_count = _target_video_mode(resolved_experiment)
    target_log_every = int(resolved_experiment.get("log_every", 0))

    exact_matches = [
        run
        for run in runs
        if run.train_scope == target_scope
        and run.batch_size == target_batch_size
        and run.max_steps == target_max_steps
        and run.sample_limit == target_sample_limit
        and run.all_videos == target_all_videos
        and (
            target_all_videos
            or run.video_count == target_video_count
        )
        and (target_log_every == 0 or run.log_every == target_log_every)
    ]

    source: str
    matched_runs: int
    if exact_matches:
        base_seconds = int(statistics.median(run.elapsed_seconds for run in exact_matches))
        source = "median of exact historical matches"
        matched_runs = len(exact_matches)
    else:
        candidates = [
            run
            for run in runs
            if run.train_scope == target_scope and run.batch_size == target_batch_size
        ]
        if not candidates:
            candidates = [run for run in runs if run.train_scope == target_scope]
        if not candidates:
            candidates = list(runs)

        per_step_seconds = statistics.median(
            run.elapsed_seconds / run.max_steps
            for run in candidates
            if run.max_steps > 0
        )
        base_seconds = int(round(per_step_seconds * target_max_steps))
        source = (
            f"median per-step time from {len(candidates)} historical runs "
            f"scaled to {target_max_steps} steps"
        )
        matched_runs = len(candidates)

    # Inflate the raw estimate a bit and round up so we do not underbid the walltime.
    adjusted_seconds = int(math.ceil(base_seconds * safety_factor))
    rounded_seconds = _round_up(adjusted_seconds, rounding_minutes * 60)
    capped_seconds = min(rounded_seconds, _time_to_seconds(max_partition_time))

    return RuntimeEstimate(
        seconds=capped_seconds,
        time_str=_seconds_to_hms(capped_seconds),
        source=source,
        matched_runs=matched_runs,
    )

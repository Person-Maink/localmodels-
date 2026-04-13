from collections import defaultdict
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


TEMPORAL_FAMILY_DIMS = {
    "temporal_camera": 3,
    "temporal_bbox_projected": 4,
    "temporal_bbox_input": 4,
}


def build_temporal_windows(
    samples: Sequence[Dict[str, Any]],
    window_size: int,
    window_stride: int,
    max_frame_gap: int,
) -> tuple[list[list[int]], Dict[str, int]]:
    if window_size < 3:
        raise ValueError(f"window_size must be >= 3. Received {window_size}.")
    if window_stride < 1:
        raise ValueError(f"window_stride must be >= 1. Received {window_stride}.")
    if max_frame_gap < 1:
        raise ValueError(f"max_frame_gap must be >= 1. Received {max_frame_gap}.")

    streams: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for sample_idx, sample in enumerate(samples):
        stream_key = (
            str(sample["video_name"]),
            int(round(float(sample["right"]))),
            int(sample["det_idx"]),
        )
        streams[stream_key].append((int(sample["frame_idx"]), sample_idx))

    all_windows: list[list[int]] = []
    dropped_window_count = 0
    broken_segment_count = 0

    for stream_entries in streams.values():
        sorted_entries = sorted(stream_entries, key=lambda item: item[0])
        current_segment: list[int] = []
        prev_frame_idx: int | None = None

        for frame_idx, sample_idx in sorted_entries:
            if prev_frame_idx is None or frame_idx - prev_frame_idx <= max_frame_gap:
                current_segment.append(sample_idx)
            else:
                segment_windows, dropped = _segment_to_windows(
                    current_segment,
                    window_size,
                    window_stride,
                )
                all_windows.extend(segment_windows)
                dropped_window_count += dropped
                broken_segment_count += 1
                current_segment = [sample_idx]
            prev_frame_idx = frame_idx

        if current_segment:
            segment_windows, dropped = _segment_to_windows(
                current_segment,
                window_size,
                window_stride,
            )
            all_windows.extend(segment_windows)
            dropped_window_count += dropped

    return all_windows, {
        "stream_count": len(streams),
        "window_count": len(all_windows),
        "dropped_window_count": dropped_window_count,
        "broken_segment_count": broken_segment_count,
    }


def _segment_to_windows(
    segment: list[int],
    window_size: int,
    window_stride: int,
) -> tuple[list[list[int]], int]:
    if len(segment) < window_size:
        return [], 1

    windows = [
        segment[start : start + window_size]
        for start in range(0, len(segment) - window_size + 1, window_stride)
    ]
    last_start = (len(segment) - window_size)
    has_remainder = last_start % window_stride != 0
    dropped_window_count = 1 if has_remainder else 0
    return windows, dropped_window_count


class TemporalWindowDataset(Dataset):
    def __init__(self, base_dataset: Dataset, windows: Sequence[Sequence[int]]):
        self.base_dataset = base_dataset
        self.windows = [list(window) for window in windows]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> list[Dict[str, Any]]:
        return [self.base_dataset[sample_idx] for sample_idx in self.windows[idx]]


def temporal_window_collate(batch_windows: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    return default_collate([default_collate(window) for window in batch_windows])


def flatten_temporal_batch(batch: Dict[str, Any]) -> tuple[Dict[str, Any], tuple[int, int]]:
    batch_size, window_size = batch["img"].shape[:2]
    return _flatten_temporal_value(batch, batch_size, window_size), (batch_size, window_size)


def _flatten_temporal_value(value: Any, batch_size: int, window_size: int) -> Any:
    if isinstance(value, dict):
        return {
            key: _flatten_temporal_value(item, batch_size, window_size)
            for key, item in value.items()
        }
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        if value.shape[0] != batch_size:
            return value
        if value.ndim > 1 and value.shape[1] != window_size:
            return value
        return value.reshape(batch_size * window_size, *value.shape[2:])
    if isinstance(value, list):
        if len(value) == batch_size and value and isinstance(value[0], list):
            return [entry for sublist in value for entry in sublist]
        return value
    return value


def reshape_temporal_output(output: Dict[str, Any], batch_size: int, window_size: int) -> Dict[str, Any]:
    return _reshape_temporal_value(output, batch_size, window_size)


def _reshape_temporal_value(value: Any, batch_size: int, window_size: int) -> Any:
    if isinstance(value, dict):
        return {
            key: _reshape_temporal_value(item, batch_size, window_size)
            for key, item in value.items()
        }
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size * window_size:
        return value.reshape(batch_size, window_size, *value.shape[1:])
    return value


def project_keypoints_to_full_image(
    pred_keypoints_2d: torch.Tensor,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    projected = pred_keypoints_2d.clone()
    multiplier = 2.0 * batch["right"].float().unsqueeze(-1) - 1.0
    projected[..., 0] = multiplier * projected[..., 0]
    projected = projected * batch["box_size"].float().unsqueeze(-1).unsqueeze(-1)
    projected = projected + batch["box_center"].float().unsqueeze(-2)
    return projected


def bbox_sequence_from_keypoints(
    pred_keypoints_2d: torch.Tensor,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    projected = project_keypoints_to_full_image(pred_keypoints_2d, batch)
    mins = projected.amin(dim=-2)
    maxs = projected.amax(dim=-2)
    centers = 0.5 * (mins + maxs)
    sizes = (maxs - mins).clamp_min(1e-6)
    bbox = torch.cat([centers, sizes], dim=-1)
    return normalize_bbox_sequence(bbox, batch["img_size"].float())


def input_bbox_sequence(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    box_size = batch["box_size"].float()
    bbox = torch.cat(
        [
            batch["box_center"].float(),
            torch.stack([box_size, box_size], dim=-1),
        ],
        dim=-1,
    )
    return normalize_bbox_sequence(bbox, batch["img_size"].float())


def normalize_bbox_sequence(
    bbox_sequence: torch.Tensor,
    img_size: torch.Tensor,
) -> torch.Tensor:
    scale = torch.cat([img_size, img_size], dim=-1).clamp_min(1e-6)
    return bbox_sequence / scale


def normalize_camera_sequence(
    pred_cam_t_full: torch.Tensor,
    scaled_focal_length: torch.Tensor,
) -> torch.Tensor:
    if scaled_focal_length.ndim == pred_cam_t_full.ndim:
        denom = scaled_focal_length.mean(dim=-1, keepdim=True)
    else:
        denom = scaled_focal_length.unsqueeze(-1)
    return pred_cam_t_full / denom.clamp_min(1e-6)


def compute_second_difference(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.shape[1] < 3:
        return sequence.new_zeros(sequence.shape[0], 0, sequence.shape[-1])
    return sequence[:, 2:] - 2.0 * sequence[:, 1:-1] + sequence[:, :-2]


def reduce_temporal_residual(
    residual: torch.Tensor,
    reduction: str = "smooth_l1",
) -> torch.Tensor:
    if residual.numel() == 0:
        return residual.new_zeros(())
    target = torch.zeros_like(residual)
    if reduction == "l1":
        return F.l1_loss(residual, target, reduction="mean")
    if reduction == "l2":
        return F.mse_loss(residual, target, reduction="mean")
    if reduction == "smooth_l1":
        return F.smooth_l1_loss(residual, target, reduction="mean")
    raise ValueError(f"Unsupported temporal reduction '{reduction}'.")


class TemporalWindowScorer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_projections = nn.ModuleDict(
            {
                family: nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
                for family, input_dim in TEMPORAL_FAMILY_DIMS.items()
            }
        )

        blocks: list[nn.Module] = []
        for _ in range(max(layers, 1)):
            blocks.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.shared_backbone = nn.Sequential(*blocks)
        self.heads = nn.ModuleDict(
            {
                family: nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(hidden_dim, 1),
                )
                for family in TEMPORAL_FAMILY_DIMS
            }
        )
        self.output_activation = nn.Softplus()

    def score_family(self, family_name: str, residual: torch.Tensor) -> torch.Tensor:
        if residual.numel() == 0:
            return residual.new_zeros(residual.shape[0])

        x = residual.transpose(1, 2)
        x = self.input_projections[family_name](x)
        x = self.shared_backbone(x)
        score = self.heads[family_name](x).squeeze(-1)
        return self.output_activation(score)


def compute_temporal_loss_bundle(
    window_batch: Dict[str, torch.Tensor],
    student_output_seq: Dict[str, torch.Tensor],
    loss_cfg: Dict[str, Dict[str, Any]],
    temporal_cfg: Dict[str, Any],
    scorer: TemporalWindowScorer | None,
) -> tuple[torch.Tensor, Dict[str, float | torch.Tensor]]:
    metrics: Dict[str, float | torch.Tensor] = {}
    total_loss = student_output_seq["pred_keypoints_2d"].new_zeros(())

    signal_map = {
        "temporal_camera": normalize_camera_sequence(
            student_output_seq["pred_cam_t_full"],
            student_output_seq["scaled_focal_length"],
        ),
        "temporal_bbox_projected": bbox_sequence_from_keypoints(
            student_output_seq["pred_keypoints_2d"],
            window_batch,
        ),
        "temporal_bbox_input": input_bbox_sequence(window_batch),
    }

    for family_name, signal in signal_map.items():
        family_cfg = loss_cfg.get(family_name, {})
        if not family_cfg.get("enabled", False):
            continue

        residual = compute_second_difference(signal)
        base_loss = reduce_temporal_residual(
            residual,
            reduction=str(temporal_cfg["reduction"]),
        )
        score_loss = base_loss.new_zeros(())
        scorer_weight = float(family_cfg.get("scorer_weight", 0.0))
        if scorer is not None and scorer_weight > 0.0:
            score_loss = scorer.score_family(family_name, residual).mean()

        total_loss = total_loss + float(family_cfg["weight"]) * base_loss
        total_loss = total_loss + scorer_weight * score_loss
        metrics[f"loss_{family_name}_base"] = base_loss
        metrics[f"loss_{family_name}_score"] = score_loss

    return total_loss, metrics

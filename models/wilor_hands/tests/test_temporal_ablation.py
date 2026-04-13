import tempfile
import unittest
from pathlib import Path
import sys

import torch
import yaml


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from experiment_config import resolve_experiment_config
from temporal_losses import (
    TemporalWindowScorer,
    bbox_sequence_from_keypoints,
    build_temporal_windows,
    compute_second_difference,
    input_bbox_sequence,
    normalize_camera_sequence,
    reduce_temporal_residual,
)


class TemporalAblationTests(unittest.TestCase):
    def test_resolve_experiment_merges_defaults(self) -> None:
        payload = {
            "defaults": {
                "train_mode": "distill",
                "optimizer": {"lr": 1.0e-5, "weight_decay": 1.0e-4},
                "temporal": {"window_size": 8},
            },
            "experiments": [
                {
                    "name": "camera_only",
                    "losses": {
                        "temporal_camera": {
                            "enabled": True,
                            "weight": 0.02,
                            "scorer_weight": 0.001,
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "experiments.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
            resolved = resolve_experiment_config(config_path, "camera_only")

        self.assertEqual(resolved["name"], "camera_only")
        self.assertEqual(resolved["train_mode"], "distill")
        self.assertEqual(resolved["temporal"]["window_size"], 8)
        self.assertTrue(resolved["losses"]["temporal_camera"]["enabled"])
        self.assertAlmostEqual(resolved["losses"]["temporal_camera"]["weight"], 0.02)
        self.assertTrue(resolved["losses"]["vipe_camera"]["enabled"])

    def test_build_temporal_windows_respects_streams_and_gaps(self) -> None:
        samples = [
            {"video_name": "clip_a", "right": 1.0, "det_idx": 0, "frame_idx": 0},
            {"video_name": "clip_a", "right": 1.0, "det_idx": 0, "frame_idx": 1},
            {"video_name": "clip_a", "right": 1.0, "det_idx": 0, "frame_idx": 2},
            {"video_name": "clip_a", "right": 1.0, "det_idx": 0, "frame_idx": 4},
            {"video_name": "clip_a", "right": 1.0, "det_idx": 0, "frame_idx": 5},
            {"video_name": "clip_a", "right": 0.0, "det_idx": 0, "frame_idx": 0},
            {"video_name": "clip_a", "right": 0.0, "det_idx": 0, "frame_idx": 1},
            {"video_name": "clip_a", "right": 0.0, "det_idx": 0, "frame_idx": 2},
        ]
        windows, stats = build_temporal_windows(
            samples,
            window_size=3,
            window_stride=1,
            max_frame_gap=1,
        )

        self.assertEqual(len(windows), 2)
        self.assertEqual(stats["stream_count"], 2)
        self.assertGreaterEqual(stats["dropped_window_count"], 1)

    def test_temporal_residual_is_zero_for_linear_sequence(self) -> None:
        linear = torch.tensor(
            [[[0.0], [1.0], [2.0], [3.0], [4.0]]],
            dtype=torch.float32,
        )
        residual = compute_second_difference(linear)
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual)))
        self.assertAlmostEqual(float(reduce_temporal_residual(residual, "smooth_l1")), 0.0)

    def test_temporal_residual_is_positive_for_oscillation(self) -> None:
        oscillating = torch.tensor(
            [[[0.0], [1.0], [0.0], [1.0], [0.0]]],
            dtype=torch.float32,
        )
        residual = compute_second_difference(oscillating)
        self.assertGreater(float(reduce_temporal_residual(residual, "smooth_l1")), 0.0)

    def test_projected_and_input_bbox_shapes(self) -> None:
        batch = {
            "right": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "box_size": torch.tensor([[100.0, 100.0]], dtype=torch.float32),
            "box_center": torch.tensor([[[50.0, 60.0], [52.0, 62.0]]], dtype=torch.float32),
            "img_size": torch.tensor([[[200.0, 200.0], [200.0, 200.0]]], dtype=torch.float32),
        }
        pred_keypoints_2d = torch.tensor(
            [[
                [[0.1, 0.2], [0.2, 0.4], [0.3, 0.5]],
                [[0.1, 0.2], [0.2, 0.4], [0.3, 0.5]],
            ]],
            dtype=torch.float32,
        )

        projected_bbox = bbox_sequence_from_keypoints(pred_keypoints_2d, batch)
        detector_bbox = input_bbox_sequence(batch)
        self.assertEqual(projected_bbox.shape, (1, 2, 4))
        self.assertEqual(detector_bbox.shape, (1, 2, 4))
        self.assertTrue(torch.all(projected_bbox >= 0.0))

    def test_camera_normalization_shape(self) -> None:
        pred_cam_t = torch.tensor(
            [[[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]],
            dtype=torch.float32,
        )
        focal = torch.tensor(
            [[[100.0, 100.0], [120.0, 120.0]]],
            dtype=torch.float32,
        )
        normalized = normalize_camera_sequence(pred_cam_t, focal)
        self.assertEqual(normalized.shape, pred_cam_t.shape)
        self.assertTrue(torch.all(normalized.abs() < pred_cam_t.abs()))

    def test_temporal_scorer_is_non_negative_and_backpropagates(self) -> None:
        scorer = TemporalWindowScorer(hidden_dim=8, layers=1, dropout=0.0)
        residual = torch.randn(2, 6, 3, requires_grad=True)
        scores = scorer.score_family("temporal_camera", residual)
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(torch.all(scores >= 0.0))
        scores.mean().backward()
        self.assertIsNotNone(residual.grad)


if __name__ == "__main__":
    unittest.main()

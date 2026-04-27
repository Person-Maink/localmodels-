import tempfile
import unittest
from pathlib import Path
import sys

import torch
import yaml


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from experiment_config import experiment_to_env_map, resolve_experiment_config
from finetune_wilor_common import apply_train_mode_for_scope, configure_trainable_scope
from temporal_losses import (
    TemporalViPECameraHead,
    TemporalWindowScorer,
    bbox_sequence_from_keypoints,
    build_temporal_windows,
    compute_temporal_loss_bundle,
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
        self.assertEqual(resolved["losses"]["temporal_camera"]["formulation"], "static")
        self.assertAlmostEqual(resolved["losses"]["temporal_camera"]["weight"], 0.02)
        self.assertTrue(resolved["losses"]["vipe_camera"]["enabled"])

    def test_env_map_exports_optimizer_values(self) -> None:
        resolved = {
            "train_mode": "distill",
            "train_scope": "refine_net",
            "validation_split": 0.2,
            "sample_limit": 256,
            "detection_conf": 0.3,
            "rescale_factor": 2.0,
            "batch_size": 8,
            "num_workers": 2,
            "max_steps": 200,
            "log_every": 10,
            "save_every": 100,
            "seed": 42,
            "optimizer": {"lr": 1.0e-5, "weight_decay": 1.0e-4},
            "all_videos": False,
            "videos": ["clip_a", "clip_b"],
            "lora": {
                "enabled": False,
                "rank": 8,
                "alpha": 16.0,
                "dropout": 0.0,
                "block_start": 24,
                "block_end": 32,
                "target_modules": ["qkv"],
            },
            "temporal": {
                "window_size": 3,
                "window_stride": 2,
                "max_frame_gap": 1,
                "reduction": "smooth_l1",
                "scorer_hidden_dim": 64,
                "scorer_layers": 2,
                "scorer_dropout": 0.0,
            },
            "losses": {
                "vipe_camera": {
                    "enabled": True,
                    "weight": 0.005,
                    "scorer_weight": 0.0,
                },
                "temporal_camera": {
                    "enabled": True,
                    "formulation": "learnable",
                    "weight": 0.03,
                    "scorer_weight": 0.001,
                },
                "temporal_bbox_projected": {
                    "enabled": True,
                    "formulation": "learnable",
                    "weight": 0.03,
                    "scorer_weight": 0.001,
                },
                "temporal_vipe_camera": {
                    "enabled": True,
                    "formulation": "learnable",
                    "weight": 0.02,
                    "scorer_weight": 0.0,
                    "smoothness_weight": 0.01,
                    "anchor_weight": 0.001,
                },
            },
        }

        env_map = experiment_to_env_map(resolved)

        self.assertEqual(env_map["LR"], "1e-05")
        self.assertEqual(env_map["WEIGHT_DECAY"], "0.0001")
        self.assertEqual(env_map["VIDEO_NAMES"], "clip_a|clip_b")
        self.assertEqual(env_map["TEMPORAL_VIPE_CAMERA_ENABLED"], "true")
        self.assertEqual(env_map["TEMPORAL_VIPE_CAMERA_SMOOTHNESS_WEIGHT"], "0.01")
        self.assertEqual(env_map["TEMPORAL_VIPE_CAMERA_ANCHOR_WEIGHT"], "0.001")

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

    def test_temporal_vipe_camera_head_backpropagates(self) -> None:
        head = TemporalViPECameraHead(hidden_dim=8, layers=1, dropout=0.0)
        window_batch = {
            "camera_target_t_full": torch.tensor(
                [
                    [[0.10, 0.20, 0.30], [0.15, 0.24, 0.34], [0.20, 0.28, 0.38], [0.25, 0.32, 0.42]],
                    [[0.05, 0.10, 0.20], [0.08, 0.13, 0.24], [0.11, 0.16, 0.28], [0.14, 0.19, 0.32]],
                ],
                dtype=torch.float32,
            ),
            "camera_target_valid": torch.ones(2, 4, dtype=torch.float32),
            "camera_target_used_fallback": torch.zeros(2, 4, dtype=torch.float32),
        }
        student_output_seq = {
            "pred_cam_t_full": torch.tensor(
                [
                    [[0.12, 0.19, 0.35], [0.17, 0.23, 0.39], [0.23, 0.30, 0.44], [0.29, 0.35, 0.48]],
                    [[0.04, 0.11, 0.19], [0.10, 0.15, 0.25], [0.13, 0.18, 0.30], [0.16, 0.22, 0.35]],
                ],
                dtype=torch.float32,
            ),
            "scaled_focal_length": torch.full((2, 4, 2), 100.0, dtype=torch.float32),
        }
        loss_cfg = {
            "vipe_camera": {"enabled": False, "weight": 0.0, "scorer_weight": 0.0},
            "temporal_camera": {
                "enabled": False,
                "formulation": "static",
                "weight": 0.0,
                "scorer_weight": 0.0,
            },
            "temporal_bbox_projected": {
                "enabled": False,
                "formulation": "static",
                "weight": 0.0,
                "scorer_weight": 0.0,
            },
            "temporal_vipe_camera": {
                "enabled": True,
                "formulation": "learnable",
                "weight": 1.0,
                "scorer_weight": 0.0,
                "smoothness_weight": 0.1,
                "anchor_weight": 0.01,
            },
        }
        temporal_cfg = {"reduction": "smooth_l1"}

        loss, metrics = compute_temporal_loss_bundle(
            window_batch,
            student_output_seq,
            loss_cfg,
            temporal_cfg,
            scorer=None,
            vipe_camera_head=head,
        )

        self.assertGreater(float(loss.detach().item()), 0.0)
        self.assertIn("loss_temporal_vipe_camera_align", metrics)
        self.assertIn("loss_temporal_vipe_camera_smooth", metrics)
        loss.backward()
        self.assertTrue(
            any(param.grad is not None for param in head.parameters() if param.requires_grad)
        )

    def test_temporal_only_scope_freezes_wilor_modules(self) -> None:
        class _TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.cam_emb = torch.nn.Linear(3, 4)
                self.backbone.deccam = torch.nn.Linear(4, 3)
                self.refine_net = torch.nn.Module()
                self.refine_net.dec_cam = torch.nn.Linear(4, 3)
                self.refine_net.other = torch.nn.Linear(4, 4)
                self.discriminator = torch.nn.Linear(4, 1)

        model = _TinyModel()
        trainable = configure_trainable_scope(model, "temporal_only")

        self.assertEqual(trainable, [])
        self.assertFalse(any(param.requires_grad for param in model.parameters()))

        apply_train_mode_for_scope(model, "temporal_only")
        self.assertFalse(model.training)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.discriminator.training)


if __name__ == "__main__":
    unittest.main()

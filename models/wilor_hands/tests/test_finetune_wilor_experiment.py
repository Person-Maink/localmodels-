import sys
import tempfile
import unittest
from pathlib import Path

import yaml


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from finetune_wilor_cli import make_argparser
from finetune_wilor_experiment import resolve_runtime_experiment


class FinetuneWiLoRExperimentTests(unittest.TestCase):
    def test_parser_rejects_removed_train_scopes_and_temporal_vipe_flags(self) -> None:
        parser = make_argparser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--train_scope", "camera_head"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--train_scope", "full"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--temporal_vipe_camera_enabled"])

    def test_resolve_runtime_experiment_applies_config_defaults(self) -> None:
        payload = {
            "defaults": {
                "train_scope": "temporal_only",
                "validation_split": 0.25,
                "videos": ["clip_b", "clip_a"],
                "optimizer": {
                    "lr": 2.0e-5,
                    "weight_decay": 2.0e-4,
                },
                "temporal": {
                    "window_size": 3,
                    "window_stride": 2,
                },
                "lora": {
                    "enabled": True,
                    "rank": 4,
                    "target_modules": ["qkv", "proj"],
                },
            },
            "experiments": [
                {
                    "name": "exp_defaults",
                    "sample_limit": 64,
                    "losses": {
                        "vipe_camera": {
                            "enabled": True,
                            "weight": 0.05,
                        }
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "experiments.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            parser = make_argparser()
            args = parser.parse_args(
                [
                    "--loss_config",
                    str(config_path),
                    "--experiment_name",
                    "exp_defaults",
                ]
            )

            _, resolved = resolve_runtime_experiment(args, parser)

        self.assertEqual(args.train_scope, "temporal_only")
        self.assertEqual(args.validation_split, 0.25)
        self.assertEqual(args.sample_limit, 64)
        self.assertEqual(args.temporal_window_size, 3)
        self.assertEqual(args.temporal_window_stride, 2)
        self.assertEqual(args.lora_target_modules, "qkv,proj")
        self.assertCountEqual(args.videos, ["clip_a", "clip_b"])
        self.assertEqual(resolved["name"], "exp_defaults")
        self.assertEqual(resolved["sample_limit"], 64)
        self.assertEqual(resolved["optimizer"]["lr"], 2.0e-5)
        self.assertEqual(resolved["lora"]["rank"], 4)
        self.assertEqual(resolved["lora"]["target_modules"], ["qkv", "proj"])
        self.assertEqual(resolved["videos"], ["clip_a", "clip_b"])
        self.assertAlmostEqual(resolved["losses"]["vipe_camera"]["weight"], 0.05)

    def test_resolve_runtime_experiment_preserves_cli_overrides(self) -> None:
        payload = {
            "defaults": {
                "batch_size": 8,
                "videos": ["clip_a"],
                "optimizer": {
                    "lr": 1.0e-5,
                    "weight_decay": 1.0e-4,
                },
                "lora": {
                    "enabled": True,
                    "rank": 8,
                    "target_modules": ["qkv"],
                },
                "losses": {
                    "vipe_camera": {
                        "enabled": True,
                        "weight": 0.01,
                    }
                },
            },
            "experiments": [
                {
                    "name": "exp_cli_override",
                    "sample_limit": 32,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "experiments.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            parser = make_argparser()
            args = parser.parse_args(
                [
                    "--loss_config",
                    str(config_path),
                    "--experiment_name",
                    "exp_cli_override",
                    "--batch_size",
                    "2",
                    "--video",
                    "manual_clip",
                    "--vipe_camera_weight",
                    "0.9",
                    "--no-lora_enabled",
                ]
            )

            _, resolved = resolve_runtime_experiment(args, parser)

        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.videos, ["manual_clip"])
        self.assertFalse(args.lora_enabled)
        self.assertEqual(resolved["batch_size"], 2)
        self.assertEqual(resolved["videos"], ["manual_clip"])
        self.assertFalse(resolved["lora"]["enabled"])
        self.assertAlmostEqual(resolved["losses"]["vipe_camera"]["weight"], 0.9)
        self.assertEqual(resolved["optimizer"]["lr"], 1.0e-5)


if __name__ == "__main__":
    unittest.main()

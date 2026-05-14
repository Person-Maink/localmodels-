import sys
import tempfile
import unittest
from pathlib import Path

import yaml


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from stride_config import MODULE_DIR, StrideHMPConfig, StrideSimpleConfig, default_stride_config_path, load_stride_config


class StrideConfigTests(unittest.TestCase):
    def test_load_hmp_config_resolves_module_relative_paths(self) -> None:
        payload = {
            "backend": "hmp",
            "runtime": {
                "use_gpu": True,
                "visualize": False,
                "mano_model_path": "./mano_data",
                "hmp_assets_root": "./_DATA/hmp_model",
                "hmp_model_config": "./hmp_clean/hmp_model_config.yaml",
            },
            "refinement": {
                "overlap": 16,
                "pose": {
                    "iters": 200,
                    "lr": 0.05,
                    "weights": {
                        "rot": 1.0,
                        "pos": 0.25,
                        "root": 0.5,
                        "latent": 0.001,
                    },
                },
                "beta": {
                    "optimize": True,
                    "iters": 200,
                    "lr": 0.05,
                    "weights": {
                        "joint": 1.0,
                        "prior": 0.1,
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "hmp.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            config = load_stride_config(config_path, "hmp")

        self.assertIsInstance(config, StrideHMPConfig)
        self.assertEqual(config.runtime.mano_model_path, (MODULE_DIR / "mano_data").resolve())
        self.assertEqual(config.hmp_assets_root, (MODULE_DIR / "_DATA/hmp_model").resolve())
        self.assertEqual(config.hmp_model_config_path, (MODULE_DIR / "hmp_clean" / "hmp_model_config.yaml").resolve())

    def test_load_simple_default_config(self) -> None:
        config = load_stride_config(None, "simple")

        self.assertIsInstance(config, StrideSimpleConfig)
        self.assertEqual(config.config_path, default_stride_config_path("simple"))
        self.assertEqual(config.backend, "simple")

    def test_load_hmp_default_config(self) -> None:
        config = load_stride_config(None, "hmp")

        self.assertIsInstance(config, StrideHMPConfig)
        self.assertEqual(config.config_path, default_stride_config_path("hmp"))
        self.assertEqual(config.backend, "hmp")

    def test_missing_required_key_raises_readable_error(self) -> None:
        payload = {
            "backend": "hmp",
            "runtime": {
                "use_gpu": True,
                "visualize": False,
                "mano_model_path": "./mano_data",
                "hmp_assets_root": "./_DATA/hmp_model",
                "hmp_model_config": "./hmp_clean/hmp_model_config.yaml",
            },
            "refinement": {
                "overlap": 16,
                "pose": {
                    "iters": 200,
                    "lr": 0.05,
                    "weights": {
                        "rot": 1.0,
                        "pos": 0.25,
                        "latent": 0.001,
                    },
                },
                "beta": {
                    "optimize": True,
                    "iters": 200,
                    "lr": 0.05,
                    "weights": {
                        "joint": 1.0,
                        "prior": 0.1,
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "broken_hmp.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            with self.assertRaisesRegex(ValueError, "refinement\\.pose\\.weights\\.root"):
                load_stride_config(config_path, "hmp")

    def test_negative_overlap_raises_readable_error(self) -> None:
        payload = {
            "backend": "hmp",
            "runtime": {
                "use_gpu": True,
                "visualize": False,
                "mano_model_path": "./mano_data",
                "hmp_assets_root": "./_DATA/hmp_model",
                "hmp_model_config": "./hmp_clean/hmp_model_config.yaml",
            },
            "refinement": {
                "overlap": -1,
                "pose": {
                    "iters": 200,
                    "lr": 0.05,
                    "weights": {
                        "rot": 1.0,
                        "pos": 0.25,
                        "root": 0.5,
                        "latent": 0.001,
                    },
                },
                "beta": {
                    "optimize": True,
                    "iters": 200,
                    "lr": 0.05,
                    "weights": {
                        "joint": 1.0,
                        "prior": 0.1,
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "negative_overlap.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            with self.assertRaisesRegex(ValueError, "refinement\\.overlap must be >= 0"):
                load_stride_config(config_path, "hmp")


if __name__ == "__main__":
    unittest.main()

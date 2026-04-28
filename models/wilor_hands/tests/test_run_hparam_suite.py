import sys
import tempfile
import unittest
from pathlib import Path

import yaml


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from run_hparam_suite import (
    SetupDefinition,
    apply_winner_to_next_stage,
    discover_setups,
    initialize_run_workspace,
)


class RunHparamSuiteTests(unittest.TestCase):
    def test_discover_setups_is_subfolder_based_and_ignores_run_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = Path(tmpdir) / "experiments"
            (experiments_root / "lora").mkdir(parents=True)
            (experiments_root / "frozen wilor").mkdir(parents=True)
            (experiments_root / "lora" / "run").mkdir(parents=True)

            self._write_stage_yaml(experiments_root / "hparam_stage_a_windows.yaml", ["main_a"])
            self._write_stage_yaml(experiments_root / "lora" / "hparam_stage_b_optimizer.yaml", ["lora_b"])
            self._write_stage_yaml(
                experiments_root / "frozen wilor" / "hparam_stage_c_scorer_weights.yaml",
                ["frozen_c"],
            )
            self._write_stage_yaml(
                experiments_root / "lora" / "run" / "hparam_stage_z_should_ignore.yaml",
                ["ignored"],
            )

            setups = discover_setups(experiments_root)

        self.assertEqual([setup.name for setup in setups], ["main", "frozen wilor", "lora"])
        self.assertEqual(
            [setup.run_dir.name for setup in setups],
            ["run", "run", "run"],
        )

    def test_initialize_run_workspace_creates_setup_local_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir) / "experiments" / "frozen lora"
            setup_dir.mkdir(parents=True)
            source_stage = setup_dir / "hparam_stage_a_windows.yaml"
            self._write_stage_yaml(source_stage, ["hp_a"])

            setup = SetupDefinition(
                name="frozen lora",
                relative_path=Path("frozen lora"),
                setup_dir=setup_dir,
                run_dir=setup_dir / "run",
                output_root=Path(tmpdir) / "outputs",
            )
            copied_paths = initialize_run_workspace(setup, start_over=True)

            self.assertEqual(copied_paths, [setup.run_dir / source_stage.name])
            self.assertTrue((setup.run_dir / "hparam_stage_a_windows.yaml").exists())

    def test_initialize_run_workspace_preserves_existing_run_copy_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir) / "experiments" / "lora"
            setup_dir.mkdir(parents=True)
            source_stage = setup_dir / "hparam_stage_a_windows.yaml"
            self._write_stage_yaml(source_stage, ["hp_a"])

            setup = SetupDefinition(
                name="lora",
                relative_path=Path("lora"),
                setup_dir=setup_dir,
                run_dir=setup_dir / "run",
                output_root=Path(tmpdir) / "outputs",
            )
            initialize_run_workspace(setup, start_over=True)
            run_copy = setup.run_dir / source_stage.name
            payload = self._read_yaml(run_copy)
            payload["defaults"] = {"optimizer": {"lr": 3.0e-5}}
            with open(run_copy, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            initialize_run_workspace(setup, start_over=False)
            preserved_payload = self._read_yaml(run_copy)

        self.assertEqual(preserved_payload["defaults"]["optimizer"]["lr"], 3.0e-5)

    def test_apply_winner_to_next_stage_uses_winner_as_base_and_preserves_stage_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            winner_path = root / "winner.yaml"
            next_stage_path = root / "hparam_stage_b_optimizer.yaml"
            with open(winner_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "train_scope": "refine_net",
                        "optimizer": {"lr": 1.0e-5, "weight_decay": 1.0e-4},
                        "temporal": {"window_size": 3, "window_stride": 2},
                    },
                    handle,
                    sort_keys=False,
                )
            with open(next_stage_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "defaults": {
                            "sample_limit": 0,
                            "optimizer": {"weight_decay": 1.0e-3},
                        },
                        "experiments": [{"name": "hp_b_lr_3e5", "optimizer": {"lr": 3.0e-5}}],
                    },
                    handle,
                    sort_keys=False,
                )

            apply_winner_to_next_stage(next_stage_path, winner_path)
            merged = self._read_yaml(next_stage_path)

        self.assertEqual(merged["defaults"]["train_scope"], "refine_net")
        self.assertEqual(merged["defaults"]["temporal"]["window_size"], 3)
        self.assertEqual(merged["defaults"]["sample_limit"], 0)
        self.assertEqual(merged["defaults"]["optimizer"]["lr"], 1.0e-5)
        self.assertEqual(merged["defaults"]["optimizer"]["weight_decay"], 1.0e-3)
        self.assertEqual(merged["experiments"][0]["name"], "hp_b_lr_3e5")

    @staticmethod
    def _write_stage_yaml(path: Path, run_names: list[str]) -> None:
        payload = {
            "defaults": {},
            "experiments": [{"name": run_name} for run_name in run_names],
        }
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    unittest.main()

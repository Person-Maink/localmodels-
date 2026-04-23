import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from plot_finetune_losses import (
    build_stage_results,
    extract_validation_series,
    generate_all_figures,
    select_best_run,
)


class PlotFinetuneLossesTests(unittest.TestCase):
    def test_extract_validation_series_sorts_and_filters_rows(self) -> None:
        rows = [
            {"step": 10, "split": "train", "loss_total": 4.0},
            {"step": 20, "split": "val", "loss_total": 1.2},
            {"step": 10, "split": "val", "loss_total": 1.5},
            {"step": 30, "split": "val", "loss_total": 1.1},
        ]

        steps, losses = extract_validation_series(rows)

        self.assertEqual(steps, [10, 20, 30])
        self.assertEqual(losses, [1.5, 1.2, 1.1])

    def test_build_stage_results_uses_yaml_defined_stage_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            runs_root = root / "runs"
            experiments_root.mkdir()
            runs_root.mkdir()

            self._write_stage_yaml(
                experiments_root / "hparam_stage_a_windows.yaml",
                ["hp_a_ws3_s2_b8"],
            )
            self._write_stage_yaml(
                experiments_root / "hparam_stage_e_tuning.yaml",
                ["tune_stage_5_videos"],
            )
            self._write_run(
                runs_root / "hp_a_ws3_s2_b8",
                [{"step": 10, "split": "val", "loss_total": 1.0}],
            )
            self._write_run(
                runs_root / "tune_stage_5_videos",
                [{"step": 25, "split": "val", "loss_total": 1.7}],
            )

            stage_results = build_stage_results(experiments_root, runs_root)

        self.assertEqual([stage.definition.stage for stage in stage_results], ["a", "e"])
        self.assertEqual(stage_results[0].runs[0].name, "hp_a_ws3_s2_b8")
        self.assertEqual(stage_results[1].runs[0].name, "tune_stage_5_videos")

    def test_select_best_run_uses_lowest_best_validation_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            runs_root = root / "runs"
            experiments_root.mkdir()
            runs_root.mkdir()

            self._write_stage_yaml(
                experiments_root / "hparam_stage_b_optimizer.yaml",
                ["hp_b_lr_1e5", "hp_b_lr_3e5", "hp_b_lr_3e6"],
            )
            self._write_run(
                runs_root / "hp_b_lr_1e5",
                [
                    {"step": 10, "split": "val", "loss_total": 1.4},
                    {"step": 20, "split": "val", "loss_total": 1.1},
                ],
            )
            self._write_run(
                runs_root / "hp_b_lr_3e5",
                [
                    {"step": 10, "split": "val", "loss_total": 1.3},
                    {"step": 20, "split": "val", "loss_total": 0.9},
                ],
            )
            self._write_run(
                runs_root / "hp_b_lr_3e6",
                [
                    {"step": 10, "split": "val", "loss_total": 1.0},
                    {"step": 20, "split": "val", "loss_total": 0.95},
                ],
            )

            stage_result = build_stage_results(experiments_root, runs_root)[0]
            best_run = select_best_run(stage_result)

        self.assertIsNotNone(best_run)
        self.assertEqual(best_run.name, "hp_b_lr_3e5")
        self.assertEqual(best_run.best_validation, (20, 0.9))

    def test_generate_all_figures_creates_per_run_and_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            runs_root = root / "runs"
            experiments_root.mkdir()
            runs_root.mkdir()

            self._write_stage_yaml(
                experiments_root / "hparam_stage_a_windows.yaml",
                ["hp_a_ws3_s2_b8", "hp_a_ws8_s4_b2"],
            )
            self._write_run(
                runs_root / "hp_a_ws3_s2_b8",
                [
                    {"step": 10, "split": "train", "loss_total": 3.8},
                    {"step": 10, "split": "val", "loss_total": 1.2},
                    {"step": 20, "split": "val", "loss_total": 1.0},
                ],
                resolved_config={
                    "batch_size": 8,
                    "train_scope": "refine_net",
                    "optimizer": {"lr": 1.0e-5, "weight_decay": 1.0e-4},
                    "temporal": {"window_size": 3, "window_stride": 2},
                },
            )
            self._write_run(
                runs_root / "hp_a_ws8_s4_b2",
                [{"step": 10, "split": "train", "loss_total": 2.5}],
                resolved_config={
                    "batch_size": 2,
                    "train_scope": "refine_net",
                    "optimizer": {"lr": 1.0e-5, "weight_decay": 1.0e-4},
                    "temporal": {"window_size": 8, "window_stride": 4},
                },
            )

            outputs = generate_all_figures(experiments_root, runs_root)

            run_plot = runs_root / "hp_a_ws3_s2_b8" / "plots" / "validation_loss_vs_step.svg"
            placeholder_plot = runs_root / "hp_a_ws8_s4_b2" / "plots" / "validation_loss_vs_step.svg"
            stage_plot = runs_root / "reports" / "stage_a_validation_comparison.svg"

            self.assertEqual(len(outputs), 1)
            self.assertTrue(run_plot.exists())
            self.assertTrue(placeholder_plot.exists())
            self.assertTrue(stage_plot.exists())

    @staticmethod
    def _write_stage_yaml(path: Path, run_names: list[str]) -> None:
        payload = {
            "defaults": {},
            "experiments": [{"name": run_name, "run_name_suffix": run_name} for run_name in run_names],
        }
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    @staticmethod
    def _write_run(
        run_dir: Path,
        rows: list[dict[str, object]],
        resolved_config: dict[str, object] | None = None,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        if resolved_config is not None:
            with open(run_dir / "resolved_experiment.yaml", "w", encoding="utf-8") as handle:
                yaml.safe_dump(resolved_config, handle, sort_keys=False)


if __name__ == "__main__":
    unittest.main()

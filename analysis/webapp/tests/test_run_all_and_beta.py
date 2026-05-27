import importlib.util
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_ALL = _load_module("run_all_test_module", "Frequency Analysis/Run All.py")
BETA_COMPARISON = _load_module("beta_comparison_test_module", "Frequency Analysis/beta comparison.py")


def _make_source(family: str, kind: str = "model", name: Optional[str] = None):
    clip_name = name or f"{family}_clip"
    return RUN_ALL.SourceItem(
        family=family,
        kind=kind,
        clip_id=clip_name,
        display_id=clip_name,
        match_id=clip_name,
        path=f"/tmp/{clip_name}",
        is_comp=False,
    )


def _make_frames(scale: float = 1.0, offset: float = 0.0):
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    frames = []
    for frame_id in range(12):
        signal = np.sin(frame_id * 0.45)
        verts = base.copy()
        verts[:, 0] += scale * signal
        verts[:, 1] += offset
        verts[2:, 2] += 0.1 * signal
        frames.append((frame_id, [{"verts": verts, "right": 1, "score": 1.0}]))
    return frames


class RunAllSelectionTests(unittest.TestCase):
    def test_single_source_compare_and_beta_scenario_jobs_are_enabled(self):
        item = RUN_ALL.SingleSourceItem(
            scenario_id="single_source_wilor",
            source=_make_source("wilor"),
            source_item_id="single_wilor_clip",
        )

        analysis_ids = set(RUN_ALL._analysis_ids_for_item("single_source", item))

        self.assertIn("compare", analysis_ids)
        self.assertIn("beta_model_compare", analysis_ids)
        self.assertIn("beta_model_compare_raw_plus_beta", analysis_ids)
        self.assertNotIn("point_to_point", analysis_ids)

    def test_beta_scenario_jobs_skip_non_beta_families(self):
        item = RUN_ALL.PairItem(
            scenario_id="wilor_vs_hamba",
            source_a=_make_source("wilor"),
            source_b=_make_source("hamba"),
            pair_id="pair_wilor_hamba",
        )

        analysis_ids = set(RUN_ALL._analysis_ids_for_item("pair", item))

        self.assertNotIn("beta_model_compare", analysis_ids)
        self.assertNotIn("beta_model_compare_raw_plus_beta", analysis_ids)
        self.assertIn("compare", analysis_ids)

    def test_beta_scenario_jobs_enable_for_beta_compatible_multi_model_items(self):
        item = RUN_ALL.MultiModelsItem(
            scenario_id="wilor_vs_dynhamr_vs_stride",
            sources=(_make_source("wilor"), _make_source("dynhamr"), _make_source("stride")),
            group_id="group_beta_models",
        )

        analysis_ids = set(RUN_ALL._analysis_ids_for_item("multi_model", item))

        self.assertIn("beta_model_compare", analysis_ids)
        self.assertIn("beta_model_compare_raw_plus_beta", analysis_ids)


class BetaComparisonTests(unittest.TestCase):
    def test_load_variant_frame_sets_keeps_wrist_ground_disabled(self):
        fake_actual_frames = [("frame", [])]
        fake_bundle = {"frames": [("frame", [])], "record_root": "/tmp/record_root"}
        fake_beta_module = mock.Mock()
        fake_beta_module.load_average_beta_frames_for_source.return_value = fake_bundle

        with mock.patch.object(BETA_COMPARISON, "_load_actual_model_frames", return_value=fake_actual_frames), mock.patch.object(
            BETA_COMPARISON,
            "_load_beta_average_module",
            return_value=fake_beta_module,
        ):
            actual_frames, beta_frames, bundle = BETA_COMPARISON._load_variant_frame_sets(
                "/tmp/source",
                mano_right_path="/tmp/MANO_RIGHT.pkl",
                hand_idx=1,
            )

        self.assertEqual(actual_frames, fake_actual_frames)
        self.assertEqual(beta_frames, fake_bundle["frames"])
        self.assertEqual(bundle, fake_bundle)
        fake_beta_module.load_average_beta_frames_for_source.assert_called_once_with(
            source_path="/tmp/source",
            mano_model_path="/tmp/MANO_RIGHT.pkl",
            wrist_ground=False,
            hand="right",
        )

    def test_single_source_compare_mode_preserves_two_variant_entries(self):
        fake_faces = np.asarray([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
        fake_j_reg = np.zeros((1, 5), dtype=np.float32)
        fake_frames = _make_frames()
        fake_beta_frames = _make_frames(scale=0.8, offset=0.1)
        fake_bundle = {"record_root": "/tmp/record_root"}

        with mock.patch.object(BETA_COMPARISON, "_resolve_mano_right_path", return_value=Path("/tmp/MANO_RIGHT.pkl")), mock.patch.object(
            BETA_COMPARISON,
            "_load_mano_assets",
            return_value=(fake_j_reg, fake_faces),
        ), mock.patch.object(
            BETA_COMPARISON,
            "_load_variant_frame_sets",
            return_value=(fake_frames, fake_beta_frames, fake_bundle),
        ):
            analysis_data = BETA_COMPARISON.run_beta_comparison_analysis(
                {
                    "source_path": "/tmp/source_a",
                    "vertex_a": 0,
                    "vertex_b": 1,
                    "n_neighbors": 1,
                    "hand_idx": 1,
                }
            )

        self.assertEqual(analysis_data["variant_mode"], "single_source_compare")
        self.assertEqual([entry["label"] for entry in analysis_data["entries"]], ["Actual model", "Beta average"])

    def test_beta_only_mode_returns_one_entry_per_source(self):
        fake_faces = np.asarray([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
        fake_j_reg = np.zeros((1, 5), dtype=np.float32)
        fake_frames = _make_frames()
        fake_beta_frames = _make_frames(scale=0.9, offset=0.05)
        fake_bundle = {"record_root": "/tmp/record_root"}

        with mock.patch.object(BETA_COMPARISON, "_resolve_mano_right_path", return_value=Path("/tmp/MANO_RIGHT.pkl")), mock.patch.object(
            BETA_COMPARISON,
            "_load_mano_assets",
            return_value=(fake_j_reg, fake_faces),
        ), mock.patch.object(
            BETA_COMPARISON,
            "_load_variant_frame_sets",
            return_value=(fake_frames, fake_beta_frames, fake_bundle),
        ):
            analysis_data = BETA_COMPARISON.run_beta_comparison_analysis(
                {
                    "sources": ["/tmp/source_a", "/tmp/source_b"],
                    "labels": ["Source A", "Source B"],
                    "variant_mode": "beta_only",
                    "vertex_a": 0,
                    "vertex_b": 1,
                    "n_neighbors": 1,
                    "hand_idx": 1,
                }
            )

        self.assertEqual([entry["label"] for entry in analysis_data["entries"]], ["Source A beta avg", "Source B beta avg"])

    def test_raw_plus_beta_mode_interleaves_variants_per_source(self):
        fake_faces = np.asarray([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
        fake_j_reg = np.zeros((1, 5), dtype=np.float32)
        fake_frames = _make_frames()
        fake_beta_frames = _make_frames(scale=0.9, offset=0.05)
        fake_bundle = {"record_root": "/tmp/record_root"}

        with mock.patch.object(BETA_COMPARISON, "_resolve_mano_right_path", return_value=Path("/tmp/MANO_RIGHT.pkl")), mock.patch.object(
            BETA_COMPARISON,
            "_load_mano_assets",
            return_value=(fake_j_reg, fake_faces),
        ), mock.patch.object(
            BETA_COMPARISON,
            "_load_variant_frame_sets",
            return_value=(fake_frames, fake_beta_frames, fake_bundle),
        ):
            analysis_data = BETA_COMPARISON.run_beta_comparison_analysis(
                {
                    "sources": ["/tmp/source_a", "/tmp/source_b"],
                    "labels": ["Source A", "Source B"],
                    "variant_mode": "raw_plus_beta",
                    "vertex_a": 0,
                    "vertex_b": 1,
                    "n_neighbors": 1,
                    "hand_idx": 1,
                }
            )

        self.assertEqual(
            [entry["label"] for entry in analysis_data["entries"]],
            ["Source A raw", "Source A beta avg", "Source B raw", "Source B beta avg"],
        )
        self.assertEqual(
            [entry["slot"] for entry in analysis_data["entries"]],
            ["A:raw", "A:beta", "B:raw", "B:beta"],
        )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METRIC_ANALYSIS = _load_module("metric_analysis_test_module", "Frequency Analysis/metric analysis.py")


def _make_chain_faces(total_verts: int) -> np.ndarray:
    faces = []
    for vertex_id in range(total_verts - 2):
        faces.append([vertex_id, vertex_id + 1, vertex_id + 2])
    return np.asarray(faces, dtype=np.int32)


class MetricAnalysisTests(unittest.TestCase):
    def test_parse_model_descriptor_recognizes_finetune_variant(self):
        key, label, clip_id = METRIC_ANALYSIS.parse_model_descriptor(
            "/tmp/outputs/wilor_finetune/main_learnable_finetnuing/demo_clip/meshes"
        )

        self.assertEqual(key, "wilor_finetune:main_learnable_finetnuing")
        self.assertEqual(label, "WiLoR Finetune Main Learnable")
        self.assertEqual(clip_id, "demo_clip")

    def test_parse_model_descriptor_recognizes_base_model(self):
        key, label, clip_id = METRIC_ANALYSIS.parse_model_descriptor("/tmp/outputs/hamba/clip_01/meshes")

        self.assertEqual(key, "hamba")
        self.assertEqual(label, "Hamba")
        self.assertEqual(clip_id, "clip_01")

    def test_is_me_clip_matches_me_variants(self):
        self.assertTrue(METRIC_ANALYSIS.is_me_clip("me 1"))
        self.assertTrue(METRIC_ANALYSIS.is_me_clip("me 4_amplified"))
        self.assertTrue(METRIC_ANALYSIS.is_me_clip("me 2_contrast_100"))
        self.assertFalse(METRIC_ANALYSIS.is_me_clip("120-2_clip_1"))

    def test_facing_region_metadata_uses_default_333_173_and_neighbors(self):
        faces = _make_chain_faces(400)
        with mock.patch.object(METRIC_ANALYSIS, "_load_facing_mano_faces", return_value=faces):
            region_a, region_b = METRIC_ANALYSIS._resolve_facing_region_metadata()

        self.assertEqual(int(region_a[0]), int(METRIC_ANALYSIS.CONFIG.FACING_MODEL_VERTEX_A))
        self.assertEqual(int(region_b[0]), int(METRIC_ANALYSIS.CONFIG.FACING_MODEL_VERTEX_B))
        self.assertEqual(len(region_a), int(METRIC_ANALYSIS.CONFIG.N_NEIGHBORS) + 1)
        self.assertEqual(len(region_b), int(METRIC_ANALYSIS.CONFIG.N_NEIGHBORS) + 1)

    def test_source_facing_skips_zero_vectors_and_normalizes_per_frame(self):
        frame_records = [
            (
                0,
                [
                    {
                        "right": 0,
                        "verts": np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
                        "cam_t": np.zeros(3, dtype=np.float32),
                        "verts_world": np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
                    }
                ],
            ),
            (
                1,
                [
                    {
                        "right": 0,
                        "verts": np.asarray([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
                        "cam_t": np.zeros(3, dtype=np.float32),
                        "verts_world": np.asarray([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
                    }
                ],
            ),
            (
                2,
                [
                    {
                        "right": 0,
                        "verts": np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
                        "cam_t": np.zeros(3, dtype=np.float32),
                        "verts_world": np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
                    }
                ],
            ),
        ]

        summary = METRIC_ANALYSIS._summarize_source_facing_from_frame_records(
            frame_records,
            region_a=np.asarray([0], dtype=np.int32),
            region_b=np.asarray([1], dtype=np.int32),
            hand_value=0,
        )

        self.assertEqual(summary["usable_frames"], 2)
        self.assertAlmostEqual(summary["facing_dir_x"], np.sqrt(0.5), places=6)
        self.assertAlmostEqual(summary["facing_dir_y"], np.sqrt(0.5), places=6)
        self.assertAlmostEqual(summary["facing_dir_z"], 0.0, places=6)
        self.assertAlmostEqual(summary["facing_consistency"], np.sqrt(0.5), places=6)
        self.assertAlmostEqual(summary["facing_angle_deg"], 90.0, places=6)

    def test_facing_aggregation_recomputes_direction_from_source_vectors(self):
        rows = [
            METRIC_ANALYSIS.FacingSourceRecord(
                model_key="wilor",
                model_label="WiLoR",
                clip_id="clip_a",
                source="/tmp/outputs/wilor/clip_a/meshes",
                hand_used="left",
                usable_frames=12,
                facing_dir_x=1.0,
                facing_dir_y=0.0,
                facing_dir_z=0.0,
                facing_angle_deg=90.0,
                facing_consistency=1.0,
            ),
            METRIC_ANALYSIS.FacingSourceRecord(
                model_key="wilor",
                model_label="WiLoR",
                clip_id="clip_b",
                source="/tmp/outputs/wilor/clip_b/meshes",
                hand_used="left",
                usable_frames=15,
                facing_dir_x=0.0,
                facing_dir_y=1.0,
                facing_dir_z=0.0,
                facing_angle_deg=90.0,
                facing_consistency=1.0,
            ),
        ]

        summary = METRIC_ANALYSIS._aggregate_facing_source_records(rows)

        self.assertAlmostEqual(summary["facing_dir_x"], np.sqrt(0.5), places=6)
        self.assertAlmostEqual(summary["facing_dir_y"], np.sqrt(0.5), places=6)
        self.assertAlmostEqual(summary["facing_dir_z"], 0.0, places=6)
        self.assertAlmostEqual(summary["facing_consistency"], np.sqrt(0.5), places=6)
        self.assertAlmostEqual(summary["facing_angle_deg"], 90.0, places=6)

    def test_summarize_by_model_balances_analysis_types(self):
        records = [
            METRIC_ANALYSIS.MetricRecord(
                model_key="wilor",
                model_label="WiLoR",
                analysis="compare",
                source="/tmp/outputs/wilor/clip_a/meshes",
                clip_id="clip_a",
                values={
                    "welch_peak_hz": 10.0,
                    "fft_peak_hz": 9.5,
                    "peak_ratio": 0.2,
                    "peak_sharpness": 2.0,
                    "temporal_noise": 1e-4,
                    "spatial_coherence": 0.9,
                    "rms_amplitude": 0.01,
                },
            ),
            METRIC_ANALYSIS.MetricRecord(
                model_key="wilor",
                model_label="WiLoR",
                analysis="compare",
                source="/tmp/outputs/wilor/clip_b/meshes",
                clip_id="clip_b",
                values={
                    "welch_peak_hz": 10.0,
                    "fft_peak_hz": 9.5,
                    "peak_ratio": 0.2,
                    "peak_sharpness": 2.0,
                    "temporal_noise": 1e-4,
                    "spatial_coherence": 0.9,
                    "rms_amplitude": 0.01,
                },
            ),
            METRIC_ANALYSIS.MetricRecord(
                model_key="wilor",
                model_label="WiLoR",
                analysis="neighbor_sweep",
                source="/tmp/outputs/wilor/clip_a/meshes",
                clip_id="clip_a",
                values={
                    "welch_peak_hz": 4.0,
                    "fft_peak_hz": 4.5,
                    "peak_ratio": 0.1,
                    "peak_sharpness": 1.0,
                    "temporal_noise": 2e-4,
                    "spatial_coherence": 0.8,
                    "rms_amplitude": 0.02,
                },
            ),
        ]

        balanced = METRIC_ANALYSIS.summarize_by_model(records, balance_analyses=True)
        raw_weighted = METRIC_ANALYSIS.summarize_by_model(records, balance_analyses=False)

        self.assertEqual(len(balanced), 1)
        self.assertAlmostEqual(balanced[0]["welch_peak_hz"], 7.0, places=6)
        self.assertAlmostEqual(raw_weighted[0]["welch_peak_hz"], 8.0, places=6)

    def test_summarize_me_vs_other_splits_model_rows(self):
        records = [
            METRIC_ANALYSIS.MetricRecord(
                model_key="wilor",
                model_label="WiLoR",
                analysis="compare",
                source="/tmp/outputs/wilor/me 1/meshes",
                clip_id="me 1",
                values={
                    "welch_peak_hz": 6.0,
                    "fft_peak_hz": 6.5,
                    "peak_ratio": 0.2,
                    "peak_sharpness": 1.5,
                    "temporal_noise": 1e-4,
                    "spatial_coherence": 0.9,
                    "rms_amplitude": 0.01,
                },
            ),
            METRIC_ANALYSIS.MetricRecord(
                model_key="wilor",
                model_label="WiLoR",
                analysis="compare",
                source="/tmp/outputs/wilor/clip_b/meshes",
                clip_id="clip_b",
                values={
                    "welch_peak_hz": 4.0,
                    "fft_peak_hz": 4.5,
                    "peak_ratio": 0.3,
                    "peak_sharpness": 1.7,
                    "temporal_noise": 2e-4,
                    "spatial_coherence": 0.8,
                    "rms_amplitude": 0.02,
                },
            ),
        ]
        facing_rows = [
            METRIC_ANALYSIS.FacingSourceRecord(
                model_key="wilor",
                model_label="WiLoR",
                clip_id="me 1",
                source="/tmp/outputs/wilor/me 1/meshes",
                hand_used="left",
                usable_frames=8,
                facing_dir_x=0.0,
                facing_dir_y=0.0,
                facing_dir_z=-1.0,
                facing_angle_deg=0.0,
                facing_consistency=0.9,
            ),
            METRIC_ANALYSIS.FacingSourceRecord(
                model_key="wilor",
                model_label="WiLoR",
                clip_id="clip_b",
                source="/tmp/outputs/wilor/clip_b/meshes",
                hand_used="left",
                usable_frames=8,
                facing_dir_x=1.0,
                facing_dir_y=0.0,
                facing_dir_z=0.0,
                facing_angle_deg=90.0,
                facing_consistency=0.8,
            ),
        ]

        rows = METRIC_ANALYSIS.summarize_me_vs_other(
            records,
            facing_source_records=facing_rows,
            balance_analyses=True,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "WiLoR")
        self.assertAlmostEqual(rows[0]["me_welch_peak_hz"], 6.0, places=6)
        self.assertAlmostEqual(rows[0]["other_welch_peak_hz"], 4.0, places=6)
        self.assertAlmostEqual(rows[0]["me_facing_angle_deg"], 0.0, places=6)
        self.assertAlmostEqual(rows[0]["other_facing_angle_deg"], 90.0, places=6)
        self.assertEqual(rows[0]["me_clips"], 1)
        self.assertEqual(rows[0]["other_clips"], 1)

    def test_main_outputs_summary_files(self):
        csv_text = """scenario,item_kind,item_id,analysis,status,slot,label,source,kind,dominant_hz,fft_peak_hz,peak_ratio,peak_sharpness,temporal_noise,spatial_coherence,rms_amplitude,num_samples
demo,pair,item1,compare,success,A,HAMBA:/clip,/tmp/outputs/hamba/clip_a/meshes,model,6.0,6.5,0.2,1.5,0.0001,0.9,0.01,120
demo,pair,item2,compare,success,B,WiLoR:/clip,/tmp/outputs/wilor/clip_b/meshes,model,5.0,5.5,0.3,1.8,0.0002,0.8,0.02,120
"""
        fake_facing = [
            METRIC_ANALYSIS.FacingSourceRecord(
                model_key="hamba",
                model_label="Hamba",
                clip_id="clip_a",
                source="/tmp/outputs/hamba/clip_a/meshes",
                hand_used="left",
                usable_frames=10,
                facing_dir_x=0.0,
                facing_dir_y=0.0,
                facing_dir_z=-1.0,
                facing_angle_deg=0.0,
                facing_consistency=0.95,
            ),
            METRIC_ANALYSIS.FacingSourceRecord(
                model_key="wilor",
                model_label="WiLoR",
                clip_id="clip_b",
                source="/tmp/outputs/wilor/clip_b/meshes",
                hand_used="left",
                usable_frames=12,
                facing_dir_x=1.0,
                facing_dir_y=0.0,
                facing_dir_z=0.0,
                facing_angle_deg=90.0,
                facing_consistency=0.80,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_csv = tmp_path / "metrics.csv"
            metrics_csv.write_text(csv_text, encoding="utf-8")
            output_dir = tmp_path / "metric_analysis"

            original_parse_args = METRIC_ANALYSIS._parse_args
            original_compute_facing = METRIC_ANALYSIS.compute_facing_source_records
            try:
                METRIC_ANALYSIS._parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "metrics_csv": metrics_csv,
                        "output_dir": output_dir,
                        "include_mediapipe": False,
                        "raw_row_weighting": False,
                    },
                )()
                METRIC_ANALYSIS.compute_facing_source_records = lambda records: list(fake_facing)
                result = METRIC_ANALYSIS.main()
            finally:
                METRIC_ANALYSIS._parse_args = original_parse_args
                METRIC_ANALYSIS.compute_facing_source_records = original_compute_facing

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "model_summary.csv").is_file())
            self.assertTrue((output_dir / "model_summary.md").is_file())
            self.assertTrue((output_dir / "model_summary_by_analysis.csv").is_file())
            self.assertTrue((output_dir / "model_summary_me_vs_other.csv").is_file())
            self.assertTrue((output_dir / "model_summary_me_vs_other.md").is_file())
            self.assertTrue((output_dir / "model_facing_by_source.csv").is_file())
            self.assertIn("facing_angle_deg", (output_dir / "model_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

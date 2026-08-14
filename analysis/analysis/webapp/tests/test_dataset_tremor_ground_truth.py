from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dataset_tremor_ground_truth import (
    attach_ground_truth,
    frequency_accuracy_summary,
    load_ground_truth,
    mean_ci95,
)


class DatasetTremorGroundTruthTests(unittest.TestCase):
    def _write_truth(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "truth.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loader_normalizes_filename_and_validates_numeric_frequency(self):
        path = self._write_truth(
            "video_filename,frequency_hz,frequency_ci95_lower_hz,frequency_ci95_upper_hz\n"
            "clip_a.mp4,4.5,4.2,4.8\n"
        )
        truth = load_ground_truth(path)
        self.assertEqual(truth.loc[0, "clip_id"], "clip_a")
        self.assertAlmostEqual(truth.loc[0, "ground_truth_hz"], 4.5)

    def test_loader_rejects_duplicate_and_non_numeric_frequency(self):
        duplicate = self._write_truth(
            "video_filename,frequency_hz,frequency_ci95_lower_hz,frequency_ci95_upper_hz\n"
            "clip_a.mp4,4,3,5\nclip_a.mp4,4,3,5\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_ground_truth(duplicate)
        non_numeric = self._write_truth(
            "video_filename,frequency_hz,frequency_ci95_lower_hz,frequency_ci95_upper_hz\n"
            "clip_a.mp4,nope,3,5\n"
        )
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            load_ground_truth(non_numeric)

    def test_attach_and_accuracy_uses_annotated_frequency(self):
        metrics = pd.DataFrame({"clip_id": ["clip_a", "clip_b"], "dominant_hz": [4.0, 6.0]})
        truth = pd.DataFrame(
            {
                "clip_id": ["clip_a", "clip_b"],
                "ground_truth_hz": [5.0, 5.0],
                "ground_truth_ci95_lower_hz": [4.8, 4.8],
                "ground_truth_ci95_upper_hz": [5.2, 5.2],
            }
        )
        merged = attach_ground_truth(metrics, truth)
        summary = frequency_accuracy_summary(merged, "wilor")
        self.assertEqual(summary["matched_clips"], 2)
        self.assertAlmostEqual(summary["mae_hz"], 1.0)
        self.assertTrue((merged["ground_truth_freq_error_hz"] == 1.0).all())

    def test_attach_requires_every_annotated_clip_and_ci_summary(self):
        metrics = pd.DataFrame({"clip_id": ["clip_a"], "dominant_hz": [4.0]})
        truth = pd.DataFrame(
            {
                "clip_id": ["clip_a", "clip_b"],
                "ground_truth_hz": [4.0, 6.0],
                "ground_truth_ci95_lower_hz": [3.8, 5.8],
                "ground_truth_ci95_upper_hz": [4.2, 6.2],
            }
        )
        with self.assertRaisesRegex(ValueError, "missing annotated"):
            attach_ground_truth(metrics, truth)
        mean, lower, upper = mean_ci95(pd.Series([4.0, 6.0]))
        self.assertAlmostEqual(mean, 5.0)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)


if __name__ == "__main__":
    unittest.main()

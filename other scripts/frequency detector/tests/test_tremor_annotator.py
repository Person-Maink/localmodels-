import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from tremor_annotator import calculate_statistics, run, save_annotation


class StatisticsTests(unittest.TestCase):
    def test_regular_peak_sequence(self):
        result = calculate_statistics([0.0, 0.5, 1.0, 1.5])
        self.assertEqual(result["peak_count"], 4)
        self.assertAlmostEqual(result["mean_interval_s"], 0.5)
        self.assertAlmostEqual(result["interval_std_s"], 0.0)
        self.assertAlmostEqual(result["frequency_hz"], 2.0)
        self.assertAlmostEqual(result["frequency_se_hz"], 0.0)
        self.assertAlmostEqual(result["frequency_ci95_lower_hz"], 2.0)
        self.assertAlmostEqual(result["frequency_ci95_upper_hz"], 2.0)

    def test_irregular_peak_sequence(self):
        result = calculate_statistics([0.0, 1.0, 3.0])
        self.assertAlmostEqual(result["mean_interval_s"], 1.5)
        self.assertAlmostEqual(result["interval_std_s"], math.sqrt(0.5))
        self.assertAlmostEqual(result["frequency_hz"], 2 / 3)
        self.assertGreater(result["frequency_se_hz"], 0)

    def test_two_peaks_have_no_variation_estimate(self):
        result = calculate_statistics([1.0, 1.5])
        self.assertAlmostEqual(result["frequency_hz"], 2.0)
        self.assertIsNone(result["interval_std_s"])
        self.assertIsNone(result["frequency_se_hz"])

    def test_invalid_timestamps(self):
        for timestamps in ([], [0.0], [0.0, 0.0], [1.0, 0.0]):
            with self.assertRaises(ValueError):
                calculate_statistics(timestamps)


class StorageTests(unittest.TestCase):
    def test_save_writes_json_and_replaces_csv_row(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            video = directory / "subject.mp4"
            video.touch()
            annotation_path = save_annotation(directory, video, [0.0, 0.5, 1.0], duration_s=3.0, fps=30.0)
            self.assertTrue(annotation_path.exists())
            detail = json.loads(annotation_path.read_text())
            self.assertEqual(detail["peak_timestamps_s"], [0.0, 0.5, 1.0])
            self.assertEqual(detail["annotation_window"], {"start_s": 0.0, "end_s": 3.0})

            save_annotation(directory, video, [0.0, 1.0, 2.0], duration_s=3.0, fps=30.0)
            with (directory / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["video_path"], "subject.mp4")
            self.assertEqual(rows[0]["frequency_hz"], "1.0")

    def test_missing_directory_returns_error(self):
        self.assertEqual(run(Path("/definitely/not/a/video/directory")), 2)


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from finetune_runtime_estimator import estimate_runtime


class FinetuneRuntimeEstimatorTests(unittest.TestCase):
    def test_all_videos_fallback_uses_longer_default(self) -> None:
        resolved = {
            "train_scope": "refine_net",
            "batch_size": 8,
            "max_steps": 10000,
            "sample_limit": 0,
            "log_every": 25,
            "all_videos": True,
            "videos": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            estimate = estimate_runtime(resolved, Path(tmpdir))

        self.assertEqual(estimate.time_str, "08:00:00")
        self.assertEqual(estimate.matched_runs, 0)

    def test_default_cap_allows_up_to_ten_hours(self) -> None:
        resolved = {
            "train_scope": "refine_net",
            "batch_size": 8,
            "max_steps": 1000,
            "sample_limit": 0,
            "log_every": 25,
            "all_videos": False,
            "videos": ["clip_a"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "wilor-train_001.out"
            log_path.write_text(
                "\n".join(
                    [
                        "Command: python -u train.py --train_scope refine_net --batch_size 8 --max_steps 1000 --sample_limit 0 --log_every 25 --video=clip_a",
                        "Finished fine-tuning.",
                        "Execution took 30.0 hours",
                    ]
                ),
                encoding="utf-8",
            )
            estimate = estimate_runtime(resolved, Path(tmpdir))

        self.assertEqual(estimate.time_str, "10:00:00")
        self.assertEqual(estimate.matched_runs, 1)


if __name__ == "__main__":
    unittest.main()

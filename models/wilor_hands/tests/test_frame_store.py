import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from frame_store import FrameStore, build_frame_caches


class FrameStoreTests(unittest.TestCase):
    def test_build_frame_cache_writes_sidecar_zip_and_store_reads_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "demo.avi"
            self._write_video(video_path)

            results = build_frame_caches(root)

            self.assertEqual(results[0]["status"], "built")
            self.assertTrue((root / "demo.frames.zip").is_file())
            self.assertTrue((root / "demo.frames.index.json").is_file())

            store = FrameStore(root)
            self.assertEqual(store.list_videos(), ["demo"])

            frames = list(store.iter_video_frames("demo"))
            self.assertEqual([frame.frame_name for frame in frames], ["frame_000000", "frame_000001"])
            self.assertEqual(frames[0].source_kind, "zip")

            decoded = store.get_frame("demo", 1)
            self.assertIsNotNone(decoded)
            self.assertEqual(tuple(decoded.shape[:2]), (8, 8))

    def test_loose_frames_and_single_images_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_dir = root / "legacy_frames"
            legacy_dir.mkdir()
            self._write_image(legacy_dir / "frame_000010.jpg", (255, 0, 0))
            self._write_image(root / "still.png", (0, 255, 0))

            store = FrameStore(root)

            self.assertEqual(store.list_videos(), ["legacy", "single_images"])
            legacy_frame = next(store.iter_video_frames("legacy"))
            self.assertEqual(legacy_frame.frame_id, 10)
            self.assertIsNotNone(store.get_frame("legacy", 10))
            self.assertEqual(store.get_frame_name("single_images", 0), "still")

    def test_invalid_cache_falls_back_to_loose_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "broken.avi").touch()
            legacy_dir = root / "broken_frames"
            legacy_dir.mkdir()
            self._write_image(legacy_dir / "frame_000000.jpg", (64, 64, 64))
            with zipfile.ZipFile(root / "broken.frames.zip", "w") as archive:
                archive.writestr("frame_000000.jpg", b"not-a-real-jpeg")

            store = FrameStore(root)

            frames = list(store.iter_video_frames("broken"))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].source_kind, "loose_frames")
            self.assertIsNotNone(store.get_frame("broken", 0))

    def test_build_frame_caches_skips_existing_cache_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "repeat.avi"
            self._write_video(video_path)

            first = build_frame_caches(root)
            second = build_frame_caches(root)

            self.assertEqual(first[0]["status"], "built")
            self.assertEqual(second[0]["status"], "skipped")

    def test_zip_cache_without_raw_video_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "cache_only.avi"
            self._write_video(video_path)
            build_frame_caches(root)
            video_path.unlink()

            store = FrameStore(root)

            self.assertEqual(store.list_videos(), ["cache_only"])
            self.assertEqual(store.get_source_path("cache_only"), root / "cache_only.frames.zip")
            self.assertIsNotNone(store.get_frame("cache_only", 0))

    def _write_video(self, path: Path) -> None:
        size = (8, 8)
        frames = [
            np.full((size[1], size[0], 3), (0, 0, 255), dtype=np.uint8),
            np.full((size[1], size[0], 3), (0, 255, 0), dtype=np.uint8),
        ]
        writer = None
        for codec in ("MJPG", "XVID", "mp4v"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 5.0, size)
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None:
            self.skipTest("OpenCV video writer codec is unavailable in this environment.")

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    @staticmethod
    def _write_image(path: Path, color: tuple[int, int, int]) -> None:
        image = np.full((8, 8, 3), color, dtype=np.uint8)
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise RuntimeError(f"Failed to write test image: {path}")


if __name__ == "__main__":
    unittest.main()

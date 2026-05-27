import unittest

import numpy as np

import analysis_metrics
from webapp.backend import analysis as backend_analysis


class AnalysisMetricTests(unittest.TestCase):
    def test_dominant_frequency_metrics_prefers_in_band_peak(self):
        fps = 30.0
        t = np.arange(300, dtype=np.float32) / fps
        signal = np.sin(2.0 * np.pi * 6.0 * t) + 0.15 * np.sin(2.0 * np.pi * 10.0 * t)

        dominant_hz, peak_ratio, peak_sharpness = analysis_metrics.dominant_frequency_metrics(signal, fps=fps)

        self.assertAlmostEqual(dominant_hz, 6.0, places=1)
        self.assertGreater(peak_ratio, 0.4)
        self.assertGreater(peak_sharpness, 1.0)

    def test_temporal_noise_distinguishes_linear_and_jittery_motion(self):
        smooth = np.stack([np.arange(20, dtype=np.float32), np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32)], axis=1)
        noisy = smooth.copy()
        noisy[:, 0] += np.asarray([0.0, 0.4, -0.2, 0.6, -0.5] * 4, dtype=np.float32)

        smooth_noise = analysis_metrics.frame_to_frame_variance(smooth)
        noisy_motion = analysis_metrics.frame_to_frame_variance(noisy)

        self.assertAlmostEqual(smooth_noise, 0.0, places=6)
        self.assertGreater(noisy_motion, smooth_noise)

    def test_spatial_coherence_tracks_neighbor_correlation(self):
        t = np.arange(16, dtype=np.float32)
        positions = np.zeros((16, 3, 3), dtype=np.float32)
        shared_signal = np.sin(t * 0.4)
        positions[:, 0, 0] = shared_signal
        positions[:, 1, 0] = shared_signal * 1.1
        positions[:, 2, 1] = np.cos(t * 0.9)

        coherence = analysis_metrics.spatial_coherence_from_positions(positions, [(0, 1), (0, 2)])

        self.assertIsNotNone(coherence)
        self.assertGreater(coherence, 0.1)

    def test_backend_mediapipe_centroid_leaves_spatial_coherence_empty(self):
        source = {"family": "mediapipe", "id": "mediapipe:test"}
        frames = [
            {
                "frame_id": 0,
                "hands": [{"right": 1, "points": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)}],
            },
            {
                "frame_id": 1,
                "hands": [{"right": 1, "points": np.asarray([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float32)}],
            },
        ]

        result = backend_analysis._analyze_centroid(source, frames, hand_value=1, wrist_joint_idx=0, fps=30.0)

        self.assertIsNone(result["spatial_coherence"])


if __name__ == "__main__":
    unittest.main()

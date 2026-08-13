import unittest

import numpy as np
from scipy.signal import welch

import analysis_metrics
from webapp.backend import analysis as backend_analysis


class AnalysisMetricTests(unittest.TestCase):
    def test_dominant_frequency_metrics_prefers_in_band_peak(self):
        fps = 30.0
        t = np.arange(300, dtype=np.float32) / fps
        signal = 2.5 * np.sin(2.0 * np.pi * 2.0 * t) + np.sin(2.0 * np.pi * 6.0 * t)

        dominant_hz, peak_ratio, peak_sharpness = analysis_metrics.dominant_frequency_metrics(signal, fps=fps)

        self.assertAlmostEqual(dominant_hz, 6.0, places=1)
        self.assertGreater(peak_ratio, 0.3)
        self.assertGreater(peak_sharpness, 1.0)

    def test_dominant_frequency_metrics_match_welch_band_metrics(self):
        fps = 30.0
        t = np.arange(240, dtype=np.float32) / fps
        signal = np.sin(2.0 * np.pi * 7.5 * t) + 0.25 * np.random.default_rng(0).normal(size=t.shape[0]).astype(np.float32)

        expected_freqs, expected_psd, _ = analysis_metrics.welch_psd(signal, fps=fps)
        expected = analysis_metrics.band_peak_metrics(expected_freqs, expected_psd)
        actual = analysis_metrics.dominant_frequency_metrics(signal, fps=fps)

        self.assertAlmostEqual(actual[0], expected[0], places=6)
        self.assertAlmostEqual(actual[1], expected[1], places=6)
        self.assertAlmostEqual(actual[2], expected[2], places=6)

    def test_welch_psd_uses_shared_two_second_defaults(self):
        fps = 30.0
        signal = np.linspace(-1.0, 1.0, 300, dtype=np.float32)

        freqs, psd, config = analysis_metrics.welch_psd(signal, fps=fps)
        expected_freqs, expected_psd = welch(
            signal,
            fs=fps,
            window="hann",
            nperseg=60,
            noverlap=30,
            detrend="constant",
            scaling="density",
        )

        self.assertEqual(config["nperseg"], 60)
        self.assertEqual(config["noverlap"], 30)
        self.assertEqual(config["window"], "hann")
        self.assertEqual(config["detrend"], "constant")
        self.assertEqual(config["scaling"], "density")
        self.assertTrue(np.allclose(freqs, expected_freqs.astype(np.float32)))
        self.assertTrue(np.allclose(psd, expected_psd.astype(np.float32)))

    def test_peak_sharpness_uses_local_welch_bin_neighborhood(self):
        freqs = np.asarray([0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 13.0], dtype=np.float32)
        psd = np.asarray([0.0, 1.0, 2.0, 10.0, 2.0, 1.0, 0.5], dtype=np.float32)

        dominant_hz, peak_ratio, peak_sharpness = analysis_metrics.band_peak_metrics(freqs, psd)

        self.assertAlmostEqual(dominant_hz, 6.0, places=6)
        self.assertAlmostEqual(peak_ratio, 10.0 / 16.0, places=6)
        self.assertAlmostEqual(peak_sharpness, 10.0 / 3.2, places=6)

    def test_fft_periodogram_and_peak_summary_find_in_band_peak(self):
        fps = 30.0
        t = np.arange(300, dtype=np.float32) / fps
        rng = np.random.default_rng(7)
        signal = np.sin(2.0 * np.pi * 7.0 * t) + 0.15 * rng.normal(size=t.shape[0]).astype(np.float32)

        freqs, spectrum = analysis_metrics.fft_periodogram(signal, fps=fps)
        summary = analysis_metrics.band_peak_summary(freqs, spectrum)

        self.assertAlmostEqual(summary["peak_hz"], 7.0, places=1)
        self.assertGreater(summary["peak_value"], 0.0)
        self.assertGreater(summary["peak_ratio"], 0.15)

    def test_finish_motion_analysis_exposes_welch_and_fft_peaks(self):
        fps = 30.0
        t = np.arange(300, dtype=np.float32) / fps
        trajectory = np.stack(
            [
                1.0 + 0.25 * np.sin(2.0 * np.pi * 6.0 * t),
                np.zeros_like(t),
                np.zeros_like(t),
            ],
            axis=1,
        )

        result = analysis_metrics.finish_motion_analysis(
            trajectory,
            fps=fps,
            filter_kind="lowpass",
            filter_order=3,
            lowpass_cutoff_hz=8.0,
        )

        self.assertAlmostEqual(result["dominant"], 6.0, places=1)
        self.assertAlmostEqual(result["fft_peak_hz"], 6.0, places=1)
        self.assertIn("fft_freqs", result)
        self.assertIn("fft_spectrum", result)
        self.assertGreater(result["peak_value"], 0.0)
        self.assertGreater(result["fft_peak_value"], 0.0)

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

    def test_backend_serialize_result_includes_fft_plot_data(self):
        fps = 30.0
        t = np.arange(180, dtype=np.float32) / fps
        trajectory = np.stack(
            [
                1.0 + 0.2 * np.sin(2.0 * np.pi * 5.5 * t),
                np.zeros_like(t),
                np.zeros_like(t),
            ],
            axis=1,
        )
        result = analysis_metrics.finish_motion_analysis(
            trajectory,
            fps=fps,
            filter_kind="lowpass",
            filter_order=3,
            lowpass_cutoff_hz=8.0,
        )

        serialized = backend_analysis._serialize_result(result, fps=fps)

        self.assertIn("fft_peak_hz", serialized)
        self.assertIn("fft_freqs_hz", serialized["plots"])
        self.assertIn("fft_spectrum", serialized["plots"])
        self.assertEqual(len(serialized["plots"]["fft_freqs_hz"]), len(serialized["plots"]["fft_spectrum"]))
        self.assertGreater(serialized["plots"]["welch_peak_value"], 0.0)
        self.assertGreater(serialized["plots"]["fft_peak_value"], 0.0)


if __name__ == "__main__":
    unittest.main()

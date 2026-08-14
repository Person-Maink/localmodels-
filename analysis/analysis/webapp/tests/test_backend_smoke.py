import unittest

from webapp.backend.main import app
from webapp.backend.registry import analysis_modes, visualization_modes
from webapp.backend.settings import parse_number_list, parse_pair_text, strip_variant_suffix


class BackendSmokeTests(unittest.TestCase):
    def test_strip_variant_suffix(self):
        self.assertEqual(strip_variant_suffix("120-2_clip_1_modified"), "120-2_clip_1")
        self.assertEqual(strip_variant_suffix("120-2_clip_1_amplified_modified"), "120-2_clip_1")
        self.assertEqual(strip_variant_suffix("me 1"), "me 1")

    def test_parse_pair_text(self):
        self.assertEqual(parse_pair_text("4-8, 9-13"), ((4, 8), (9, 13)))
        self.assertEqual(parse_pair_text([{"a": 1, "b": 2}, (3, 4)]), ((1, 2), (3, 4)))

    def test_parse_number_list(self):
        self.assertEqual(parse_number_list("10,20,30"), (10, 20, 30))
        self.assertEqual(parse_number_list([5, 10]), (5, 10))

    def test_mode_registry_has_expected_entries(self):
        viz_ids = {mode.id for mode in visualization_modes()}
        analysis_ids = {mode.id for mode in analysis_modes()}
        self.assertIn("free_view", viz_ids)
        self.assertIn("camera_trajectories", viz_ids)
        self.assertIn("centroid_compare", analysis_ids)
        self.assertIn("beta_multi_point", analysis_ids)

    def test_fastapi_routes_exist(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/library/tree", paths)
        self.assertIn("/api/visualization/manifest", paths)
        self.assertIn("/api/analysis/run", paths)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from main import make_argparser


class MainCliTests(unittest.TestCase):
    def test_parser_rejects_removed_stride_vipe_mode_and_flag(self) -> None:
        parser = make_argparser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--mode", "stride-vipe"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--vipe_output_root", "/tmp/vipe"])

    def test_parser_accepts_surviving_modes(self) -> None:
        parser = make_argparser()

        wilor_args = parser.parse_args(["--mode", "wilor"])
        stride_args = parser.parse_args(["--mode", "stride"])

        self.assertEqual(wilor_args.mode, "wilor")
        self.assertEqual(stride_args.mode, "stride")


if __name__ == "__main__":
    unittest.main()

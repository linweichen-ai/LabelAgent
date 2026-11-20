import unittest

from tools.evaluate_pallet_lidar_point_prompts import summarize_sam_attempts


class PalletLidarInstanceSamTest(unittest.TestCase):
    def test_summary_keeps_frames_and_attempts_distinct(self):
        rows = [
            {"sam_called": True, "sam_attempts": [
                {"sam_mask_pixels": 10}, {"http_status": 400, "error": "rejected"}
            ]},
            {"sam_called": False},
        ]
        self.assertEqual(summarize_sam_attempts(rows), {
            "sam_frame_count": 1,
            "sam_attempt_count": 2,
            "sam_successful_attempt_count": 1,
            "sam_http_error_count": 1,
            "sam_other_error_count": 0,
        })


if __name__ == "__main__":
    unittest.main()

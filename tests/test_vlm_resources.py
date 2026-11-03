import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("vlm_resources", Path(__file__).resolve().parents[1] / "tools/vlm_resources.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VLMResourceTests(unittest.TestCase):
    def test_occupied_card_is_rejected_even_with_memory_available(self):
        rows = MODULE.parse_inventory("GPU-a, NVIDIA GeForce RTX 4090, 24564, 10, 0, 570\n", "GPU-a, 123\n")
        with self.assertRaisesRegex(ValueError, "existing compute"):
            MODULE.require_idle_gpu(rows, "GPU-a")

    def test_selected_uuid_must_match_idle_4090_with_free_memory(self):
        rows = MODULE.parse_inventory("GPU-a, NVIDIA GeForce RTX 4090, 24564, 10, 0, 570\n", "")
        self.assertEqual(MODULE.require_idle_gpu(rows, "GPU-a")["uuid"], "GPU-a")
        for uuid in ("GPU-other",):
            with self.assertRaises(ValueError):
                MODULE.require_idle_gpu(rows, uuid)
        rows[0]["free_mib"] = 1000
        with self.assertRaisesRegex(ValueError, "insufficient"):
            MODULE.require_idle_gpu(rows, "GPU-a")


if __name__ == "__main__":
    unittest.main()

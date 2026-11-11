import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("vlm_resources", Path(__file__).resolve().parents[1] / "tools/vlm_resources.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VLMResourceTests(unittest.TestCase):
    def test_inventory_uses_reported_free_memory_not_total_minus_allocations(self):
        rows=MODULE.parse_inventory('GPU-a, NVIDIA GeForce RTX 4090, 24564, 8000, 0, 580, 15000','GPU-a, 123')
        self.assertEqual(rows[0]['free_mib'],15000)
        self.assertTrue(rows[0]['free_memory_reported'])

    def test_resident_models_allowed_only_in_explicit_capped_probe(self):
        rows=MODULE.parse_inventory('GPU-a, NVIDIA GeForce RTX 4090, 24564, 8000, 0, 580, 15000','GPU-a, 123')
        result=MODULE.require_shared_probe_gpu(rows,'GPU-a')
        self.assertEqual(result['process_allocator_fraction'],.35)
        self.assertGreaterEqual(result['reserved_free_mib'],4096)
        with self.assertRaisesRegex(ValueError,'existing compute'):MODULE.require_idle_gpu(rows,'GPU-a')

    def test_probe_requires_budget_plus_reserve_and_context_headroom(self):
        rows=MODULE.parse_inventory('GPU-a, NVIDIA GeForce RTX 4090, 24564, 11000, 0, 580, 12000','GPU-a, 123')
        with self.assertRaisesRegex(ValueError,'headroom'):MODULE.require_shared_probe_gpu(rows,'GPU-a')

    def test_active_compute_prevents_new_probe(self):
        rows=MODULE.parse_inventory('GPU-a, NVIDIA GeForce RTX 4090, 24564, 8000, 6, 580, 15000','GPU-a, 123')
        with self.assertRaisesRegex(ValueError,'active compute'):MODULE.require_shared_probe_gpu(rows,'GPU-a')

    def test_probe_cannot_weaken_its_memory_or_reserve_bounds(self):
        rows=MODULE.parse_inventory('GPU-a, NVIDIA GeForce RTX 4090, 24564, 8000, 0, 580, 15000','GPU-a, 123')
        with self.assertRaises(ValueError):MODULE.require_shared_probe_gpu(rows,'GPU-a',.5,4096)
        with self.assertRaises(ValueError):MODULE.require_shared_probe_gpu(rows,'GPU-a',.35,0)

    def test_probe_rejects_legacy_estimated_free_memory(self):
        rows=MODULE.parse_inventory('GPU-a, NVIDIA GeForce RTX 4090, 24564, 8000, 0, 580','GPU-a, 123')
        with self.assertRaisesRegex(ValueError,'direct free-memory'):MODULE.require_shared_probe_gpu(rows,'GPU-a')

    def test_local_profile_preserves_display_processes_and_rejects_full_gpu(self):
        rows = MODULE.parse_inventory("GPU-local, NVIDIA GeForce RTX 5060 Laptop GPU, 8151, 700, 5, 573.24", "GPU-local, 100")
        self.assertEqual(MODULE.require_local_gpu(rows, "GPU-local")["uuid"], "GPU-local")
        rows[0]["free_mib"] = 100
        with self.assertRaises(ValueError):
            MODULE.require_local_gpu(rows, "GPU-local")

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

    def test_dedicated_5090_requires_direct_free_memory_and_idle_card(self):
        rows = MODULE.parse_inventory(
            "GPU-5090, NVIDIA GeForce RTX 5090, 32607, 88, 0, 595.99, 32519", "")
        selected = MODULE.require_dedicated_5090_gpu(rows, "GPU-5090")
        self.assertEqual(selected["uuid"], "GPU-5090")

        rows[0]["has_compute_process"] = True
        with self.assertRaisesRegex(ValueError, "existing compute"):
            MODULE.require_dedicated_5090_gpu(rows, "GPU-5090")

    def test_dedicated_5090_rejects_wrong_hardware_or_low_headroom(self):
        wrong = MODULE.parse_inventory(
            "GPU-a, NVIDIA GeForce RTX 4090, 24564, 10, 0, 595.99, 24554", "")
        with self.assertRaisesRegex(ValueError, "not the requested RTX 5090"):
            MODULE.require_dedicated_5090_gpu(wrong, "GPU-a")

        low = MODULE.parse_inventory(
            "GPU-5090, NVIDIA GeForce RTX 5090, 32607, 6000, 0, 595.99, 26607", "")
        with self.assertRaisesRegex(ValueError, "insufficient"):
            MODULE.require_dedicated_5090_gpu(low, "GPU-5090")

        legacy = MODULE.parse_inventory(
            "GPU-5090, NVIDIA GeForce RTX 5090, 32607, 88, 0, 595.99", "")
        with self.assertRaisesRegex(ValueError, "direct free-memory"):
            MODULE.require_dedicated_5090_gpu(legacy, "GPU-5090")


if __name__ == "__main__":
    unittest.main()

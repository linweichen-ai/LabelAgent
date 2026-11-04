import hashlib
import unittest

import numpy as np
from PIL import Image
from tools.evaluate_target_cards import card_payload
from tools.sam_evidence_recheck import choose_support_mask, evidence_views, needs_recheck, rechecked_pair


class EvidenceRecheckChecks(unittest.TestCase):
    def test_trigger_is_symmetric_and_needs_unique_support(self):
        for category, other in (("pallet", "person"), ("person", "pallet")):
            self.assertTrue(needs_recheck(["uncertain", other], [category]))
            self.assertFalse(needs_recheck([category, category], [category]))
            self.assertFalse(needs_recheck(["uncertain", other], []))
            self.assertFalse(needs_recheck(["uncertain", other], [category, other]))

    def test_choose_max_overlap_stable_tie_not_score_or_class(self):
        target = np.ones((2, 2), bool); a = target.copy()
        a[0, 0] = False
        index, hits = choose_support_mask(target, [a, target, target.copy()])
        self.assertEqual((index, hits), (1, 4))
        with self.assertRaises(ValueError):
            choose_support_mask(target, [np.zeros_like(target)])

    def test_recheck_needs_both_fresh_views_and_matching_support(self):
        for category, other in (("pallet", "person"), ("person", "pallet")):
            previous = [category, "uncertain"]
            self.assertEqual(rechecked_pair(previous, [category], [category, category])["categories"], [category, category])
            for fresh in ([category, "uncertain"], [category, other], [other, other]):
                self.assertEqual(rechecked_pair(previous, [category], fresh)["categories"], previous)

    def test_complete_target_retained_when_auxiliary_only_partly_overlaps(self):
        target = np.zeros((60, 100), bool); target[10:15, 4:94] = True
        aux = np.zeros_like(target); aux[8:30, 40:99] = True
        source = Image.new("RGB", (100, 60), "#446688")
        pictures, meta = evidence_views(source, target, aux)
        self.assertEqual(meta["retained_target_pixels"], 450)
        self.assertEqual(pictures["overview"].size, (512, 512))
        self.assertEqual(pictures["linked"].size, (1536, 512))
        self.assertFalse(meta["category_ID_score_rendered"])

    def test_source_and_both_masks_unchanged(self):
        source = Image.new("RGB", (50, 50), "red")
        target = np.zeros((50, 50), bool); target[0:10, 0:10] = True
        aux = np.zeros_like(target); aux[:30, :30] = True
        before = source.tobytes(), target.tobytes(), aux.tobytes()
        evidence_views(source, target, aux)
        self.assertEqual(before, (source.tobytes(), target.tobytes(), aux.tobytes()))

    def test_expected_class_ID_and_SAM_class_are_not_in_payload(self):
        raw = b"bound synthetic bytes"; ref = {"path": "evidence.jpg", "sha256": hashlib.sha256(raw).hexdigest()}
        fixture = {"categories": ["pallet", "person", "uncertain"], "model": "labelagent-vlm",
                   "temperature": 0, "max_tokens": 192, "prompts": {"overview": "neutral visual review"},
                   "image_token_budgets": {"overview": 256}}
        case = {"id": "private", "expected_category": "person", "supported_class": "pallet", "views": {"overview": [ref]}}
        first = card_payload(fixture, {"evidence.jpg": (raw, "image/jpeg")}, case, "overview")
        second = card_payload(fixture, {"evidence.jpg": (raw, "image/jpeg")},
                              {**case, "id": "changed", "expected_category": "pallet", "supported_class": "person"}, "overview")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

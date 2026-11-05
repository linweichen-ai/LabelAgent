from copy import deepcopy
import unittest

from tools.evaluate_scene_capacity import content_fingerprint, same_actual_encoding, without_budget


class CapacityMatchingChecks(unittest.TestCase):
    def test_only_budget_field_may_differ_in_semantic_request(self):
        local = {'model': 'labelagent-vlm', 'image_token_budget': 768, 'temperature': 0,
                 'messages': [{'role': 'user', 'content': 'bound scene bytes'}], 'response_format': {'type': 'json_schema'}}
        before = deepcopy(local); remote = without_budget(local)
        self.assertEqual(local, before)
        self.assertEqual(content_fingerprint(local), content_fingerprint(remote))
        for key, value in (('model', 'different'), ('temperature', 1), ('messages', []), ('response_format', {})):
            changed = deepcopy(remote); changed[key] = value
            self.assertNotEqual(content_fingerprint(local), content_fingerprint(changed))

    def test_actual_grid_must_match_even_when_nominal_caps_match(self):
        a = {'actual_image_token_counts': [768], 'image_grid_thw': [[1, 32, 96]]}
        self.assertTrue(same_actual_encoding(a, deepcopy(a)))
        self.assertFalse(same_actual_encoding(a, {'actual_image_token_counts': [256], 'image_grid_thw': [[1, 32, 32]]}))
        self.assertFalse(same_actual_encoding(a, {'actual_image_token_counts': [768], 'image_grid_thw': [[1, 64, 48]]}))
        self.assertFalse(same_actual_encoding({}, {}))


if __name__ == '__main__':
    unittest.main()

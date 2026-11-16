from copy import deepcopy
import hashlib
import json
import unittest

import numpy as np
from PIL import Image
from tools.grounded_target_views import CELLS, classification_views, localization_payload, localization_view, location_ok, target_anchor, target_cells, validated_location


class GroundedTargetChecks(unittest.TestCase):
    def test_location_accepts_target_cell_and_rejects_other_object(self):
        mask = np.zeros((90, 90), bool); mask[42:48, 82:88] = True
        self.assertEqual(target_cells(mask), ['E9'])
        self.assertTrue(location_ok('E9', mask)); self.assertFalse(location_ok('G8', mask)); self.assertFalse(location_ok('uncertain', mask))

    def test_arrow_anchor_is_foreground_even_with_hole(self):
        mask = np.ones((90, 90), bool); mask[20:70, 20:70] = False
        x, y = target_anchor(mask); self.assertTrue(mask[y, x])

    def test_all_target_pixels_retained_across_grid_boundary(self):
        mask = np.zeros((90, 90), bool); mask[19:25, 18:25] = True
        _, meta = classification_views(Image.new('RGB', (90, 90), 'red'), mask, 'B2')
        self.assertEqual(meta['retained_target_pixels'], int(mask.sum()))

    def test_wrong_location_prevents_classification_inputs(self):
        mask = np.zeros((90, 90), bool); mask[42:48, 82:88] = True
        with self.assertRaises(ValueError):
            classification_views(Image.new('RGB', (90, 90)), mask, 'G8')

    def test_original_source_and_mask_not_mutated(self):
        source = Image.new('RGB', (90, 90), '#123456'); mask = np.zeros((90, 90), bool); mask[42:48, 82:88] = True
        before = source.tobytes(), mask.tobytes()
        picture, meta = localization_view(source, mask)
        views, _ = classification_views(source, mask, 'E9')
        self.assertEqual(picture.size, (1024, 1024)); self.assertEqual(views['overview'].size, (512, 512))
        self.assertEqual(before, (source.tobytes(), mask.tobytes()))

    def test_localizer_payload_has_all_cells_and_no_expected_answer(self):
        raw = b'bound test image'; ref = {'sha256': hashlib.sha256(raw).hexdigest()}
        fixture = {'model': 'labelagent-vlm', 'localization_prompt': 'Locate the purple contour; no class question.'}
        case = {'localization_image': ref, 'expected_category': 'pallet', 'acceptable_cells': ['E9'], 'id': 'private'}
        first = localization_payload(fixture, (raw, 'image/jpeg'), case)
        second = localization_payload(fixture, (raw, 'image/jpeg'), {**case, 'expected_category': 'person', 'acceptable_cells': ['A1'], 'id': 'other'})
        self.assertEqual(first, second)
        self.assertEqual(first['response_format']['json_schema']['schema']['properties']['cell']['enum'], [*CELLS, 'uncertain'])

    def test_localizer_cannot_accept_semantic_or_truncated_output(self):
        raw = {'local_vlm': {'backend': 'transformers_local_cuda', 'image_count': 1},
               'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'cell': 'E9', 'reason': 'purple contour'})}}]}
        self.assertEqual(validated_location(raw)['cell'], 'E9')
        bad = deepcopy(raw); bad['choices'][0]['message']['content'] = json.dumps({'category': 'pallet', 'reason': 'other task'})
        with self.assertRaises(ValueError): validated_location(bad)
        raw['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(ValueError): validated_location(raw)

    def test_small_crop_grid_is_drawn_at_display_scale(self):
        mask = np.zeros((90, 90), bool); mask[42:48, 82:88] = True
        _, meta = localization_view(Image.new('RGB', (90, 90), 'red'), mask)
        self.assertEqual(meta['grid_font_display_pixels'], 16)
        x, y = meta['display_arrow_tip']; self.assertTrue(32 <= x < 992 and 32 <= y < 992)


if __name__ == '__main__':
    unittest.main()

"""Geometry and leakage guards; no model calls or original annotation writes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from tools.evaluate_target_cards import fit_panel

# Works both before installation beside the module and after installation in tests.
HERE = Path(__file__).resolve().parent
MODULE = HERE / 'contour_visibility.py'
if MODULE.exists():
    SPEC = importlib.util.spec_from_file_location('contour_visibility_under_test', MODULE)
    contour = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(contour)
else:
    from tools import contour_visibility as contour


class ContourVisibilityTests(unittest.TestCase):
    def test_interior_anchor_avoids_concave_centroid_boundary(self):
        mask = np.zeros((60, 70), dtype=bool)
        mask[10:50, 10:60] = True
        mask[10:38, 22:48] = False
        original = mask.copy()
        views, meta = contour.render(Image.new('RGB', (70, 60), (95, 100, 110)), mask, [0, 0, 70, 60])
        ax, ay = meta['anchor_source']
        self.assertTrue(mask[ay, ax])
        self.assertGreater(meta['max_distance'], meta['old_anchor_distance'])
        self.assertGreater(meta['max_distance'], 1)
        self.assertEqual(meta['original_target_pixels'], meta['retained_target_pixels'])
        self.assertFalse(meta['filled_overlay'])
        self.assertEqual(set(views), {'overview', 'linked'})
        np.testing.assert_array_equal(mask, original)

    def test_image_edge_has_zero_padding_and_deterministic_center(self):
        mask = np.ones((31, 31), dtype=bool)
        source = Image.new('RGB', (31, 31), (91, 91, 91))
        first, meta = contour.render(source, mask, [0, 0, 31, 31])
        second, again = contour.render(source, mask, [0, 0, 31, 31])
        self.assertEqual(meta['anchor_source'], [15, 15])
        self.assertEqual(meta['max_distance'], 16)
        self.assertEqual(meta, again)
        self.assertEqual(first['overview'].tobytes(), second['overview'].tobytes())

    def test_disconnected_parts_all_retained_and_crop_escape_rejected(self):
        mask = np.zeros((40, 50), dtype=bool)
        mask[10:15, 12:20] = True
        mask[24:26, 35:37] = True
        source = Image.new('RGB', (50, 40))
        _, meta = contour.render(source, mask, [10, 8, 40, 30])
        self.assertEqual(meta['retained_target_pixels'], 44)
        self.assertEqual(meta['contour_count'], 2)
        for box in ([10, 8, 30, 30], [-1, 0, 50, 40], [10., 8, 40, 30]):
            with self.assertRaises(ValueError):
                contour.render(source, mask, box)

    def test_pixels_outside_fixed_context_cannot_change_render(self):
        mask = np.zeros((40, 50), dtype=bool)
        mask[12:23, 16:30] = True
        images = [Image.new('RGB', (50, 40), color) for color in ((255, 0, 0), (0, 255, 255))]
        for im in images:
            im.paste((43, 67, 91), (10, 8, 40, 30))
        first, meta = contour.render(images[0], mask, [10, 8, 40, 30])
        second, other = contour.render(images[1], mask, [10, 8, 40, 30])
        self.assertEqual(meta, other)
        for view in first:
            self.assertEqual(first[view].tobytes(), second[view].tobytes())

    def test_blank_markers_share_identical_geometry_and_letterbox(self):
        mask = np.zeros((60, 80), dtype=bool)
        mask[20:25, 18:53] = True
        changed = []
        metas = []
        for value in (128, 255):
            source = Image.new('RGB', (80, 60), (value,) * 3)
            views, meta = contour.render(source, mask, [5, 10, 75, 40])
            clean, fit = fit_panel(source.crop((5, 10, 75, 40)), 512, 512)
            self.assertEqual(meta['fit'], fit)
            changed.append({v: np.any(np.asarray(im) != np.asarray(clean), axis=2) for v, im in views.items()})
            metas.append(meta)
        self.assertEqual(metas[0], metas[1])
        for view in ('overview', 'linked'):
            np.testing.assert_array_equal(changed[0][view], changed[1][view])
            self.assertTrue(changed[0][view].any())
        self.assertLess(changed[0]['linked'].sum(), 512 * 512 // 20)

    @staticmethod
    def fixture():
        assets = {'clean': (b'clean RGB', 'image/jpeg'), 'marked': (b'outlined RGB', 'image/jpeg')}
        refs = [{'path': p, 'sha256': hashlib.sha256(raw).hexdigest()} for p, (raw, _) in assets.items()]
        fixture = dict(model='local', categories=['pallet', 'person', 'uncertain'],
                       geometry={'private_id': {'seed': [321, 654]}}, private_truth='DO_NOT_SEND')
        case = dict(id='private_id', expected='DO_NOT_SEND', views={v: refs for v in ('overview', 'linked')})
        grounding = dict(visibility='visible', x=321, y=654, features='PRIVATE_FEATURES', category='PRIVATE_CLASS')
        return fixture, assets, case, grounding

    def test_no_inspection_fields_coordinates_features_or_truth_enter_request(self):
        fixture, assets, case, grounding = self.fixture()
        changed_fixture, changed_case = copy.deepcopy(fixture), copy.deepcopy(case)
        changed_fixture['geometry'] = {'different': {'seed': [999, 999]}}
        changed_fixture['private_truth'] = 'OTHER_TRUTH'
        changed_case.update(id='different', expected='OTHER_TRUTH', human_category='OTHER_TRUTH')
        changed_grounding = {**grounding, 'visibility': 'empty', 'x': 999, 'y': 999, 'features': 'OTHER_FEATURES', 'category': 'OTHER_CATEGORY'}
        for view in ('overview', 'linked'):
            before = contour.classify_payload(fixture, assets, case, view, grounding)
            after = contour.classify_payload(changed_fixture, assets, changed_case, view, changed_grounding)
            self.assertEqual(before, after)
            encoded = json.dumps(before[0])
            for secret in ('321', '654', 'PRIVATE_', 'DO_NOT_SEND', 'private_id'):
                self.assertNotIn(secret, encoded)
            self.assertNotIn('image_token_budget', before[0])
            self.assertNotIn('visibility=', before[0]['messages'][0]['content'][0]['text'])

    def test_every_visibility_keeps_all_three_categories_unforced(self):
        fixture, assets, case, grounding = self.fixture()
        digests = set()
        for visibility in ('visible', 'empty', 'unclear'):
            body, digest = contour.classify_payload(fixture, assets, case, 'overview', {**grounding, 'visibility': visibility})
            schema = body['response_format']['json_schema']['schema']
            self.assertEqual(schema['properties']['category']['enum'], ['pallet', 'person', 'uncertain'])
            self.assertNotIn('const', schema['properties']['category'])
            digests.add(digest)
        self.assertEqual(len(digests), 1)

    def test_grounding_builder_is_exactly_the_original(self):
        from tools.target_first import grounding_payload
        self.assertIs(contour.grounding_payload, grounding_payload)


if __name__ == '__main__':
    unittest.main()

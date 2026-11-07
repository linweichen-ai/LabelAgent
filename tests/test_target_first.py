"""No model calls: protect target pixels, coordinate mapping and prompt provenance."""
import base64
import copy
import hashlib
import json
import unittest

import numpy as np
from PIL import Image

from tools.target_first import (
    classify_payload, geometry_evidence, grounding_payload,
    target_enlargement, validated_grounding,
)


class TargetFirstTests(unittest.TestCase):
    @staticmethod
    def raw(parsed, **metadata):
        return {
            'local_vlm': dict(backend='transformers_local_cuda', image_count=2, **metadata),
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(parsed)}}],
        }

    @staticmethod
    def fixture():
        raw_assets = [('tight', b'tight-source'), ('context', b'clean-context'), ('marked', b'marked-context')]
        assets = {name: (raw, 'image/jpeg') for name, raw in raw_assets}
        refs = {name: dict(path=name, sha256=hashlib.sha256(raw).hexdigest()) for name, raw in raw_assets}
        case = dict(id='PRIVATE_TARGET_ID', human_category='pallet', expected_category='pallet',
                    role='PRIVATE_EVALUATION_ROLE', localization=[refs['tight'], refs['context']],
                    views={'overview': [refs['tight'], refs['marked']]})
        fixture = dict(model='test-model', categories=['pallet', 'person', 'uncertain'],
                       geometry={case['id']: dict(seed=[342, 617], human_category='DO_NOT_SEND')},
                       private_reference='DO_NOT_SEND', expected_category='DO_NOT_SEND')
        inspection = dict(visibility='visible', x=340, y=614, features='A light horizontal surface.')
        return fixture, assets, case, inspection

    def test_bbox_retains_all_mask_pixels_including_disconnected_extent(self):
        source = Image.new('RGB', (31, 23), (11, 29, 51))
        mask = np.zeros((23, 31), dtype=bool)
        mask[4:7, 5:10] = True
        mask[17, 25] = True
        original = mask.copy()
        _, meta = target_enlargement(source, mask)
        self.assertEqual(meta['source_box'], [5, 4, 26, 18])
        self.assertEqual(meta['original_target_pixels'], 16)
        self.assertEqual(meta['retained_target_pixels'], 16)
        packed = np.frombuffer(base64.b64decode(meta['mask_packbits_b64']), dtype=np.uint8)
        reconstructed = np.unpackbits(packed)[:14*21].reshape(14, 21).astype(bool)
        np.testing.assert_array_equal(reconstructed, mask[4:18, 5:26])
        np.testing.assert_array_equal(mask, original)

    def test_changing_pixels_outside_bbox_cannot_change_target_image(self):
        a = Image.new('RGB', (32, 24), (255, 0, 0))
        b = Image.new('RGB', a.size, (0, 255, 255))
        for picture in (a, b):
            picture.paste((16, 41, 90), (7, 8, 19, 13))
            picture.putpixel((7, 8), (240, 170, 20))
            picture.putpixel((18, 12), (10, 140, 240))
        mask = np.zeros((24, 32), dtype=bool)
        mask[8:13, 7:19] = True
        x, mx = target_enlargement(a, mask)
        y, my = target_enlargement(b, mask)
        self.assertEqual(x.tobytes(), y.tobytes())
        self.assertEqual(mx, my)

    def test_seed_roundtrip_lands_on_selected_source_pixel(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[10:13, 4:23] = True
        _, meta = target_enlargement(Image.new('RGB', (40, 30)), mask)
        evidence = geometry_evidence(dict(visibility='visible', x=meta['seed'][0], y=meta['seed'][1]), meta)
        self.assertTrue(evidence['point_inside_target'])
        self.assertTrue(evidence['point_inside_display'])
        self.assertEqual(evidence['source_point'], [13, 11])
        self.assertTrue(evidence['geometry_only_not_human_localization_truth'])

    def test_pixel_center_roundtrip_at_target_extremes(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[10:13, 4:23] = True
        _, meta = target_enlargement(Image.new('RGB', (40, 30)), mask)
        ox, oy = meta['fit']['offset']
        dw, dh = meta['fit']['display_size']
        for sx, sy in ((4, 10), (22, 10), (4, 12), (22, 12)):
            x = round((ox + (sx-4+.5)*dw/19-.5)/511*1000)
            y = round((oy + (sy-10+.5)*dh/3-.5)/511*1000)
            evidence = geometry_evidence(dict(visibility='visible', x=x, y=y), meta)
            self.assertEqual(evidence['source_point'], [sx, sy])
            self.assertTrue(evidence['point_inside_target'])

    def test_letterbox_point_is_outside_even_when_in_canvas(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[10:13, 4:23] = True
        _, meta = target_enlargement(Image.new('RGB', (40, 30)), mask)
        self.assertGreater(meta['fit']['offset'][1], 0)
        evidence = geometry_evidence(dict(visibility='visible', x=500, y=0), meta)
        self.assertFalse(evidence['point_inside_display'])
        self.assertFalse(evidence['point_inside_target'])

    def test_bbox_hole_is_not_reported_as_selected_foreground(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[5:20, 10:25] = True
        mask[10:15, 15:20] = False
        _, meta = target_enlargement(Image.new('RGB', (40, 30)), mask)
        evidence = geometry_evidence(dict(visibility='visible', x=500, y=500), meta)
        self.assertTrue(evidence['point_inside_display'])
        self.assertFalse(evidence['point_inside_target'])

    def test_human_answers_case_id_and_roles_cannot_affect_requests(self):
        f, a, c, inspection = self.fixture()
        f2, c2 = copy.deepcopy(f), copy.deepcopy(c)
        c2.update(id='OTHER_PRIVATE_ID', human_category='person', expected_category='person', role='DIFFERENT_ROLE')
        f2['geometry'][c2['id']] = f2['geometry'].pop(c['id'])
        f2['private_reference'] = 'CHANGED_REFERENCE'
        self.assertEqual(grounding_payload(f, a, c), grounding_payload(f2, a, c2))
        self.assertEqual(classify_payload(f, a, c, 'overview', inspection),
                         classify_payload(f2, a, c2, 'overview', inspection))
        for body in (grounding_payload(f, a, c)[0], classify_payload(f, a, c, 'overview', inspection)[0]):
            text = json.dumps(body)
            for secret in ('PRIVATE_TARGET_ID', 'PRIVATE_EVALUATION_ROLE', 'DO_NOT_SEND'):
                self.assertNotIn(secret, text)

    def test_two_image_requests_keep_order_without_single_image_budget_override(self):
        f, a, c, inspection = self.fixture()
        for body, expected_second in ((grounding_payload(f, a, c)[0], b'clean-context'),
                                     (classify_payload(f, a, c, 'overview', inspection)[0], b'marked-context')):
            self.assertNotIn('image_token_budget', body)
            images = [item for item in body['messages'][0]['content'] if item['type'] == 'image_url']
            self.assertEqual(len(images), 2)
            self.assertEqual(base64.b64decode(images[0]['image_url']['url'].split(',')[1]), b'tight-source')
            self.assertEqual(base64.b64decode(images[1]['image_url']['url'].split(',')[1]), expected_second)

    def test_classification_injects_only_literal_four_model_fields(self):
        f, a, c, inspection = self.fixture()
        extra = {**inspection, 'human_category': 'PRIVATE_HUMAN_ANSWER', 'category': 'PRIVATE_REVIEWED_ANSWER',
                 'point_inside_target': 'PRIVATE_GEOMETRY_REVIEW', 'fixture_role': 'PRIVATE_ROLE'}
        original = classify_payload(f, a, c, 'overview', inspection)
        enriched = classify_payload(f, a, c, 'overview', extra)
        self.assertEqual(original, enriched)
        prompt = enriched[0]['messages'][0]['content'][0]['text']
        self.assertIn(json.dumps(inspection, separators=(',', ':')), prompt)
        self.assertNotIn('PRIVATE_', prompt)

    def test_empty_and_unclear_require_abstaining_coordinates(self):
        for visibility in ('empty', 'unclear'):
            good = dict(visibility=visibility, x=-1, y=-1, features='Uniform gray source.')
            self.assertEqual(validated_grounding(self.raw(good)), good)
            for x, y in ((0, 0), (500, -1), (-1, 500)):
                with self.assertRaises(ValueError):
                    validated_grounding(self.raw({**good, 'x': x, 'y': y}))

    def test_visible_coordinates_are_integers_not_bool_or_missing_point(self):
        good = dict(visibility='visible', x=500, y=500, features='Textured horizontal surface.')
        self.assertEqual(validated_grounding(self.raw(good)), good)
        for x, y in ((True, 500), (500.0, 500), (-1, -1), (1001, 10)):
            with self.assertRaises(ValueError):
                validated_grounding(self.raw({**good, 'x': x, 'y': y}))

    def test_invalid_masks_fail_before_image_construction(self):
        source = Image.new('RGB', (20, 10))
        for mask in (np.zeros((10, 20), dtype=bool), np.ones((9, 20), dtype=bool), np.ones((10, 20), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                target_enlargement(source, mask)

    def test_features_reject_ascii_control_characters(self):
        good = dict(visibility='empty', x=-1, y=-1, features='Uniform source.')
        for bad in ('Line\nfeed', 'Tab\there', 'Delete\x7fhere'):
            with self.assertRaises(ValueError):
                validated_grounding(self.raw({**good, 'features':bad}))


if __name__ == '__main__':
    unittest.main()

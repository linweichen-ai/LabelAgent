from copy import deepcopy
import unittest
import numpy as np
from tools.person_guard_envelope import person_envelope_mask


def polygon(identifier,points,label='person',holes=None):
    return {'id':identifier,'label':label,'points':points,'holes':holes or []}


class PersonEnvelopeTests(unittest.TestCase):
    def test_concave_person_envelopes_expand_in_order(self):
        shapes=[polygon('person',[[4,4],[8,4],[8,16],[16,16],[16,20],[4,20]])]
        mask,hull,box=[person_envelope_mask(shapes,32,32,m) for m in ('mask','convex_hull','bounding_box')]
        self.assertTrue(np.all(mask<=hull));self.assertTrue(np.all(hull<=box))
        self.assertGreater(int(hull.sum()),int(mask.sum()));self.assertGreater(int(box.sum()),int(hull.sum()))

    def test_separate_people_do_not_create_a_region_between_them(self):
        shapes=[polygon(1,[[2,2],[6,2],[6,8],[2,8]]),polygon(2,[[22,2],[26,2],[26,8],[22,8]])]
        for mode in ('convex_hull','bounding_box'):
            mask=person_envelope_mask(shapes,32,16,mode)
            self.assertEqual(int(mask[:,10:20].sum()),0)

    def test_holes_are_preserved_by_mask_and_filled_by_envelopes(self):
        shapes=[polygon(1,[[2,2],[22,2],[22,22],[2,22]],holes=[[[8,8],[16,8],[16,16],[8,16]]])]
        self.assertEqual(person_envelope_mask(shapes,32,32,'mask')[12,12],0)
        for mode in ('convex_hull','bounding_box'):
            self.assertEqual(person_envelope_mask(shapes,32,32,mode)[12,12],1)

    def test_shapes_are_immutable_and_pallets_are_not_person_guards(self):
        shapes=[polygon('archived-id',[[2,2],[10,2],[10,10],[2,10]]),
                polygon(2,[[20,20],[28,20],[28,28],[20,28]],'pallet')]
        before=deepcopy(shapes)
        for mode in ('mask','convex_hull','bounding_box'):
            mask=person_envelope_mask(shapes,32,32,mode)
            self.assertEqual(mask[24,24],0)
        self.assertEqual(shapes,before)

    def test_no_people_gives_an_empty_guard(self):
        for mode in ('mask','convex_hull','bounding_box'):
            mask=person_envelope_mask([],32,16,mode)
            self.assertEqual(mask.shape,(16,32));self.assertEqual(int(mask.sum()),0)

    def test_invalid_modes_or_dimensions_fail_closed(self):
        with self.assertRaises(ValueError):person_envelope_mask([],32,16,'global_hull')
        with self.assertRaises(ValueError):person_envelope_mask([],0,16,'mask')
        with self.assertRaises(ValueError):person_envelope_mask([],True,16,'mask')


if __name__=='__main__':unittest.main()

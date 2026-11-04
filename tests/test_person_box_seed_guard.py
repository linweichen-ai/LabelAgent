from copy import deepcopy
import unittest
import cv2
import numpy as np
from tools.person_box_seed_guard import person_box_seed_hits
from tools.person_guard_envelope import person_envelope_mask
from tools.multiview_person_guard import merge_point_views
from tools.person_seed_neighborhood import expand_person_seeds


def person(points, identifier=1, **kwargs):
    return {'id': identifier, 'label': 'person', 'points': points, 'holes': [], **kwargs}


class PersonBoxSeedTests(unittest.TestCase):
    def test_person_concavity_is_guarded_without_new_margin_parameter(self):
        shapes = [person([[2,2],[8,2],[8,32],[38,32],[38,40],[2,40]])]
        pixels = np.array([[25,12], [3,12], [52,12]], dtype=np.int64)
        original = cv2.dilate(person_envelope_mask(shapes,64,64,'mask'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
        self.assertEqual(int(original[12,25]), 0)
        self.assertEqual(person_box_seed_hits(shapes,pixels,64,64).tolist(), [True,True,False])

    def test_disjoint_people_do_not_protect_the_gap_between_instances(self):
        shapes = [person([[2,2],[8,2],[8,20],[2,20]]), person([[48,2],[54,2],[54,20],[48,20]],2)]
        self.assertEqual(person_box_seed_hits(shapes,np.array([[28,12],[5,12],[50,12]]),64,32).tolist(), [False,True,True])

    def test_cross_view_raw_ids_and_unchanged15cm_hop(self):
        shapes = [person([[2,2],[8,2],[8,32],[38,32],[38,40],[2,40]])]
        xyz = np.array([[.10,0,0],[0,0,0],[.40,0,0]], dtype=float)
        hit = person_box_seed_hits(shapes,np.array([[25,12],[55,55]]),64,64)
        merged = merge_point_views(3, {0:{'indices':np.array([2]),'person_hits':np.array([False])},
            1:{'indices':np.array([1,0]),'person_hits':hit}})['person_any']
        self.assertEqual(merged.tolist(),[False,True,False])
        self.assertEqual(expand_person_seeds(xyz,merged,.15).tolist(),[True,True,False])

    def test_input_shapes_and_pixels_immutable_and_no_people_empty(self):
        shapes=[person([[2,2],[12,2],[12,12],[2,12]])]
        pixels=np.array([[4,4],[24,24]])
        before=deepcopy(shapes);stored=pixels.copy()
        person_box_seed_hits(shapes,pixels,32,32)
        self.assertEqual(shapes,before);np.testing.assert_array_equal(pixels,stored)
        shapes[0]['label']='pallet'
        self.assertFalse(person_box_seed_hits(shapes,pixels,32,32).any())

    def test_empty_pixel_set_and_invalid_pixel_coordinates(self):
        self.assertEqual(person_box_seed_hits([],np.empty((0,2),dtype=int),32,32).shape,(0,))
        for pixels in [np.array([[32,1]]),np.array([[-1,0]]),np.array([[1.,2.]]),np.array([1,2])]:
            with self.assertRaises(ValueError):person_box_seed_hits([],pixels,32,32)


if __name__=='__main__':unittest.main()

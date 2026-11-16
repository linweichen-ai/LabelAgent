"""Verify the VLM cutout preserves candidate pixels, holes and source data."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location('candidate_review_views',Path(__file__).resolve().parents[1]/'tools/candidate_review_views.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReviewViewsTest(unittest.TestCase):
    def setUp(self):
        self.source=Image.fromarray(np.arange(80*100*3,dtype=np.uint8).reshape(80,100,3))
        self.shape={'id':'archived','label':'pallet','points':[[20,20],[40,20],[40,40],[20,40]],
            'holes':[[[25,25],[35,25],[35,35],[25,35]]]}

    def test_foreground_is_exact_and_holes_and_outside_are_gray(self):
        pictures,info=module.candidate_review_views(self.source,self.shape)
        crop=pictures[0]
        x0,y0,_,_=info['tight_bounds']
        self.assertEqual(crop.getpixel((21-x0,21-y0)),self.source.getpixel((21,21)))
        self.assertEqual(crop.getpixel((30-x0,30-y0)),(128,128,128))
        self.assertEqual(crop.getpixel((10-x0,10-y0)),(128,128,128))
        self.assertEqual(pictures[1].size,pictures[2].size)

    def test_input_pixels_and_geometry_remain_unchanged(self):
        pixels=self.source.tobytes()
        shape=deepcopy(self.shape)
        module.candidate_review_views(self.source,self.shape)
        self.assertEqual(self.source.tobytes(),pixels)
        self.assertEqual(self.shape,shape)

    def test_multipart_and_image_edges(self):
        parts=[{'points':[[0,0],[10,0],[10,10],[0,10]],'holes':[]},
            {'points':[[80,60],[99,60],[99,79],[80,79]],'holes':[]}]
        shape={'label':'pallet','id':1,'points':parts[0]['points'],'parts':parts}
        pictures,info=module.candidate_review_views(self.source,shape)
        self.assertEqual(info['tight_bounds'],(0,0,100,80))
        self.assertEqual(pictures[0].getpixel((95,75)),self.source.getpixel((95,75)))
        self.assertEqual(pictures[0].getpixel((50,40)),(128,128,128))

    def test_invalid_geometry_or_background_rejected(self):
        with self.assertRaises(ValueError):
            module.candidate_review_views(self.source,{**self.shape,'points':[[1,1]]})
        with self.assertRaises(ValueError):
            module.candidate_review_views(self.source,self.shape,background=(300,0,0))


if __name__=='__main__':
    unittest.main()

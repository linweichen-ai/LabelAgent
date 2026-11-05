import unittest
import numpy as np
from PIL import Image
from tools.centered_target_views import centered_target_views

class CenteredChecks(unittest.TestCase):
    def data(self):
        pixels=np.random.default_rng(2).integers(0,256,(180,200,3),dtype=np.uint8)
        target=np.zeros((180,200),bool);target[104:112,160:192]=True
        return pixels,target,(120,50,200,170)

    def test_outside_verified_pixels_do_not_affect_either_input(self):
        pixels,target,box=self.data();changed=np.zeros_like(pixels);changed[50:170,120:200]=pixels[50:170,120:200]
        a,_=centered_target_views(Image.fromarray(pixels),target,box)
        b,_=centered_target_views(Image.fromarray(changed),target,box)
        self.assertEqual({k:v.tobytes() for k,v in a.items()},{k:v.tobytes() for k,v in b.items()})

    def test_thin_original_target_retained_in_both_scales(self):
        pixels,target,box=self.data();views,meta=centered_target_views(Image.fromarray(pixels),target,box)
        self.assertTrue(all(r['retained_target_pixels']==target.sum() for r in meta['views'].values()))
        self.assertTrue(all(v.size==(512,512) for v in views.values()))
        self.assertEqual(meta['views']['overview']['factor'],2)
        self.assertEqual(meta['views']['linked']['factor'],3)

    def test_nested_boxes_stay_within_verified_region(self):
        pixels,target,box=self.data();_,meta=centered_target_views(Image.fromarray(pixels),target,box)
        a=meta['views']['overview']['source_box'];b=meta['views']['linked']['source_box']
        self.assertTrue(box[0]<=b[0]<=a[0]<a[2]<=b[2]<=box[2])
        self.assertTrue(box[1]<=b[1]<=a[1]<a[3]<=b[3]<=box[3])

    def test_pixel_and_geometry_inputs_unmodified(self):
        pixels,target,box=self.data();image=Image.fromarray(pixels);before=(image.tobytes(),target.tobytes())
        centered_target_views(image,target,box);self.assertEqual(before,(image.tobytes(),target.tobytes()))

    def test_clip_and_invalid_source_rejected(self):
        pixels,target,box=self.data()
        with self.assertRaises(ValueError):centered_target_views(Image.fromarray(pixels),target,(0,0,20,20))
        with self.assertRaises(ValueError):centered_target_views(Image.new('RGB',(30,30)),target,box)

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from PIL import Image
from tools.boxed_target_context import boxed_target_context
from tools.centered_target_views import centered_target_views

class BoxedContextChecks(unittest.TestCase):
    def test_same_original_crop_and_complete_foreground_with_two_distinct_cues(self):
        source=Image.new('RGB',(200,180),(70,80,90));mask=np.zeros((180,200),bool);mask[80:90,90:120]=True;box=(50,40,160,140)
        views,meta=boxed_target_context(source,mask,box);old,_=centered_target_views(source,mask,box)
        self.assertEqual(views['overview'].tobytes(),old['linked'].tobytes());self.assertNotEqual(views['overview'].tobytes(),views['linked'].tobytes())
        self.assertEqual(meta['retained_target_pixels'],mask.sum())
        self.assertFalse(np.any(np.all(np.array(views['linked'])==(255,150,0),axis=2)))
    def test_pixels_outside_verified_region_cannot_change_either_input(self):
        p=np.random.default_rng(6).integers(0,256,(180,200,3),np.uint8);q=np.zeros_like(p);q[40:140,50:160]=p[40:140,50:160];mask=np.zeros((180,200),bool);mask[80:90,90:120]=True
        a,_=boxed_target_context(Image.fromarray(p),mask,(50,40,160,140));b,_=boxed_target_context(Image.fromarray(q),mask,(50,40,160,140))
        self.assertEqual({k:v.tobytes() for k,v in a.items()},{k:v.tobytes() for k,v in b.items()})

if __name__=='__main__':unittest.main()

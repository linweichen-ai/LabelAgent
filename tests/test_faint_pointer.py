import unittest
import numpy as np
from PIL import Image
from tools.faint_pointer import faint_pointer

class FaintPointerTests(unittest.TestCase):
    def test_only_existing_arrow_pixels_change(self):
        contour=np.full((12,12,3),127,dtype=np.uint8);contour[2,:,0]=255
        arrow=contour.copy();arrow[4:9,5]=[255,150,0]
        out=np.array(faint_pointer(Image.fromarray(contour),Image.fromarray(arrow)))
        changed=np.any(arrow!=contour,axis=2)
        np.testing.assert_array_equal(out[~changed],contour[~changed])
        np.testing.assert_array_equal(np.any(out!=contour,axis=2),changed)
        np.testing.assert_array_equal(out[changed],(contour[changed]*.75+arrow[changed]*.25).astype(np.uint8))
    def test_identical_panels_unchanged(self):
        im=Image.new('RGB',(12,12),(128,128,128));self.assertEqual(faint_pointer(im,im).tobytes(),im.tobytes())
    def test_mismatched_panels_rejected(self):
        with self.assertRaises(ValueError):faint_pointer(Image.new('RGB',(12,12)),Image.new('RGB',(13,12)))
if __name__=='__main__':unittest.main()

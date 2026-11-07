import unittest,numpy as np
from PIL import Image
from bbox_center_views import views


class CenterTests(unittest.TestCase):
    def test_asymmetric_thick_foot_cannot_move_bbox_center(self):
        m=np.zeros((180,360),np.uint8);m[70:75,30:280]=1;m[65:95,260:290]=1;m[70,4]=1
        images,bindings=views(Image.new('RGB',(360,180),'white'),m)
        for b in bindings:
            self.assertEqual(b['foreground_anchor'],[146,79]);self.assertTrue(b['all_candidate_pixels_retained']);self.assertEqual(np.prod(b['geometry']['grid_width_height']),256)

    def test_disconnected_and_image_edge_preserved(self):
        m=np.zeros((80,160),np.uint8);m[0:9,0:40]=1;m[65:80,130:160]=1
        ims,bindings=views(Image.new('RGB',(160,80),'white'),m)
        self.assertEqual(bindings[0]['candidate_pixels'],810)
        for b in bindings:
            x1,y1,x2,y2=b['source_box'];self.assertEqual(int(m[y1:y2,x1:x2].sum()),810)


if __name__=='__main__':unittest.main()

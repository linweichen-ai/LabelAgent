import sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
here=Path(__file__).resolve()
for root in [here.parent]+[p/'tools/labelagent_experiments' for p in here.parents]:
 if (root/'rectangular_views.py').exists():sys.path.insert(0,str(root));break
from rectangular_views import rectangular_views,render
class Views(unittest.TestCase):
 def test_disconnected_and_border_pixels_all_retained(self):
  m=np.zeros((90,150),np.uint8);m[0,0]=1;m[35:42,20:130]=1;m[89,149]=1;rgb=np.random.default_rng(3).integers(0,256,(90,150,3),np.uint8);source=Image.fromarray(rgb)
  _,bindings=rectangular_views(source,m)
  for b in bindings:
   self.assertEqual(b['candidate_pixels'],m.sum());self.assertTrue(b['center_on_mask']);box=b['canvas_box'];clip=b['source_box'];_,native,_=render(source,b);actual=np.asarray(native)[clip[1]-box[1]:clip[3]-box[1],clip[0]-box[0]:clip[2]-box[0]];np.testing.assert_array_equal(actual,rgb[clip[1]:clip[3],clip[0]:clip[2]])
 def test_thin_candidate_has_rectangular_content(self):
  m=np.zeros((120,220),np.uint8);m[56:63,25:195]=1;_,b=rectangular_views(Image.new('RGB',(220,120),'white'),m);self.assertGreater(b[0]['native_canvas_size'][0],5*b[0]['native_canvas_size'][1]);self.assertEqual(b[0]['output_size'],[1024,1024])
 def test_matching_gray_stays_uniform_white_padding_differs(self):
  m=np.zeros((120,220),np.uint8);m[56:63,25:195]=1;_,b=rectangular_views(Image.new('RGB',(220,120)),m);gray=render(Image.new('RGB',(220,120),(127,127,127)),b[0])[0];self.assertTrue((np.asarray(gray)==127).all());white=render(Image.new('RGB',(220,120),'white'),b[0])[0];self.assertGreater(len(np.unique(np.asarray(white))),1)
 def test_empty_candidate_rejected(self):
  with self.assertRaises(ValueError):rectangular_views(Image.new('RGB',(10,10)),np.zeros((10,10),np.uint8))
if __name__=='__main__':unittest.main()

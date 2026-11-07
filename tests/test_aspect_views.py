import sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
here=Path(__file__).resolve()
for root in [here.parent]+[p/'tools/labelagent_experiments' for p in here.parents]:
 if (root/'aspect_views.py').exists():sys.path.insert(0,str(root));break
from rectangular_views import rectangular_views
from aspect_views import aspect_view
class Aspect(unittest.TestCase):
 def test_wide_and_tall_keep_area_budget_and_increase_candidate_sampling(self):
  for shape in [(160,500),(500,160)]:
   h,w=shape;m=np.zeros(shape,np.uint8)
   if w>h:m[70:78,30:470]=1
   else:m[30:470,70:78]=1
   source=Image.new('RGB',(w,h),'white');_,bindings=rectangular_views(source,m);image,meta=aspect_view(source,bindings[0]);self.assertEqual(image.width*image.height,256*32*32);self.assertGreater(meta['candidate_area_fraction_estimate'],meta['square_candidate_fraction']);self.assertTrue(meta['no_crop_of_native_rectangle'])
 def test_uniform_gray_no_spurious_shape(self):
  source=Image.new('RGB',(500,160),(127,127,127));m=np.zeros((160,500),np.uint8);m[70:78,30:470]=1;b=rectangular_views(source,m)[1][0];image,_=aspect_view(source,b);self.assertTrue((np.asarray(image)==127).all())
 def test_disconnected_border_pixels_keep_full_native_rectangle(self):
  source=Image.new('RGB',(180,100));m=np.zeros((100,180),np.uint8);m[0,0]=m[-1,-1]=1;m[45:55,40:140]=1;_,bs=rectangular_views(source,m)
  for b in bs:
   image,meta=aspect_view(source,b);self.assertTrue(b['all_candidate_pixels_retained']);self.assertTrue(meta['no_crop_of_native_rectangle']);self.assertLessEqual(abs(meta['fitted_size'][0]/meta['native_size'][0]-meta['fitted_size'][1]/meta['native_size'][1]),1/min(meta['native_size']))
if __name__=='__main__':unittest.main()

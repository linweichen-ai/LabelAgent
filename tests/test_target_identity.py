"""Preserve target evidence and output while changing input/point geometry."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).resolve().parents[1]/'tools/labelagent_experiments')]
import numpy as np
from PIL import Image
from target_identity_policy import centered_photos,deck_foot_prompts,png,extent_views
class TargetIdentityTests(unittest.TestCase):
 def test_center_is_foreground_and_every_target_pixel_retained_for_each_context(self):
  source=Image.new('RGB',(200,160),'white');m=np.zeros((160,200),np.uint8);m[52:58,38:169]=1;m[90:99,160:169]=1;before=m.copy();views,bind=centered_photos(source,m)
  for b in bind:
   x,y=b['foreground_anchor'];self.assertTrue(m[y,x]);self.assertTrue(b['all_target_pixels_retained']);self.assertEqual(b['target_pixels'],int(m.sum()))
  self.assertEqual(len(views),2);np.testing.assert_array_equal(m,before)
 def test_full_control_mask_does_not_select_a_corner_as_distance_transform_anchor(self):
  _,bind=centered_photos(Image.new('RGB',(100,100),'gray'),np.ones((100,100),np.uint8));x,y=bind[0]['foreground_anchor'];self.assertLessEqual(abs(x-49),1);self.assertLessEqual(abs(y-49),1)
 def test_pixels_outside_each_bound_source_region_do_not_reach_that_view(self):
  a=np.random.default_rng(6).integers(0,256,(180,220,3),dtype=np.uint8);m=np.zeros((180,220),np.uint8);m[70:77,90:130]=1;views,binding=centered_photos(Image.fromarray(a),m)
  for b in binding:
   mod=np.zeros_like(a);x1,y1,x2,y2=b['source_box'];mod[y1:y2,x1:x2]=a[y1:y2,x1:x2];other,_=centered_photos(Image.fromarray(mod),m);self.assertEqual(png(views[b['view']]),png(other[b['view']]))
 def test_prompt_anchors_span_deck_and_lower_connection_without_mask_prior(self):
  a=np.zeros((120,180,3),np.uint8)
  for y in range(0,120,12):a[y:y+6]=180
  m=np.zeros((120,180),np.uint8);m[10:110,20:160]=1;before=m.copy();rows,_=deck_foot_prompts(Image.fromarray(a),{'mask':m,'box':[20,10,160,110],'area':int(m.sum())});self.assertTrue(rows)
  for r in rows:
   self.assertIsNone(r['mask_prior']);self.assertLessEqual(len(r['points']),7);self.assertEqual(sum(p['label']==1 for p in r['points']),3)
   for p in r['points']:self.assertTrue(m[p['y'],p['x']])
  np.testing.assert_array_equal(m,before)
if __name__=='__main__':unittest.main()

"""Regression checks for lossless geometry, source isolation and instance identity."""
import sys,json,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
ROOT=Path(__file__).resolve().parents[2]
config_file=ROOT/'task_workspaces/labelagent_instance_separation_20261105/config.json'
if config_file.exists():sys.path.insert(0,json.loads(config_file.read_text(encoding='utf-8'))['source_snapshot'])
else:sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/labelagent_experiments'))
import numpy as np
from PIL import Image
from source.agent.edit_agent import _rasterize
from instance_separation_geometry import exact_binary_shape,visible_seam_prompts
from instance_evidence_views import target_views,png,single_physical_base
class InstanceInvariants(unittest.TestCase):
 def test_repeated_evidence_overrides_single_enum_without_changing_old_gate(self):
  from instance_identity_consistency import single_physical_base as checked,repeated_control_rejected
  d={'object':'pallet_base','category':'pallet_base','instance_status':'single','evidence':'a stack of blue pallet bases with slats and feet'}
  self.assertFalse(checked(d));self.assertTrue(repeated_control_rejected(d))
  self.assertTrue(checked({**d,'evidence':'one wooden pallet base with slats and feet'}))
 def test_reparse_cannot_accept_ambiguous_or_nonpallet_subject(self):
  from instance_identity_consistency import single_physical_base as checked
  d={'object':'pallet_base','category':'pallet_base','instance_status':'single','evidence':'visible deck and feet'}
  for override in [{'object':'possibly a pallet'},{'object':'pallet jack'},{'instance_status':'uncertain'},{'category':'machine'},{'evidence':'several visible bases'}]:self.assertFalse(checked({**d,**override}))
 def test_holes_and_disconnected_pixels_survive_or_entire_mask_is_rejected(self):
  m=np.zeros((100,140),np.uint8);m[10:40,12:80]=1;m[15:35,20:70]=0;m[50:52,100:110]=1
  s=exact_binary_shape(m);self.assertIsNotNone(s);np.testing.assert_array_equal(m,_rasterize(s,140,100))
  rng=np.random.default_rng(29);complex_mask=(rng.random((100,140))>.88).astype('uint8');s=exact_binary_shape(complex_mask)
  if s is not None:np.testing.assert_array_equal(complex_mask,_rasterize(s,140,100))
 def test_outside_pixels_cannot_change_evidence(self):
  a=np.random.default_rng(5).integers(0,256,(150,200,3),dtype=np.uint8);m=np.zeros((150,200),np.uint8);m[42:57,60:140]=1;m[47:52,90:100]=0
  views,b=target_views(Image.fromarray(a),m);x1,y1,x2,y2=b['source_box'];modified=np.zeros_like(a);modified[y1:y2,x1:x2]=a[y1:y2,x1:x2]
  other,b2=target_views(Image.fromarray(modified),m)
  self.assertEqual(b['original_target_pixels'],b['retained_target_pixels']);self.assertTrue(b['pointer_in_target']);self.assertFalse(b['semantic_location_verified'])
  self.assertEqual([png(v) for v in views.values()],[png(v) for v in other.values()])
 def test_stack_blank_uncertainty_are_not_single_base(self):
  good={'object':'wooden pallet base','category':'pallet_base','instance_status':'single','evidence':'visible deck and feet'};self.assertTrue(single_physical_base(good))
  for override in [{'instance_status':'repeated'},{'instance_status':'uncertain'},{'object':'possibly a pallet'},{'object':'pallet jack'},{'category':'floor'},{'evidence':''}]:self.assertFalse(single_physical_base({**good,**override}))
 def test_seam_clicks_stay_on_visible_parent_and_separate_neighbor_layers(self):
  a=np.zeros((120,180,3),np.uint8)
  for y in range(0,120,12):a[y:y+6]=190
  m=np.zeros((120,180),np.uint8);m[10:110,20:160]=1;p={'mask':m,'box':[20,10,160,110],'area':int(m.sum())}
  prompts,_=visible_seam_prompts(Image.fromarray(a),p);self.assertTrue(prompts);self.assertLessEqual(len(prompts),2)
  for row in prompts:
   self.assertLessEqual(len(row['points']),7);lo,hi=row['seam_interval'];self.assertTrue(row['no_mask_prior'])
   for q in row['points']:
    self.assertTrue(m[q['y'],q['x']]);self.assertEqual(lo<=q['y']<=hi,q['label']==1)
if __name__=='__main__':unittest.main()

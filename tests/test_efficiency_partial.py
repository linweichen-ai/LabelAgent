import io, sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/labelagent_experiments'))
from source.agent import workflow
from source.agent.edit_agent import _rasterize,_shape_from_mask,_candidate_gap_regions
from ground_action_review_v4 import approved_gap_sam_operation,gap_polygon

class CompleteSubsetChecks(unittest.TestCase):
 def test_real_inclusive_open_gap_bbox_includes_last_row_and_column(self):
  import cv2
  mask=np.ones((64,64),np.uint8);mask[10:25,62:64]=0
  shape=_shape_from_mask(mask,'floor',1);hole=next(h for h in _candidate_gap_regions(shape,64,64) if h['kind']=='open_gap')
  selected=np.zeros_like(mask);cv2.fillPoly(selected,[np.array(gap_polygon(hole,64,64),np.int32)],1)
  self.assertEqual(int((selected&hole['mask']).sum()),int(hole['mask'].sum()))
 def test_uncertain_keep_does_not_become_negative_prompt(self):
  mask=np.ones((64,64),np.uint8);mask[10:18,10:18]=0;mask[28:36,28:36]=0;mask[45:53,45:53]=0
  shape=_shape_from_mask(mask,'floor',1);holes=_candidate_gap_regions(shape,64,64)
  assess=[{'hole_id':h['id'],'semantic':'ground' if i<2 else 'non_ground','reason':'uncertain' if i==2 else 'visible'} for i,h in enumerate(holes)]
  op=approved_gap_sam_operation(holes,assess,[shape],64,64,set())
  self.assertEqual(len(op['points']),3);self.assertTrue(all(p['label']==1 for p in op['points']))
  self.assertEqual(len(op['reviewed_gap_ids']),2)
 def test_warning_keeps_only_wholly_satisfied_gap_and_pending_gets_real_next_round(self):
  mask=np.zeros((64,64),np.uint8);mask[2:62,2:62]=1
  mask[10:18,10:18]=0;mask[28:36,28:36]=0
  shape=_shape_from_mask(mask,'floor',1);holes=_candidate_gap_regions(shape,64,64)
  self.assertEqual(len(holes),2)
  full=mask|holes[0]['mask'];raw=full.copy();ys,xs=np.where(holes[1]['mask']);raw[ys[0],xs[0]]=1
  sam_shape=_shape_from_mask(raw,'floor',1);seen=[]
  def review(config,image,overlay,description,shapes,labels,history,**kw):
   seen.append(kw);hs=kw['holes']
   assess=[{'hole_id':h['id'],'semantic':'ground','reason':'floor'} for h in hs]
   if len(seen)==1:ops=[approved_gap_sam_operation(hs,assess,kw['candidate_geometry'],64,64,set())]
   elif len(seen)==2:
    self.assertEqual(kw['sam_comparison']['reviewed_gap_ids'],[holes[0]['id']])
    self.assertEqual(kw['sam_comparison']['unresolved_gap_ids'],[holes[1]['id']])
    self.assertTrue(np.array_equal(_rasterize(kw['candidate_geometry'][0],64,64),full))
    ops=[{'operation':'fill_gap','shape_ids':[1],'hole_id':hs[0]['id'],'polygon':gap_polygon(hs[0],64,64),'reason':'floor'}]
   else:ops=[]
   decision={'status':'continue' if ops else 'done','reason':'actual observation','hole_assessment':assess,'operations':ops}
   if kw.get('sam_comparison'):decision['sam_assessment']={'improved':True,'reason':'successful full subset'}
   return workflow._validate_review_result(decision,hs,shapes,labels,64,64,kw.get('sam_comparison'))
  class Seed(workflow.AnnotationAgent):
   def generate(self):self.shapes=[shape];self.generated=1
  stream=io.BytesIO();Image.new('RGB',(64,64),'gray').save(stream,'PNG')
  with patch.object(workflow,'review',side_effect=review),patch.object(workflow,'sam_request',return_value={'parts':sam_shape['parts'],'warnings':['SAM 3 未覆盖全部正点；已保留候选轮廓，请继续调整提示点']}) as sam:
   result=Seed({'llm_batch_reviewed_gaps':True,'sam_accept_complete_gap_subsets':True},stream.getvalue(),{'description':'floor','prompts':[]},['floor'],3).execute()
  self.assertEqual(sam.call_count,1);self.assertTrue(result['agent']['converged']);self.assertEqual(result['agent']['rounds'],3)
  self.assertTrue(np.array_equal(_rasterize(result['shapes'][0],64,64),mask|holes[0]['mask']|holes[1]['mask']))
  outcome=result['agent']['audit'][0]['tool_results'][0]
  self.assertEqual(outcome['accepted_gap_coverage'],1.0);self.assertTrue(outcome['partial_batch']);self.assertTrue(outcome['raw_model_warnings'])

if __name__=='__main__':unittest.main(verbosity=2)

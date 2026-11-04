import io,sys,unittest,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/labelagent_experiments'))
from source.agent import workflow
from source.agent.edit_agent import _rasterize,_shape_from_mask,_candidate_gap_regions
from ground_action_review_v5 import approved_gap_sam_operation,gap_polygon
from residual_gap_policy import choose_residual_schedule

class RecoveryPolicyTests(unittest.TestCase):
 def test_full_cumulative_budget_never_adds_a_thirty_third_point(self):
  mask=np.ones((128,128),np.uint8)
  for y in [12,48,84]:
   for x in [12,48,84]:mask[y:y+8,x:x+8]=0
  shape=_shape_from_mask(mask,'floor',1);holes=_candidate_gap_regions(shape,128,128)
  assessments=[{'hole_id':h['id'],'semantic':'ground','reason':'reviewed'} for h in holes]
  batch=approved_gap_sam_operation(holes,assessments,[shape],128,128,set());previous=copy.deepcopy(batch['points'])
  for x in range(128):
   if len(previous)>=32:break
   point={'x':x,'y':2,'label':1}
   if point not in previous:previous.append(point)
  self.assertEqual(len(previous),32)
  ops=[{'operation':'fill_gap','shape_ids':[1],'hole_id':h['id'],'polygon':gap_polygon(h,128,128),'reason':'reviewed'} for h in holes]
  chosen=choose_residual_schedule(holes,assessments,ops,[shape],128,128,{1:previous},{1:{'mask':_rasterize(shape,128,128)}},approved_gap_sam_operation)
  self.assertEqual(len(chosen),3);self.assertEqual(len(chosen[-1]['points']),32)
  self.assertEqual(chosen[-1]['points'],previous)
 def test_real_controller_keeps_three_ops_and_actual_next_round_review(self):
  mask=np.zeros((128,128),np.uint8);mask[2:126,2:126]=1
  for y in [12,48,84]:
   for x in [12,48,84]:mask[y:y+8,x:x+8]=0
  mask[108:116,108:116]=0
  shape=_shape_from_mask(mask,'floor',1);base=_rasterize(shape,128,128);holes=_candidate_gap_regions(shape,128,128)
  ground={h['id'] for h in holes[:9]};kept=holes[9]['mask'];first=base|np.maximum.reduce([h['mask'] for h in holes[:3]])
  full=_shape_from_mask(np.ones_like(base),'floor',1);firstshape=_shape_from_mask(first,'floor',1);seen=[]
  def review(config,image,overlay,description,shapes,labels,history,**kw):
   hs=kw['holes'];geometry=kw['candidate_geometry'];seen.append(copy.deepcopy(geometry))
   assessments=[{'hole_id':h['id'],'semantic':'ground' if h['id'] in ground else 'non_ground','reason':'prior image evidence'} for h in hs]
   ops=[{'operation':'fill_gap','shape_ids':[h['shape_id']],'hole_id':h['id'],'polygon':gap_polygon(h,128,128),'reason':'reviewed'} for h in hs if h['id'] in ground]
   if len(seen)==1:chosen=[approved_gap_sam_operation(hs,assessments,geometry,128,128,set())]
   elif ops:chosen=choose_residual_schedule(hs,assessments,ops,geometry,128,128,config['_gap_points_by_shape'],config['_gap_feedback'],approved_gap_sam_operation)
   else:chosen=[]
   result={'status':'continue' if chosen else 'done','reason':'real observation required','hole_assessment':assessments,'operations':chosen}
   if kw.get('sam_comparison'):result['sam_assessment']={'improved':True,'reason':'before/after pixels seen'}
   return workflow._validate_review_result(result,hs,shapes,labels,128,128,kw.get('sam_comparison'))
  class Seed(workflow.AnnotationAgent):
   def generate(self):self.shapes=[shape];self.generated=1
  stream=io.BytesIO();Image.new('RGB',(128,128),'gray').save(stream,'PNG')
  with patch.object(workflow,'review',side_effect=review),patch.object(workflow,'sam_request',side_effect=[{'parts':firstshape['parts']},{'parts':full['parts']}]) as sam:
   result=Seed({'llm_residual_gap_recovery':True,'llm_batch_reviewed_gaps':True,'sam_accept_complete_gap_subsets':True},stream.getvalue(),{'description':'floor','prompts':[]},['floor'],3).execute()
  self.assertTrue(result['agent']['converged']);self.assertEqual(result['agent']['rounds'],3);self.assertEqual(sam.call_count,2)
  self.assertEqual([o['operation'] for o in result['agent']['audit'][1]['decision']['operations']],['fill_gap','fill_gap','sam_points'])
  after=_rasterize(result['shapes'][0],128,128);target=np.maximum.reduce([h['mask'] for h in holes[:9]])
  self.assertTrue(np.array_equal(after,base|target));self.assertEqual(int((after&kept).sum()),0)
  for audit in result['agent']['audit']:
   self.assertLessEqual(len(audit['decision']['operations']),3)
   for row in audit.get('tool_results',[]):self.assertLessEqual(row.get('point_count',0),32)
  second=result['agent']['audit'][1]['tool_results'][-1]
  self.assertLess(second['active_request_point_count'],second['point_count'])
  original_resolved=[p for p in sam.call_args_list[0].kwargs['points'] if any(h['mask'][int(p['y']),int(p['x'])] for h in holes[:3])]
  self.assertTrue(original_resolved);self.assertTrue(all(p not in sam.call_args_list[1].kwargs['points'] for p in original_resolved))

if __name__=='__main__':unittest.main(verbosity=2)

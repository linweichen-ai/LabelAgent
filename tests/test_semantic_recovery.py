import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools/labelagent_experiments"))
import sys,unittest,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'task_workspaces/labelagent_semantic_recovery_20261105/source_snapshot'))
from semantic_evidence_gate import pallet_evidence,CAPTION_PROMPT
from semantic_subject_guard import compatible_pallet_subject
from spatial_gap_recovery import choose_context_first
from ground_action_review_v6 import approved_gap_sam_operation,gap_polygon
from source.agent.edit_agent import _shape_from_mask,_candidate_gap_regions,_rasterize
class SemanticRecoveryTests(unittest.TestCase):
 def test_neutral_observation_cannot_authorize_category_echo_or_blank(self):
  for caption in ['uniform gray surface','uniform white image','no pallet is visible','uncertain pallet outline','a gray floor cleaning machine']:
   self.assertFalse(pallet_evidence({'object':caption}),caption)
  for caption in ['stack of blue plastic pallets','a wooden pallet with feet']:
   self.assertTrue(pallet_evidence({'object':caption}),caption)
  self.assertNotIn('pallet',CAPTION_PROMPT.lower())
  for text in ['a black electric pallet jack with blue pallet in background','a forklift carrying a pallet','possibly a wooden pallet']:
   self.assertFalse(compatible_pallet_subject({'object':text}),text)
  self.assertTrue(compatible_pallet_subject({'object':'a stack of blue plastic pallets'}))
 def test_context_first_uses_exactly_two_individual_operations_and_one_SAM(self):
  mask=np.ones((128,128),np.uint8)
  for x,y in [(12,12),(40,40),(65,65),(88,88),(112,112)]:mask[y:y+6,x:x+6]=0
  shape=_shape_from_mask(mask,'floor',1);original=_rasterize(shape,128,128)
  holes=_candidate_gap_regions(shape,128,128);assess=[{'hole_id':h['id'],'semantic':'ground','reason':'reviewed'} for h in holes]
  ops=[{'operation':'fill_gap','shape_ids':[1],'hole_id':h['id'],'polygon':gap_polygon(h,128,128),'reason':'reviewed'} for h in holes]
  scheduled=choose_context_first(holes,assess,ops,[shape],128,128,approved_gap_sam_operation)
  self.assertEqual([op['operation'] for op in scheduled],['fill_gap','fill_gap','sam_points'])
  self.assertLessEqual(len(scheduled[-1]['points']),32)
  self.assertEqual(len(scheduled[-1]['reviewed_gap_ids']),len(holes)-2)
  self.assertFalse(set(o['hole_id'] for o in scheduled[:2])&set(scheduled[-1]['reviewed_gap_ids']))
  self.assertTrue(np.array_equal(original,_rasterize(shape,128,128)))
if __name__=='__main__':unittest.main(verbosity=2)

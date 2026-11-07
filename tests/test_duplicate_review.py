"""Contract regressions for VLM-directed duplicate edits and pixel preservation."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from source.agent import workflow
from source.agent.edit_agent import _rasterize


def shape(identifier, label='person', left=10):
    points=[[left,10],[left+18,10],[left+18,35],[left,35]]
    return {'id':identifier,'label':label,'points':points,'holes':[],
            'parts':[{'points':points,'holes':[]}]}


class DuplicateReviewTests(unittest.TestCase):
    def setUp(self):
        data=io.BytesIO()
        Image.new('RGB',(80,60),'gray').save(data,'JPEG')
        self.image=data.getvalue()
        self.description={'description':'Visible person outlines.',
                          'prompts':[{'label':'person','text':'person'}]}
        self.config={'llm_prompt_profile':'compact','llm_duplicate_review':True}
        self.done={'status':'done','reason':'No supported edit.','operations':[],'hole_assessment':[]}

    def agent(self,seeds):
        agent=workflow.AnnotationAgent(self.config,self.image,self.description,['person'],max_rounds=3)
        def replay():
            agent.shapes=deepcopy(seeds)
            agent.next_id=max(item['id'] for item in seeds)+1
        agent.generate=replay
        return agent

    def test_equal_boxes_and_different_classes_do_not_imply_equal_masks(self):
        first,second=shape(1),shape(2)
        triangle=shape(3)
        triangle['points']=triangle['parts'][0]['points']=[[10,10],[28,10],[28,35]]
        other_class=shape(4,'pallet')
        before=deepcopy([first,second,triangle,other_class])
        result=workflow._exact_duplicate_groups(before,80,60)
        self.assertEqual([item['shape_ids'] for item in result],[[1,2]])
        self.assertEqual(before,[first,second,triangle,other_class])

    def test_vlm_merge_is_reobserved_and_preserves_every_candidate_mask(self):
        seeds=[shape(1),shape(2),shape(3,left=45)]
        with patch.object(workflow,'chat',side_effect=[{'evidence':'present','reason':'A person is visible.'},
                {'decision':'merge','reason':'One visible person appears twice.'},self.done]) as call:
            result=self.agent(seeds).execute()
        self.assertEqual(call.call_count,3)
        self.assertTrue(result['agent']['converged'])
        self.assertEqual(result['agent']['rounds'],2)
        self.assertEqual(len(result['shapes']),2)
        untouched=next(item for item in result['shapes'] if item['id']==3)
        self.assertEqual(untouched,seeds[2])
        merged=next(item for item in result['shapes'] if item['id']!=3)
        self.assertTrue(np.array_equal(_rasterize(merged,80,60),_rasterize(seeds[0],80,60)))
        self.assertEqual(merged['label'],'person')
        focused_image=Image.open(io.BytesIO(call.call_args_list[0].args[2]))
        general_image=Image.open(io.BytesIO(call.call_args_list[-1].args[2]))
        self.assertLess(focused_image.width,80)
        self.assertEqual(general_image.size,(80,60))

    def test_vlm_keep_runs_general_review_without_silently_merging(self):
        seeds=[shape(1),shape(2)]
        with patch.object(workflow,'chat',side_effect=[{'evidence':'present','reason':'A person is visible.'},
                {'decision':'keep','reason':'Separate objects cannot be ruled out.'},self.done]) as call:
            result=self.agent(seeds).execute()
        self.assertEqual(call.call_count,3)
        self.assertEqual(result['shapes'],seeds)
        self.assertFalse(result['applied'])

    def test_absent_image_evidence_cannot_trigger_a_merge(self):
        seeds=[shape(1),shape(2)]
        with patch.object(workflow,'chat',side_effect=[{'evidence':'absent','reason':'No person is visible.'},self.done]) as call:
            result=self.agent(seeds).execute()
        self.assertEqual(call.call_count,2)
        self.assertEqual(result['shapes'],seeds)
        self.assertFalse(result['applied'])

    def test_invalid_presence_decision_preserves_geometry_and_stops(self):
        seeds=[shape(1),shape(2)]
        with patch.object(workflow,'chat',return_value={'evidence':'unknown','reason':'Invalid protocol'}):
            result=self.agent(seeds).execute()
        self.assertEqual(result['shapes'],seeds)
        self.assertFalse(result['agent']['converged'])
        self.assertEqual(result['agent']['stop_reason'],'error')

    def test_model_failure_keeps_the_original_candidates_and_reports_failure(self):
        seeds=[shape(1),shape(2)]
        with patch.object(workflow,'chat',side_effect=OSError('Unavailable model')):
            result=self.agent(seeds).execute()
        self.assertEqual(result['shapes'],seeds)
        self.assertFalse(result['agent']['converged'])
        self.assertEqual(result['agent']['stop_reason'],'error')

    def test_experimental_focus_is_disabled_by_default(self):
        seeds=[shape(1),shape(2)]
        self.config.pop('llm_duplicate_review')
        with patch.object(workflow,'_review_exact_overlap') as focused,patch.object(workflow,'chat',return_value=self.done):
            result=self.agent(seeds).execute()
        focused.assert_not_called()
        self.assertEqual(result['shapes'],seeds)

    def test_pending_sam_assessment_cannot_be_bypassed_by_duplicate_focus(self):
        seeds=[shape(1),shape(2)]
        summaries=[workflow._shape_summary(item) for item in seeds]
        answer={**self.done,'sam_assessment':{'improved':False,'reason':'No verified gain.'}}
        with patch.object(workflow,'_review_exact_overlap') as focused,patch.object(workflow,'chat',return_value=answer):
            result=workflow.review(self.config,self.image,self.image,self.description,summaries,['person'],[],
                sam_comparison={'shape_id':1,'before_image':self.image},
                duplicate_groups=workflow._exact_duplicate_groups(seeds,80,60))
        focused.assert_not_called()
        self.assertFalse(result['sam_assessment']['improved'])


if __name__=='__main__':
    unittest.main()

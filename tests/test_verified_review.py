import io,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from contextvars import ContextVar
import numpy as np
from PIL import Image
from source.agent import workflow
from source.agent.edit_agent import _shape_from_mask
from source.agent.protocol_schema import review_schema
from tools.ground_action_review_v2 import install_review_batches,ordinary_floor_confirmed

class VerifiedReviewChecks(unittest.TestCase):
    def test_no_holes_cannot_invent_unused_assessment(self):
        with self.assertRaises(ValueError):workflow._validate_review_result(
            {'status':'done','operations':[],'hole_assessment':[{'hole_id':'unused','semantic':'non_ground','reason':'none'}]},[],[],[],40,40)
    def test_identity_schema_omits_unenforceable_zero_array(self):
        self.assertNotIn('hole_assessment',review_schema(['floor'],[{'id':1}],[],object_only=True)['properties'])
    def test_raw_description_category_conflict_vetoes_fill(self):
        self.assertFalse(ordinary_floor_confirmed({'category':'bare_floor_or_floor_marking','object':'a gray floor cleaning machine'}))
        self.assertTrue(ordinary_floor_confirmed({'category':'bare_floor_or_floor_marking','object':'concrete with painted stripe'}))
    def test_robot_raw_image_veto_and_original_person_protection(self):
        mask=np.ones((40,40),np.uint8);mask[10:15,12:20]=0
        floor=_shape_from_mask(mask,'floor',1);person=_shape_from_mask(np.pad(np.ones((10,10),np.uint8),((25,5),(25,5))),'person',2)
        hm=np.zeros_like(mask);hm[10:15,12:20]=1
        hole={'id':'a','shape_id':1,'bbox':[12,10,20,15],'mask':hm}
        calls=[]
        def chat(*args,**kwargs):
            calls.append(kwargs['response_schema'])
            return {'action':'fill_gap','semantic':'ground','reason':'floor'} if 'anyOf' in kwargs['response_schema'] else {'category':'robot_or_machine','object':'a gray machine'}
        fake=SimpleNamespace(review=lambda *a,**k:{'status':'continue','reason':'remove person','hole_assessment':[],
            'operations':[{'operation':'remove','shape_ids':[2],'reason':'false person'}]},chat=chat,
            _review_call=ContextVar('test'),MAX_PROTOCOL_RETRIES=0,ModelOutputError=workflow.ModelOutputError,
            _validate_review_result=workflow._validate_review_result,_emit_trace=lambda r:None)
        image=io.BytesIO();Image.new('RGB',(40,40),'gray').save(image,'PNG')
        with tempfile.TemporaryDirectory() as folder:
            install_review_batches(fake,cache_dir=folder)
            result=fake.review({'llm_model':'test','protected_initial_ids':[1,2]},image.getvalue(),image.getvalue(),
                {'description':'floor only'},[{'id':1},{'id':2}],['floor','person'],[],holes=[hole],candidate_geometry=[floor,person])
        self.assertEqual(len(calls),2);self.assertEqual(result['operations'],[])
        self.assertEqual(result['hole_assessment'][0]['semantic'],'non_ground')
        self.assertEqual(result['guarded_proposals'][0]['shape_ids'],[2])

if __name__=='__main__':unittest.main(verbosity=2)

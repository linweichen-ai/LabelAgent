import io
import unittest
from types import SimpleNamespace
from PIL import Image
from tools.bounded_review import install_review_batches


class BoundedReviewTests(unittest.TestCase):
    def setup_workflow(self, propose=False):
        calls=[]
        def review(*args, **kwargs):
            holes=kwargs.get('holes') or [];calls.append([h['id'] for h in holes])
            return {'status':'continue' if propose else 'done','reason':'visible evidence',
                    'hole_assessment':[{'hole_id':h['id'],'semantic':'ground' if propose else 'non_ground','reason':'equipment'} for h in holes],
                    'operations':[{'operation':'fill_gap','shape_ids':[1],'hole_id':h['id'],
                                   'polygon':[[0,0],[3,0],[0,3]],'reason':'floor'} for h in holes[:3]] if propose else []}
        def validate(result,*args):return result
        return SimpleNamespace(review=review,_validate_review_result=validate),calls
    def invoke(self, workflow, count):
        image=io.BytesIO();Image.new('RGB',(8,8)).save(image,'PNG')
        return workflow.review({},image.getvalue(),b'',{'description':'fixed task'},[{'id':1}],['floor'],[],
                               holes=[{'id':f'h{i}','shape_id':1} for i in range(count)])
    def test_all_holes_once_including_final_short_batch(self):
        workflow,calls=self.setup_workflow();install_review_batches(workflow,10,3)
        result=self.invoke(workflow,23)
        self.assertEqual(list(map(len,calls)),[10,10,3]);self.assertEqual(len(result['hole_assessment']),23)
        self.assertEqual([h for row in calls for h in row],[f'h{i}' for i in range(23)])
        self.assertEqual(result['status'],'done')
    def test_same_three_edit_cap_for_whole_agent_round(self):
        workflow,calls=self.setup_workflow(True);install_review_batches(workflow,10,3)
        result=self.invoke(workflow,31)
        self.assertEqual(len(result['operations']),3);self.assertEqual(len(result['hole_assessment']),31)
        self.assertEqual(result['status'],'continue')
    def test_small_or_empty_review_keeps_native_path(self):
        workflow,calls=self.setup_workflow();install_review_batches(workflow,10,3)
        result=self.invoke(workflow,0);self.assertEqual(calls,[[]]);self.assertEqual(result['operations'],[])


if __name__=='__main__':unittest.main(verbosity=2)

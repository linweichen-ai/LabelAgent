import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from source.agent import workflow
from tools.run_focused_class_recovery import current_view_hashes,matching_operation,focused_review_policy


class FocusedClassRecoveryTests(unittest.TestCase):
    def test_one_changed_view_or_uncertain_proposal_never_changes_a_class(self):
        shape={'id':7,'label':'pallet'};hashes=('a','b','c')
        self.assertIsNone(matching_operation(('a','b','changed'),{hashes:'person'},shape,['pallet','person']))
        self.assertIsNone(matching_operation(hashes,{hashes:'uncertain'},shape,['pallet','person']))
        self.assertIsNone(matching_operation(hashes,{hashes:'pallet'},shape,['pallet','person']))
        self.assertEqual(matching_operation(hashes,{hashes:'person'},shape,['pallet','person'])['label'],'person')

    def test_class_flip_keeps_identical_image_cache_binding(self):
        stream=io.BytesIO();Image.new('RGB',(40,30),(80,110,50)).save(stream,'PNG')
        shape={'id':7,'label':'person','points':[[4,4],[25,4],[25,24],[4,24]]}
        before=current_view_hashes(stream.getvalue(),shape)
        shape['label']='pallet';shape['id']=19
        self.assertEqual(current_view_hashes(stream.getvalue(),shape),before)

    def test_cache_miss_delegates_to_real_review_and_scope_restores_after_error(self):
        stream=io.BytesIO();Image.new('RGB',(40,30),(80,110,50)).save(stream,'PNG')
        agent=SimpleNamespace(image=stream.getvalue(),width=40,height=30,labels=['pallet','person'],
            shapes=[{'id':7,'label':'person','points':[[4,4],[25,4],[25,24],[4,24]]}])
        events=[]
        def fail(*args,**kwargs):raise RuntimeError('original review called')
        with patch.object(workflow,'review',fail):
            with self.assertRaisesRegex(RuntimeError,'original review called'):
                with focused_review_policy(agent,{},events.append):workflow.review()
            self.assertIs(workflow.review,fail)
        self.assertFalse(events[0]['cache_match'])
        self.assertFalse(events[0]['operation_proposed'])


if __name__=='__main__':unittest.main()

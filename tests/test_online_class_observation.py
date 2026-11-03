import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from PIL import Image
from source.agent import workflow
from tools import online_class_observation as policy


def agent_fixture():
    stream=io.BytesIO();Image.new('RGB',(40,30),(20,70,120)).save(stream,'PNG')
    return stream.getvalue(),SimpleNamespace(shapes=[{'id':7,'label':'pallet','points':[[2,2],[30,2],[30,25],[2,25]]}],width=40,height=30,labels=['pallet','person'])


class OnlineObservationTests(unittest.TestCase):
    def test_observation_does_not_choose_operations_or_replace_model_decision(self):
        image,agent=agent_fixture();events=[]
        observation={'candidate_id':7,'single_observed_class':'person','context_observed_class':'person','two_view_consensus':'person'}
        returned={'status':'done','operations':[]}
        original=Mock(return_value=returned);schema={'properties':{'operations':{}}}
        with patch.object(policy,'classify_online',return_value=observation.copy()) as classify,patch.object(workflow,'chat',original):
            with policy.scoped_online_observation('online_observation',agent,{},lambda event:None,events.append):
                result=workflow.chat({},'base',image,extra_images=(b'overlay',),response_schema=schema)
        self.assertIs(result,returned)
        classify.assert_called_once()
        args,kwargs=original.call_args
        self.assertTrue(args[1].startswith('base'))
        self.assertIn('"current_label_from_candidate":"pallet"',args[1])
        self.assertNotIn('"operation":"reclassify"',args[1])
        self.assertEqual(kwargs['extra_images'],(b'overlay',))
        self.assertIs(kwargs['response_schema'],schema)

    def test_each_review_observes_again_without_a_classification_cache(self):
        image,agent=agent_fixture();original=Mock(return_value={})
        with patch.object(policy,'classify_online',return_value={'two_view_consensus':'uncertain'}) as classify,patch.object(workflow,'chat',original):
            with policy.scoped_online_observation('online_observation',agent,{},lambda event:None,lambda event:None):
                workflow.chat({},'base',image);workflow.chat({},'base',image)
        self.assertEqual(classify.call_count,2)
        self.assertEqual(original.call_count,2)

    def test_baseline_and_failed_observation_restore_original_chat(self):
        image,agent=agent_fixture();original=Mock(return_value={})
        with patch.object(policy,'classify_online',side_effect=ValueError('invalid image response')) as classify,patch.object(workflow,'chat',original):
            with policy.scoped_online_observation('baseline',agent,{},lambda event:None,lambda event:None):
                self.assertIs(workflow.chat,original)
            classify.assert_not_called()
            with self.assertRaisesRegex(ValueError,'invalid image response'):
                with policy.scoped_online_observation('online_observation',agent,{},lambda event:None,lambda event:None):workflow.chat({},'base',image)
            self.assertIs(workflow.chat,original)
            original.assert_not_called()

    def test_conflict_gets_one_extra_model_decision_and_only_model_selects_the_operation(self):
        image,agent=agent_fixture();events=[]
        observation={'two_view_consensus':'person'}
        final={'status':'continue','operations':[{'operation':'reclassify','shape_ids':[7],'label':'person','reason':'Photo evidence.'}]}
        original=Mock(side_effect=[{'status':'done','operations':[],'reason':'Wrong class.'},final])
        with patch.object(policy,'classify_online',return_value=observation),patch.object(workflow,'chat',original):
            with policy.scoped_online_observation('online_reconsideration',agent,{},lambda event:None,events.append):
                result=workflow.chat({},'base',image)
        self.assertIs(result,final)
        self.assertEqual(original.call_count,2)
        self.assertIn('Consistency reconsideration',original.call_args.args[1])
        self.assertTrue(events[0]['consistency_reconsideration_called'])

    def test_reconsideration_can_keep_done_without_any_forced_edit(self):
        image,agent=agent_fixture()
        refused={'status':'done','operations':[],'reason':'Insufficient evidence.'}
        original=Mock(side_effect=[{'status':'done','operations':[]},refused])
        with patch.object(policy,'classify_online',return_value={'two_view_consensus':'person'}),patch.object(workflow,'chat',original):
            with policy.scoped_online_observation('online_reconsideration',agent,{},lambda event:None,lambda event:None):
                result=workflow.chat({},'base',image)
        self.assertIs(result,refused)
        self.assertEqual(result['operations'],[])

    def test_same_or_uncertain_class_does_not_trigger_extra_decisions(self):
        image,agent=agent_fixture()
        for category in ('pallet','uncertain'):
            original=Mock(return_value={'status':'done','operations':[]})
            with patch.object(policy,'classify_online',return_value={'two_view_consensus':category}),patch.object(workflow,'chat',original):
                with policy.scoped_online_observation('online_reconsideration',agent,{},lambda event:None,lambda event:None):workflow.chat({},'base',image)
            self.assertEqual(original.call_count,1)


if __name__=='__main__':unittest.main()

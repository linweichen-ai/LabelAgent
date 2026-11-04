import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from source.agent import server

class TopicSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.state={'video':SimpleNamespace(topics={'camera0':{},'camera1':{}}),
            'topic':'camera0','labels':['pallet'],'truth':{'0':[{'label':'pallet','points':[[1,1],[5,1],[5,5]]}]},
            'candidates':{'76':[{'id':1,'label':'person'}]},'agent_runs':{'76':{'rounds':1,'stop_reason':'done'}},
            'truths_by_topic':{'camera1':{'1':[]}},'candidates_by_topic':{},'agent_runs_by_topic':{},
            'reference_frames':{'camera0':[0]}}
        self.patcher=patch.multiple(server,state=self.state,WORK=Path(self.temp.name))
        self.patcher.start()
        self.client=server.app.test_client()
    def tearDown(self):
        self.patcher.stop();self.temp.cleanup()
    def test_current_camera_is_read_only_even_if_storage_unavailable(self):
        before=copy.deepcopy({k:v for k,v in self.state.items() if k!='video'})
        with patch.object(server,'save',side_effect=RuntimeError('disk unavailable')) as save:
            reply=self.client.post('/api/topic',json={'topic':'camera0'})
        self.assertEqual(reply.status_code,200)
        save.assert_not_called()
        self.assertEqual(reply.json['truth'],self.state['truth'])
        self.assertEqual(reply.json['candidates'],self.state['candidates'])
        self.assertEqual(reply.json['agent_runs'],self.state['agent_runs'])
        self.assertEqual(before,{k:v for k,v in self.state.items() if k!='video'})
        self.assertFalse(server.state_path().exists())
    def test_other_camera_persists_and_roundtrip_restores_annotations(self):
        original=copy.deepcopy({k:self.state[k] for k in ('truth','candidates','agent_runs')})
        with patch.object(server,'save',wraps=server.save) as save:
            self.assertEqual(self.client.post('/api/topic',json={'topic':'camera1'}).status_code,200)
            self.assertTrue(server.state_path().exists())
            self.assertEqual(self.state['truth'],{'1':[]})
            self.assertEqual(self.client.post('/api/topic',json={'topic':'camera0'}).status_code,200)
            self.assertEqual(save.call_count,2)
        for key,value in original.items():self.assertEqual(self.state[key],value)
        self.assertEqual(self.state['reference_frames'],{'camera0':[0]})
    def test_invalid_camera_cannot_mutate_or_save(self):
        with patch.object(server,'save') as save:
            reply=self.client.post('/api/topic',json={'topic':'cloud'})
        self.assertEqual(reply.status_code,400)
        self.assertEqual(self.state['topic'],'camera0');save.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)

import base64,importlib.util,sys,types,unittest
from pathlib import Path
from unittest.mock import patch


class PriorAPITests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[2];p=root/'task_workspaces/labelagent_soft_prior_20261108/deployment/server.new.py'
        if not p.exists():p=Path(__file__).parent/'fixtures/sam_server_prior.py'
        package=types.ModuleType('test_prior_service');package.__path__=[];self.calls=[];modules={'test_prior_service':package}
        def add(name,function):
            m=types.ModuleType('test_prior_service.'+name);setattr(m,function,lambda *a,**kw:self.reply(name,a,kw));modules[m.__name__]=m;return m
        point=add('point_segment','segment');point.model_loaded=lambda:False;add('raw_diagnostic','segment_with_capture');add('boxed_capture','segment_with_box_capture');add('soft_prior_capture','segment_with_prior_capture');patcher=patch.dict(sys.modules,modules);patcher.start();self.addCleanup(patcher.stop);spec=importlib.util.spec_from_file_location('test_prior_service.server',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);self.client=m.app.test_client();self.body={'image':base64.b64encode(b'image').decode(),'points':[{'x':5,'y':6,'label':1}]}
    def reply(self,name,args,kwargs):
        self.calls.append((name,args,kwargs));return {'route':name} if name=='point_segment' else ({'route':name},200)
    def test_default_and_raw_paths_unchanged(self):
        self.assertEqual(self.client.post('/segment_points',json=self.body).json,{'route':'point_segment'});self.assertEqual(self.client.post('/segment_points',json=dict(self.body,include_raw_capture=True)).json,{'route':'raw_diagnostic'})
    def test_existing_box_diagnostic_still_explicit(self):
        self.assertEqual(self.client.post('/segment_points',json=dict(self.body,include_raw_capture=True,diagnostic_box=[1,2,7,9])).json,{'route':'boxed_capture'})
    def test_paired_hard_soft_route_and_exact_payload(self):
        for mode in ['hard','soft']:
            response=self.client.post('/segment_points',json=dict(self.body,include_raw_capture=True,mask=base64.b64encode(b'binary').decode(),diagnostic_prior_logits=base64.b64encode(b'float32-npy').decode(),diagnostic_prior_mode=mode));self.assertEqual(response.status_code,200);self.assertEqual(self.calls[-1][1],(b'image',self.body['points'],b'binary',b'float32-npy',mode))
    def test_incomplete_or_mixed_route_rejected(self):
        full=dict(self.body,include_raw_capture=True,mask=base64.b64encode(b'binary').decode(),diagnostic_prior_logits=base64.b64encode(b'npy').decode(),diagnostic_prior_mode='soft')
        for change in [{'include_raw_capture':False},{'mask':None},{'diagnostic_box':[1,2,7,9]},{'diagnostic_prior_mode':'other'},{'diagnostic_prior_logits':None}]:
            response=self.client.post('/segment_points',json=dict(full,**change));self.assertEqual(response.status_code,400)
        self.assertEqual(self.calls,[])
    def test_bad_base64_fails_before_predictor(self):
        response=self.client.post('/segment_points',json=dict(self.body,include_raw_capture=True,mask=base64.b64encode(b'prior').decode(),diagnostic_prior_logits='###',diagnostic_prior_mode='soft'));self.assertEqual(response.status_code,400);self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()

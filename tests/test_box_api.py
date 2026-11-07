"""Real Flask routing with stub predictors; no GPU or network."""
import base64,importlib.util,sys,types,unittest
from pathlib import Path
from unittest.mock import patch


class BoxAPITests(unittest.TestCase):
    def setUp(self):
        folder=Path(__file__).resolve().parents[2]/'task_workspaces/labelagent_box_layers_20261107/deployment'
        server_file=folder/'server.new.py'
        if not server_file.exists():server_file=Path(__file__).resolve().parent/'fixtures/sam_server_box.py'
        package=types.ModuleType('test_box_service');package.__path__=[]
        self.default=[];self.boxed=[];self.raw=[]
        point=types.ModuleType('test_box_service.point_segment');point.model_loaded=lambda:False
        def default(*args,**kwargs):self.default.append((args,kwargs));return {'default_unchanged':True}
        point.segment=default
        boxed=types.ModuleType('test_box_service.boxed_capture')
        def box(*args):self.boxed.append(args);return {'experimental_box_prompt':True},200
        boxed.segment_with_box_capture=box
        raw=types.ModuleType('test_box_service.raw_diagnostic')
        def capture(*args):self.raw.append(args);return {'raw_only':True},200
        raw.segment_with_capture=capture
        self.patch=patch.dict(sys.modules,{'test_box_service':package,'test_box_service.point_segment':point,'test_box_service.boxed_capture':boxed,'test_box_service.raw_diagnostic':raw});self.patch.start();self.addCleanup(self.patch.stop)
        spec=importlib.util.spec_from_file_location('test_box_service.server',server_file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);self.client=module.app.test_client();self.body={'image':base64.b64encode(b'fake-photo').decode(),'points':[{'x':4,'y':5,'label':1}]}

    def test_default_path_unchanged(self):
        response=self.client.post('/segment_points',json=self.body);self.assertEqual(response.status_code,200);self.assertEqual(response.json,{'default_unchanged':True});self.assertEqual(len(self.default),1);self.assertEqual(self.boxed,[])

    def test_raw_only_path_unchanged(self):
        response=self.client.post('/segment_points',json=dict(self.body,include_raw_capture=True));self.assertEqual(response.json,{'raw_only':True});self.assertEqual(len(self.raw),1);self.assertEqual(self.boxed,[])

    def test_box_requires_explicit_capture_and_no_prior(self):
        for extra in [{'diagnostic_box':[1,2,6,8]},{'diagnostic_box':[1,2,6,8],'include_raw_capture':True,'mask':base64.b64encode(b'prior').decode()}]:
            response=self.client.post('/segment_points',json=dict(self.body,**extra));self.assertEqual(response.status_code,400)
        self.assertEqual(self.boxed,[])

    def test_box_only_explicit_route(self):
        response=self.client.post('/segment_points',json=dict(self.body,diagnostic_box=[1,2,6,8],include_raw_capture=True));self.assertEqual(response.status_code,200);self.assertEqual(self.boxed[0],(b'fake-photo',self.body['points'],[1,2,6,8]));self.assertEqual(self.default,[])


if __name__=='__main__':unittest.main()

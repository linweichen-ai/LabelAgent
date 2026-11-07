import base64,io,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from source.sam3_service import server,point_segment

class RawAPI(unittest.TestCase):
 def setUp(self):
  b=io.BytesIO();Image.new('RGB',(64,64)).save(b,'PNG')
  self.body={'image':base64.b64encode(b.getvalue()).decode(),'points':[{'x':10,'y':10,'label':1}]}
  self.client=server.app.test_client()
 def test_default_unchanged(self):
  with patch.object(server,'segment',return_value={'parts':[], 'score':.7}) as call:
   r=self.client.post('/segment_points',json=self.body)
  self.assertEqual(r.json,{'parts':[],'score':.7});call.assert_called_once()
 def test_invalid_flag_never_infers(self):
  with patch.object(server,'segment') as call:
   r=self.client.post('/segment_points',json={**self.body,'include_raw_capture':'true'})
  self.assertEqual(r.status_code,400);call.assert_not_called()
 def test_full_internal_array_and_same_cached_model(self):
  masks=np.zeros((1,64,64),np.uint8);masks[0,5:20,5:20]=1;masks[0,35:55,35:55]=1
  class Model:
   def predict_inst(self,*args,**kw):return masks,np.array([.9]),None
  class Processor:
   def set_image(self,image):return {}
  with patch.object(point_segment,'_predictor',return_value=(Model(),Processor())) as factory,patch('torch.cuda.is_available',return_value=False):
   r=self.client.post('/segment_points',json={**self.body,'include_raw_capture':True})
  self.assertEqual(r.status_code,200);factory.assert_called_once()
  capture=r.json['raw_capture'];a=np.load(io.BytesIO(base64.b64decode(capture['arrays_npz_base64'])),allow_pickle=False)
  np.testing.assert_array_equal(a['0_masks'],masks)
  self.assertEqual(capture['metadata']['selected_index'],0)
  self.assertTrue(capture['metadata']['raw_capture_complete'])
  self.assertEqual(len(r.json['parts']),1)
 def test_failure_retains_internal_arrays(self):
  class Model:
   def predict_inst(self,*args,**kw):return np.zeros((1,64,64),np.uint8),np.array([.9]),None
  class Processor:
   def set_image(self,image):return {}
  with patch.object(point_segment,'_predictor',return_value=(Model(),Processor())),patch('torch.cuda.is_available',return_value=False):
   r=self.client.post('/segment_points',json={**self.body,'include_raw_capture':True})
  self.assertEqual(r.status_code,400)
  self.assertTrue(r.json['raw_capture']['metadata']['raw_capture_complete'])
  self.assertEqual(r.json['raw_capture']['metadata']['status'],'segment_raised')
if __name__=='__main__':unittest.main()

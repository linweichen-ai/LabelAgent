"""Request-level image budgets are local, single-image, restored and attested."""
import base64
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from PIL import Image

spec=importlib.util.spec_from_file_location('image_budget_server',Path(__file__).resolve().parents[1]/'tools/local_vlm_server.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def messages(count=1):
    buffer=io.BytesIO();Image.new('RGB',(32,32)).save(buffer,'PNG')
    return [{'role':'user','content':[{'type':'text','text':'Return JSON'}]+[
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}} for _ in range(count)]}]

class Backend:
    model_id='explicit-test-double';backend_name='contract_test_double';model_loaded=True
    quantization='nf4';image_token_budget=256;process_allocator_fraction=None
    model_config_sha256='explicit-test-model-config'
    allow_single_image_budget_override=True
    def __init__(self):self.calls=[]
    def generate(self,messages,images,max_tokens,response_schema=None,image_token_budget=None):
        self.calls.append(image_token_budget)
        self.last_inference_stats={'effective_image_token_budget':image_token_budget or 256,
            'actual_image_token_counts':[986],'image_grid_thw':[[1,58,68]]}
        return '{"category":"uncertain"}',1200,8

class VLMImageBudgetTest(unittest.TestCase):
    def test_API_reports_request_bound_actual_grid_and_preserves_default(self):
        backend=Backend();client=m.create_app(backend).test_client()
        result=client.post('/v1/chat/completions',json={'model':backend.model_id,'messages':messages(),'image_token_budget':1024})
        self.assertEqual(result.status_code,200)
        self.assertEqual(backend.calls,[1024])
        self.assertEqual(result.json['local_vlm']['effective_image_token_budget'],1024)
        self.assertEqual(result.json['local_vlm']['actual_image_token_counts'],[986])
        self.assertEqual(result.json['local_vlm']['quantization'],'nf4')
        self.assertEqual(result.json['local_vlm']['model_config_sha256'],backend.model_config_sha256)
        self.assertEqual(client.get('/health').json['image_token_budget'],256)
        client.post('/v1/chat/completions',json={'model':backend.model_id,'messages':messages()})
        self.assertEqual(backend.calls,[1024,None])

    def test_invalid_budget_or_multiple_images_rejected_before_generation(self):
        backend=Backend();client=m.create_app(backend).test_client()
        for budget in [None,True,256.,'256',63,1025]:
            result=client.post('/v1/chat/completions',json={'model':backend.model_id,'messages':messages(),'image_token_budget':budget})
            self.assertEqual(result.status_code,400)
        result=client.post('/v1/chat/completions',json={'model':backend.model_id,'messages':messages(2),'image_token_budget':1024})
        self.assertEqual(result.status_code,400)
        self.assertEqual(backend.calls,[])

    def test_nonlocal_and_shared_probe_modes_cannot_bypass_budget(self):
        backend=Backend();backend.allow_single_image_budget_override=False
        with self.assertRaises(ValueError):m.checked_single_image_budget(1024,backend,1)
        backend.allow_single_image_budget_override=True;backend.process_allocator_fraction=.35
        with self.assertRaises(ValueError):m.checked_single_image_budget(1024,backend,1)
        with self.assertRaises(ValueError):m.TransformersBackend('unused','unused',quantization='nf4',memory_fraction=.35,allow_single_image_budget_override=True)
        with self.assertRaises(ValueError):m.TransformersBackend('unused','unused',quantization='none',allow_single_image_budget_override=True)

    def test_temporary_size_restored_on_success_exception_and_nested_use(self):
        original={'longest_edge':256*1024,'shortest_edge':64*1024}
        processor=SimpleNamespace(image_processor=SimpleNamespace(size=original))
        with m.temporary_image_budget(processor,1024):
            self.assertEqual(processor.image_processor.size['longest_edge'],1024*1024)
            with m.temporary_image_budget(processor,256):self.assertEqual(processor.image_processor.size['longest_edge'],256*1024)
            self.assertEqual(processor.image_processor.size['longest_edge'],1024*1024)
        self.assertIs(processor.image_processor.size,original)
        with self.assertRaises(RuntimeError):
            with m.temporary_image_budget(processor,1024):raise RuntimeError('processor failed')
        self.assertIs(processor.image_processor.size,original)

    def test_actual_counts_come_from_real_grid_not_declared_ceiling(self):
        processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
        inputs={'image_grid_thw':SimpleNamespace(tolist=lambda:[[1,58,68],[1,16,16]])}
        grid,counts=m.actual_image_encoding(inputs,processor,2)
        self.assertEqual(counts,[986,64])
        self.assertEqual(grid[0],[1,58,68])
        with self.assertRaises(RuntimeError):m.actual_image_encoding(inputs,processor,1)
        with self.assertRaises(RuntimeError):m.actual_image_encoding({},processor,1)

    def test_disabled_override_leaves_legacy_API_usable(self):
        backend=Backend();backend.allow_single_image_budget_override=False
        client=m.create_app(backend).test_client()
        self.assertFalse(client.get('/health').json['supports_single_image_budget_override'])
        self.assertEqual(client.post('/v1/chat/completions',json={'model':backend.model_id,'messages':messages()}).status_code,200)
        self.assertEqual(backend.calls,[None])

if __name__=='__main__':unittest.main()

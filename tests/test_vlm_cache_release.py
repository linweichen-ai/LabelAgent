"""A cache trim must preserve live tensors and run before releasing ownership."""
import base64
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from PIL import Image
from tools.local_vlm_server import trim_idle_cuda_cache,create_app


class CacheReleaseTests(unittest.TestCase):
    def test_disabled_and_small_unused_cache_are_not_touched(self):
        self.assertFalse(trim_idle_cuda_cache(None,False)['performed'])
        cuda=SimpleNamespace(memory_allocated=lambda:1024,memory_reserved=lambda:2048,empty_cache=Mock())
        result=trim_idle_cuda_cache(SimpleNamespace(cuda=cuda),True)
        self.assertFalse(result['performed'])
        cuda.empty_cache.assert_not_called()

    def test_large_unused_cache_is_released_without_changing_live_allocation(self):
        state={'reserved':2048*1024**2}
        cuda=SimpleNamespace(memory_allocated=lambda:1024*1024**2,memory_reserved=lambda:state['reserved'])
        def release():state['reserved']=1024*1024**2
        cuda.empty_cache=release
        result=trim_idle_cuda_cache(SimpleNamespace(cuda=cuda),True)
        self.assertTrue(result['performed'])
        self.assertEqual(result['released_reserved_bytes'],1024*1024**2)
        self.assertEqual(result['allocated_bytes_before'],result['allocated_bytes_after'])

    def test_failed_trim_keeps_successful_output_and_releases_inference_lock(self):
        class Backend:
            backend_name='explicit_test_double';model_id='test';idle_cache_release=True
            def __init__(self):
                self.torch=SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda:1024,
                    memory_reserved=lambda:1024**3,empty_cache=Mock(side_effect=RuntimeError('trim failed'))))
            def generate(self,messages,images,max_tokens,**kwargs):return '{}',1,1
        backend=Backend();client=create_app(backend).test_client()
        data=io.BytesIO();Image.new('RGB',(8,8)).save(data,'PNG')
        body={'model':'test','messages':[{'role':'user','content':[{'type':'text','text':'test'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()}}]}]}
        for _ in range(2):
            response=client.post('/v1/chat/completions',json=body)
            self.assertEqual(response.status_code,200,response.json)
            self.assertEqual(response.json['choices'][0]['message']['content'],'{}')
            self.assertEqual(response.json['local_vlm']['cache_release']['error_type'],'RuntimeError')
        self.assertEqual(backend.torch.cuda.empty_cache.call_count,2)


if __name__=='__main__':unittest.main()

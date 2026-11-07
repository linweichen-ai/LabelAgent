import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from tools.centered_target_views import centered_target_views
from tools.context_target_views import context_target_views
from tools.evaluate_context_targets import checked_cached_response

class ContextChecks(unittest.TestCase):
    def test_three_span_input_exactly_preserves_existing_input(self):
        source=Image.new('RGB',(200,180),(80,90,100));mask=np.zeros((180,200),bool);mask[80:90,90:120]=True;box=(50,40,160,140)
        old,_=centered_target_views(source,mask,box);new,meta=context_target_views(source,mask,box)
        self.assertEqual(old['linked'].tobytes(),new['overview'].tobytes())
        self.assertTrue(all(m['retained_target_pixels']==mask.sum() for m in meta['views'].values()))

    def test_outside_region_mutation_cannot_change_new_context(self):
        pixels=np.random.default_rng(4).integers(0,256,(180,200,3),np.uint8);other=np.zeros_like(pixels);other[40:140,50:160]=pixels[40:140,50:160]
        mask=np.zeros((180,200),bool);mask[80:90,90:120]=True;box=(50,40,160,140)
        a,_=context_target_views(Image.fromarray(pixels),mask,box);b,_=context_target_views(Image.fromarray(other),mask,box)
        self.assertEqual({k:v.tobytes() for k,v in a.items()},{k:v.tobytes() for k,v in b.items()})

    def test_cached_response_from_another_payload_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.json';p.write_text(json.dumps({'full_payload_sha256':'old'}));ref={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            with self.assertRaises(ValueError):checked_cached_response(ref,'different',{})

    def test_changed_cached_response_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.json';p.write_text('{}');ref={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):checked_cached_response(ref,'x',{})

if __name__=='__main__':unittest.main()

import io
import json
import unittest
import numpy as np
from PIL import Image

from tools.evaluate_instance_focus import render_instance_views,largest_mask_candidate,consensus_category,focus_checks
from tools.evaluate_vlm_grounding import build_payload


class InstanceFocusTests(unittest.TestCase):
    def test_crop_preserves_foreground_holes_and_source_and_never_uses_label_text(self):
        picture=Image.new('RGB',(40,30),(10,60,120));stream=io.BytesIO();picture.save(stream,'PNG');raw=stream.getvalue()
        shape={'id':1,'label':'person','points':[[5,4],[25,4],[25,24],[5,24]],'holes':[[[12,10],[18,10],[18,16],[12,16]]]}
        views,meta=render_instance_views(raw,shape,padding=0)
        self.assertEqual(views[0].size,(21,21))
        self.assertEqual(views[0].getpixel((0,0)),(10,60,120))
        self.assertEqual(views[0].getpixel((10,9)),(127,127,127))
        self.assertEqual(views[1].getpixel((10,9)),(10,60,120))
        changed={**shape,'label':'WRONG_CLASS','id':999}
        other,_=render_instance_views(raw,changed,padding=0)
        for first,second in zip(views,other):self.assertEqual(first.tobytes(),second.tobytes())
        self.assertEqual(stream.getvalue(),raw)
        self.assertFalse(meta['label_text_rendered'])

    def test_largest_mask_selection_ignores_classes_and_model_scores(self):
        small={'id':1,'label':'person','score':1,'points':[[1,1],[3,1],[3,3],[1,3]]}
        large={'id':2,'label':'pallet','score':.01,'parts':[{'points':[[5,5],[15,5],[15,15],[5,15]],'holes':[]}]}
        self.assertIs(largest_mask_candidate([small,large],20,20),large)
        large['label']='person';small['label']='pallet'
        self.assertIs(largest_mask_candidate([small,large],20,20),large)

    def test_blank_called_person_is_failure_even_when_no_pallet_is_accepted(self):
        fixture={'minimum_pallet_consensus':3,'cases':[{'id':'blank','role':'blank_control'}]}
        row={'id':'blank','consensus':'person','views':{kind:{'parsed':{'category':'person'}} for kind in ('single','context')}}
        self.assertFalse(focus_checks(fixture,[row],2)['blank_views_always_uncertain'])
        self.assertEqual(consensus_category('person','uncertain'),'uncertain')
        self.assertEqual(consensus_category('person','person'),'person')

    def test_partial_or_all_uncertain_cannot_pass_usefulness_checks(self):
        cases=[{'id':str(i),'role':role} for i,role in enumerate(['person_fixture']*2+['pallet_fixture']*5+['blank_control']*2+['suspected_non_pallet'])]
        fixture={'cases':cases,'minimum_pallet_consensus':3}
        rows=[{'id':case['id'],'consensus':'uncertain','views':{kind:{'parsed':{'category':'uncertain'}} for kind in ('single','context')}} for case in cases]
        checks=focus_checks(fixture,rows,20)
        self.assertFalse(checks['all_person_fixtures_agree_person'])
        self.assertFalse(checks['minimum_pallet_consensus'])
        self.assertFalse(any(focus_checks(fixture,rows[:1],2).values()))

    def test_fixture_role_and_current_label_do_not_enter_model_request(self):
        fixture={'categories':['pallet','person','uncertain'],'temperature':0,'max_tokens':192,
            'prompts':{'single':'Describe the foreground.'}}
        case={'id':'SECRET_CONTROL','role':'person_fixture','current_label':'WRONG_LABEL',
            'views':{'single':[{'path':'image','sha256':__import__('hashlib').sha256(b'bytes').hexdigest()}]}}
        body,_=build_payload(fixture,{'image':(b'bytes','image/jpeg')},case,'single','model')
        text=json.dumps(body)
        for secret in ('SECRET_CONTROL','person_fixture','WRONG_LABEL'):self.assertNotIn(secret,text)


if __name__=='__main__':unittest.main()

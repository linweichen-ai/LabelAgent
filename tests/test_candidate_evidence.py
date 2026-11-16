import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from PIL import Image
from source.agent import workflow
from tools.candidate_evidence import candidate_evidence_images,scoped_evidence_policy


def fixture():
    stream=io.BytesIO();Image.new('RGB',(100,80),(20,90,140)).save(stream,'PNG')
    shape={'id':7,'label':'person','points':[[4,4],[85,4],[85,75],[4,75]]}
    return stream.getvalue(),shape


class CandidateEvidenceTests(unittest.TestCase):
    def test_large_readable_ID_preserves_label_blind_rendering(self):
        image,shape=fixture()
        before,metadata=candidate_evidence_images(image,[shape])
        shape['label']='pallet';after,_=candidate_evidence_images(image,[shape])
        self.assertEqual(before,after)
        self.assertEqual(metadata['header_font_pixels'],48)
        self.assertGreaterEqual(Image.open(io.BytesIO(before[0])).width,640)
        shape['id']=19;changed,_=candidate_evidence_images(image,[shape])
        self.assertNotEqual(before,changed)
        self.assertFalse(metadata['current_class_text_rendered'])

    def test_real_chat_is_called_with_original_prefix_images_and_same_schema(self):
        image,shape=fixture();agent=SimpleNamespace(shapes=[shape]);events=[]
        schema={'required':['status']};extra=(b'original-overlay',b'reference',b'reference-outline')
        original=Mock(return_value={'status':'done'})
        with patch.object(workflow,'chat',original):
            with scoped_evidence_policy('candidate_crops',agent,events.append):
                value=workflow.chat({},'original prompt',image,extra_images=extra,response_schema=schema)
            self.assertIs(workflow.chat,original)
        self.assertEqual(value,{'status':'done'})
        args,kwargs=original.call_args
        self.assertTrue(args[1].startswith('original prompt'))
        self.assertEqual(args[2],image)
        self.assertEqual(kwargs['extra_images'][:3],extra)
        self.assertEqual(len(kwargs['extra_images']),6)
        self.assertIs(kwargs['response_schema'],schema)
        self.assertTrue(events[0]['appended'])

    def test_insufficient_image_slots_delegate_without_changing_the_request(self):
        image,shape=fixture();agent=SimpleNamespace(shapes=[shape]);events=[]
        original=Mock(return_value={});extra=(b'x',)*6
        with patch.object(workflow,'chat',original):
            with scoped_evidence_policy('candidate_crops',agent,events.append):
                workflow.chat({},'base',image,extra_images=extra)
        original.assert_called_once_with({},'base',image,extra_images=extra,response_schema=None)
        self.assertFalse(events[0]['appended'])

    def test_default_and_exception_paths_restore_production_chat(self):
        image,shape=fixture();agent=SimpleNamespace(shapes=[shape])
        original=Mock(side_effect=RuntimeError('real chat failed'))
        with patch.object(workflow,'chat',original):
            with scoped_evidence_policy('baseline',agent,lambda event:None):self.assertIs(workflow.chat,original)
            with self.assertRaisesRegex(RuntimeError,'real chat failed'):
                with scoped_evidence_policy('candidate_crops',agent,lambda event:None):workflow.chat({},'base',image)
            self.assertIs(workflow.chat,original)


if __name__=='__main__':unittest.main()

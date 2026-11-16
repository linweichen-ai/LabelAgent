"""Tests for consequential review/split/evaluation boundaries; no GPU required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from data_tools import check_records,export_reviewed,sha256
from evaluate_qwen import decode_answer,summarize


class ReviewBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'a.jpg').write_bytes(b'test-only-image');(self.root/'b.jpg').write_bytes(b'crop')
        self.row={'sample_id':'a','capture_id':'c','group_id':'c:block0','source_image_sha256':'sourcehash',
            'source_kind':'model_candidate_unconfirmed','images':['a.jpg','b.jpg'],
            'image_sha256':{n:sha256(self.root/n) for n in ['a.jpg','b.jpg']},'proposed_label':'pallet',
            'review_status':'human_reviewed','reviewer':'test-fixture-only','reviewed_at':'2026-11-04T00:00:00Z','decision':'keep','correct_label':'pallet'}
    def tearDown(self):self.tmp.cleanup()
    def test_pending_cannot_train(self):
        self.row['review_status']='pending'
        with self.assertRaisesRegex(ValueError,'Pending'):check_records([self.row],self.root,True)
    def test_reviewer_required(self):
        self.row['reviewer']=''
        with self.assertRaisesRegex(ValueError,'receipt'):check_records([self.row],self.root,True)
    def test_noop_relabel_rejected(self):
        self.row['decision']='relabel'
        with self.assertRaisesRegex(ValueError,'no-op'):check_records([self.row],self.root,True)
    def test_tampered_image_rejected(self):
        (self.root/'a.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'digest'):check_records([self.row],self.root)
    def test_cross_split_same_source_rejected(self):
        b=copy.deepcopy(self.row);b.update(sample_id='b',group_id='c:block1')
        p=self.root/'reviewed.jsonl';p.write_text('\n'.join(json.dumps(x) for x in [self.row,b]))
        with self.assertRaisesRegex(ValueError,'leakage'):export_reviewed(p,self.root,self.root/'out','c:block1',1,1)
    def test_valid_group_export_and_answer(self):
        b=copy.deepcopy(self.row);b.update(sample_id='b',group_id='c:block1',source_image_sha256='different')
        # Distinct source hashes do not alone prove distinct pixels, so export must reject reused derived images too.
        (self.root/'c.jpg').write_bytes(b'another scene');(self.root/'d.jpg').write_bytes(b'another crop')
        b['images']=['c.jpg','d.jpg'];b['image_sha256']={n:sha256(self.root/n) for n in b['images']}
        p=self.root/'reviewed.jsonl';p.write_text('\n'.join(json.dumps(x) for x in [self.row,b]))
        result=export_reviewed(p,self.root,self.root/'out','c:block1',1,1)
        self.assertEqual((result['train'],result['dev']),(1,1))
        data=json.loads((self.root/'out/train.json').read_text());self.assertEqual(json.loads(data[0]['conversations'][1]['value']),{'decision':'keep','label':'pallet'})
    def test_cross_split_derived_image_rejected(self):
        b=copy.deepcopy(self.row);b.update(sample_id='b',group_id='c:block1',source_image_sha256='different')
        p=self.root/'reviewed.jsonl';p.write_text('\n'.join(json.dumps(x) for x in [self.row,b]))
        with self.assertRaisesRegex(ValueError,'derived image'):export_reviewed(p,self.root,self.root/'out','c:block1',1,1)
    def test_duplicate_sample_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):check_records([self.row,self.row],self.root)
    def test_strict_json(self):
        self.assertIsNone(decode_answer('```json\n{"decision":"keep","label":"pallet"}\n```'))
        self.assertIsNone(decode_answer('{"decision":"uncertain","label":"pallet"}'))
    def test_abstention_not_counted_as_success(self):
        d=summarize([{'expected':{'decision':'keep','label':'pallet'},'parsed':{'decision':'uncertain','label':'unknown'},'correct':False,'proposed_label':'pallet'}])
        self.assertEqual(d['accuracy'],0);self.assertEqual(d['abstained_on_decidable'],1)


if __name__=='__main__':unittest.main()

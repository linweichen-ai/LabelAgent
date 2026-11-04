import io,json,sys,unittest
from pathlib import Path
from source.agent.keyed_holes import decode_keyed_assessments,reject_duplicate_json_keys
from source.agent.protocol_schema import review_schema
from source.agent.workflow import _validate_review_result

class KeyedReviewChecks(unittest.TestCase):
    holes=[{'id':'17:hole:31:610:103:619','shape_id':17}, {'id':'17:gap:637:556:643:569','shape_id':17}]
    def reply(self):return {'status':'done','reason':'equipment edges','operations':[],
        'hole_assessment':{'h0':{'semantic':'non_ground','reason':'equipment'},'h1':{'semantic':'non_ground','reason':'equipment'}}}
    def test_required_keys_encode_coverage_without_copying_ids(self):
        schema=review_schema(['floor'],[{'id':17}],self.holes,keyed_holes=True)['properties']['hole_assessment']
        self.assertEqual(schema['required'],['h0','h1']);self.assertFalse(schema['additionalProperties'])
    def test_roundtrip_preserves_actual_judgments_and_original_validator(self):
        reply=self.reply();out=decode_keyed_assessments(reply,self.holes)
        self.assertEqual([x['hole_id'] for x in out['hole_assessment']],[x['id'] for x in self.holes])
        self.assertIsInstance(reply['hole_assessment'],dict)
        self.assertEqual(_validate_review_result(out,self.holes,[{'id':17}],['floor'],1920,1536)['status'],'done')
    def test_missing_extra_empty_and_unknown_assessments_are_not_invented(self):
        for mode in ['missing','extra','empty','unknown']:
            r=self.reply()
            if mode=='missing':del r['hole_assessment']['h1']
            if mode=='extra':r['hole_assessment']['h2']=r['hole_assessment']['h1']
            if mode=='empty':r['hole_assessment']['h0']['reason']=''
            if mode=='unknown':r['hole_assessment']['h0']['semantic']='uncertain'
            with self.assertRaises(ValueError):decode_keyed_assessments(r,self.holes)
    def test_duplicate_json_keys_rejected_before_lossy_decoding(self):
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):
            json.loads('{"h0":1,"h0":2}',object_pairs_hook=reject_duplicate_json_keys)
    def test_ground_without_edit_still_cannot_claim_done(self):
        r=self.reply();r['hole_assessment']['h0']['semantic']='ground'
        with self.assertRaises(ValueError):_validate_review_result(decode_keyed_assessments(r,self.holes),self.holes,[{'id':17}],['floor'],1920,1536)
    def test_default_schema_remains_original_array(self):
        self.assertEqual(review_schema(['floor'],[{'id':17}],self.holes)['properties']['hole_assessment']['type'],'array')

if __name__=='__main__':unittest.main(verbosity=2)

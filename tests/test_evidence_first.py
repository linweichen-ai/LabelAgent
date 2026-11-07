"""Protect model-label fidelity in the evidence-first response decoder."""
import unittest
from tools.evidence_first import decode_result, validated_evidence_first

CATEGORIES=['pallet','person','uncertain']


def packet(value,images=2,finish='stop'):
    import json
    return dict(local_vlm=dict(backend='transformers_local_cuda',image_count=images),
                choices=[dict(finish_reason=finish,message=dict(content=json.dumps(value)))])


class EvidenceFirstIntegrity(unittest.TestCase):
    def test_explicit_suffix_is_not_rewritten_from_description(self):
        self.assertEqual(decode_result({'result':'No object visible :: pallet'},CATEGORIES),
                         dict(reason='No object visible',category='pallet'))
        self.assertEqual(decode_result({'result':'Wooden pallet structure :: uncertain'},CATEGORIES)['category'],'uncertain')

    def test_final_separator_preserves_description(self):
        self.assertEqual(decode_result({'result':'Visible :: material :: person'},CATEGORIES),
                         dict(reason='Visible :: material',category='person'))

    def test_outside_vocabulary_rejected(self):
        with self.assertRaises(ValueError):decode_result({'result':'Visible object :: car'},CATEGORIES)

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValueError):decode_result({'result':'Wood :: pallet','category':'person'},CATEGORIES)

    def test_backend_image_count_and_completion_required(self):
        for raw in (packet({'result':'Wood :: pallet'},images=1),packet({'result':'Wood :: pallet'},finish='length')):
            with self.assertRaises(ValueError):validated_evidence_first(raw,CATEGORIES,2)

    def test_normal_baseline_and_guard_are_preserved(self):
        value=dict(category='person',reason='Visible clothing')
        for n in (1,2):self.assertEqual(validated_evidence_first(packet(value,images=n),CATEGORIES,n),value)


if __name__=='__main__':unittest.main()

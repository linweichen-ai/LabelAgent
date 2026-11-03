"""Actual grammar tests for valid changes versus identity class operations."""
import json
import unittest

from source.agent import protocol_schema
from tools.reclassification_policy import exclude_identity_reclassifications,scoped_schema_policy
try:
    from lmformatenforcer import JsonSchemaParser
except ImportError:
    JsonSchemaParser=None


@unittest.skipIf(JsonSchemaParser is None,'Requires the local VLM environment')
class ReclassificationGrammarTests(unittest.TestCase):
    def accepts(self,schema,operation,sam_pending=False):
        value={'status':'continue','reason':'A supported local edit.',
            'hole_assessment':[],'operations':[operation]}
        if sam_pending:value['sam_assessment']={'improved':False,'reason':'No improvement.'}
        parser=JsonSchemaParser(schema)
        for character in json.dumps(value):
            if character not in parser.get_allowed_characters():return False
            parser=parser.add_character(character)
        return parser.can_end()

    def test_coupled_ids_and_labels_allow_corrections_and_reject_identity_in_three_classes(self):
        labels=['pallet','person','other']
        shapes=[{'id':7,'label':'person'},{'id':9,'label':'pallet'},{'id':11,'label':'other'}]
        baseline=protocol_schema.review_schema(labels,shapes,[])
        constrained=exclude_identity_reclassifications(baseline,labels,shapes)
        for shape in shapes:
            for target in labels:
                operation={'operation':'reclassify','shape_ids':[shape['id']],'label':target,'reason':'Visual evidence.'}
                self.assertTrue(self.accepts(baseline,operation))
                self.assertEqual(self.accepts(constrained,operation),target!=shape['label'])
        self.assertEqual(baseline,protocol_schema.review_schema(labels,shapes,[]))

    def test_single_class_has_no_reclassification_but_preserves_other_tools(self):
        labels=['person'];shapes=[{'id':7,'label':'person'}]
        schema=exclude_identity_reclassifications(protocol_schema.review_schema(labels,shapes,[]),labels,shapes)
        self.assertFalse(self.accepts(schema,{'operation':'reclassify','shape_ids':[7],'label':'person','reason':'No change.'}))
        self.assertTrue(self.accepts(schema,{'operation':'remove','shape_ids':[7],'reason':'Visible false positive.'}))

    def test_pending_sam_assessment_and_allowed_ids_stay_enforced(self):
        labels=['pallet','person'];shapes=[{'id':7,'label':'pallet'}]
        schema=exclude_identity_reclassifications(protocol_schema.review_schema(labels,shapes,[],True),labels,shapes)
        operation={'operation':'reclassify','shape_ids':[7],'label':'person','reason':'Visible person.'}
        self.assertTrue(self.accepts(schema,operation,True))
        self.assertFalse(self.accepts(schema,operation,False))
        operation['shape_ids']=[999]
        self.assertFalse(self.accepts(schema,operation,True))


class ReclassificationPolicyTests(unittest.TestCase):
    def test_unknown_candidate_facts_fail_closed(self):
        with self.assertRaises(ValueError):
            exclude_identity_reclassifications({},['person'],[{'id':7}])
        with self.assertRaises(ValueError):
            exclude_identity_reclassifications({},['person'],[{'id':7,'label':'person'},{'id':7,'label':'person'}])

    def test_scoped_policy_restores_production_function_after_errors_and_keeps_default(self):
        original=protocol_schema.review_schema
        with scoped_schema_policy('baseline'):self.assertIs(protocol_schema.review_schema,original)
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with scoped_schema_policy('exclude_identity'):
                self.assertIsNot(protocol_schema.review_schema,original)
                raise RuntimeError('stop')
        self.assertIs(protocol_schema.review_schema,original)


if __name__=='__main__':unittest.main()

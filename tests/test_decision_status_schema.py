import json
import unittest

from source.agent.protocol_schema import review_schema
from tools.decision_status_schema import continue_requires_operation
try:
    from lmformatenforcer import JsonSchemaParser
except ImportError:
    JsonSchemaParser=None


@unittest.skipIf(JsonSchemaParser is None,'Requires the local VLM environment')
class DecisionStatusSchemaTests(unittest.TestCase):
    def accepts(self,schema,value):
        parser=JsonSchemaParser(schema)
        for character in json.dumps(value):
            if character not in parser.get_allowed_characters():return False
            parser=parser.add_character(character)
        return parser.can_end()

    def schema(self):return review_schema(['pallet','person'],[{'id':7,'label':'pallet'}],[])
    def decision(self,status,operations):return {'status':status,'reason':'Visible evidence.','operations':operations,'hole_assessment':[]}

    def test_done_is_still_allowed_and_continue_empty_is_rejected(self):
        baseline=self.schema();guarded=continue_requires_operation(baseline)
        self.assertTrue(self.accepts(baseline,self.decision('continue',[])))
        self.assertTrue(self.accepts(guarded,self.decision('done',[])))
        self.assertFalse(self.accepts(guarded,self.decision('continue',[])))
        self.assertEqual(baseline,self.schema())

    def test_model_can_choose_valid_tools_without_forcing_a_specific_label(self):
        guarded=continue_requires_operation(self.schema())
        operations=[{'operation':'remove','shape_ids':[7],'reason':'False positive.'},
            {'operation':'reclassify','shape_ids':[7],'label':'person','reason':'Visible person.'},
            {'operation':'reclassify','shape_ids':[7],'label':'pallet','reason':'Same class, still syntactically allowed.'}]
        for operation in operations:self.assertTrue(self.accepts(guarded,self.decision('continue',[operation])))
        self.assertTrue(self.accepts(guarded,self.decision('done',[operations[0]])))

    def test_unknown_ids_and_unsupported_labels_remain_rejected(self):
        guarded=continue_requires_operation(self.schema())
        for identifier,label in [(999,'person'),(7,'invented')]:
            operation={'operation':'reclassify','shape_ids':[identifier],'label':label,'reason':'Invalid.'}
            self.assertFalse(self.accepts(guarded,self.decision('continue',[operation])))


if __name__=='__main__':unittest.main()

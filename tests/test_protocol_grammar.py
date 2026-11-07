"""Exercise the actual optional decoding grammar, including early string endings."""
import json
import unittest

from source.agent.protocol_schema import description_schema, review_schema

try:
    from lmformatenforcer import JsonSchemaParser
except ImportError:
    JsonSchemaParser = None


@unittest.skipIf(JsonSchemaParser is None, 'Requires the local VLM environment')
class ProtocolGrammarTests(unittest.TestCase):
    def accepts(self, schema, value):
        parser = JsonSchemaParser(schema)
        for character in json.dumps(value, ensure_ascii=True):
            if character not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(character)
        return parser.can_end()

    def test_short_description_and_escaped_evidence_are_not_forced_to_the_limit(self):
        value = {'description': 'Label the "person" outline; preserve visible parts.',
                 'prompts': [{'label': 'person', 'text': 'person'}]}
        self.assertTrue(self.accepts(description_schema(['person']), value))

    def test_decoding_rejects_oversized_description_and_prompt(self):
        schema = description_schema(['person'])
        for description, prompt in [('x' * 701, 'person'), ('Visible person', 'x' * 41)]:
            self.assertFalse(self.accepts(schema, {'description': description,
                'prompts': [{'label': 'person', 'text': prompt}]}))

    def test_review_allows_short_reason_but_rejects_unknown_ids_and_holes(self):
        schema = review_schema(['person'], [{'id': 7}], [])
        valid = {'status': 'continue', 'reason': 'Inspect the "person" label.',
                 'hole_assessment': [], 'operations': [
                     {'operation': 'remove', 'shape_ids': [7], 'reason': 'Duplicate instance.'}]}
        self.assertTrue(self.accepts(schema, valid))
        valid['operations'][0]['shape_ids'] = [999]
        self.assertFalse(self.accepts(schema, valid))
        self.assertFalse(self.accepts(schema, {'status': 'done', 'reason': 'No edit.',
            'hole_assessment': [{'hole_id': 'invented', 'semantic': 'ground', 'reason': 'Unsupported'}],
            'operations': []}))

    def test_review_reason_length_limit_and_pending_sam_assessment_remain_enforced(self):
        schema = review_schema(['person'], [{'id': 7}], [], sam_pending=True)
        value = {'status': 'done', 'reason': 'No supported edit.', 'hole_assessment': [],
                 'operations': [], 'sam_assessment': {'improved': False, 'reason': 'No visible gain.'}}
        self.assertTrue(self.accepts(schema, value))
        value['reason'] = 'x' * 181
        self.assertFalse(self.accepts(schema, value))
        value['reason'] = 'No edit.'
        del value['sam_assessment']
        self.assertFalse(self.accepts(schema, value))


if __name__ == '__main__':
    unittest.main()

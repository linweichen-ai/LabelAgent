"""Prevent prompt experiments from changing tools, images or production state."""
import json
import unittest
from unittest.mock import patch

from source.agent import workflow, protocol_schema
from tools.evidence_order import REASON_FIRST_HINT, scoped_evidence_order
try:
    from lmformatenforcer import JsonSchemaParser
except ImportError:
    JsonSchemaParser = None


class EvidenceOrderTests(unittest.TestCase):
    def test_only_reason_first_appends_hint_and_reports_actual_output_order(self):
        schema = protocol_schema.review_schema(['person'], [{'id': 7, 'label': 'person'}], [])
        images = (b'original', b'reference')
        for policy in ('baseline', 'reason_first'):
            observations = []
            response = {'status': 'done', 'reason': 'No supported defect.', 'hole_assessment': [], 'operations': []}
            with patch.object(workflow, 'chat', return_value=response) as original:
                with scoped_evidence_order(policy, observations.append):
                    result = workflow.chat({}, 'unchanged prefix', images[0], extra_images=images[1:], response_schema=schema)
                self.assertIs(workflow.chat, original)
                args, kwargs = original.call_args
                self.assertEqual(args[1], 'unchanged prefix' + (REASON_FIRST_HINT if policy == 'reason_first' else ''))
                self.assertEqual((args[2], *kwargs['extra_images']), images)
                self.assertIs(kwargs['response_schema'], schema)
                self.assertIs(result, response)
                self.assertFalse(observations[0]['reason_first_observed'])

    def test_exception_restores_chat_and_does_not_fabricate_response(self):
        observations = []
        with patch.object(workflow, 'chat', side_effect=RuntimeError('actual failure')) as original:
            with self.assertRaisesRegex(RuntimeError, 'actual failure'):
                with scoped_evidence_order('reason_first', observations.append):
                    workflow.chat({}, 'prefix', b'image')
            self.assertIs(workflow.chat, original)
        self.assertEqual(observations, [])
        with self.assertRaises(ValueError):
            with scoped_evidence_order('invalid', observations.append):
                pass

    @unittest.skipIf(JsonSchemaParser is None, 'Requires local VLM environment')
    def test_actual_unchanged_grammar_permits_reason_first_done_and_real_edit(self):
        schema = protocol_schema.review_schema(['pallet', 'person'], [{'id': 7, 'label': 'pallet'}], [])
        for status, operations in [('done', []), ('continue', [{'operation': 'reclassify', 'shape_ids': [7], 'label': 'person', 'reason': 'Visible person.'}])]:
            value = {'reason': 'Visible evidence.', 'status': status, 'hole_assessment': [], 'operations': operations}
            parser = JsonSchemaParser(schema)
            for character in json.dumps(value):
                self.assertIn(character, parser.get_allowed_characters())
                parser = parser.add_character(character)
            self.assertTrue(parser.can_end())


if __name__ == '__main__':
    unittest.main()

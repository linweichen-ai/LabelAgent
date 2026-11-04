import json
import unittest
from unittest.mock import patch
from source.agent import workflow


class CompactReplyBudgetTests(unittest.TestCase):
    def response(self, request, **kwargs):
        self.payload = json.loads(request.data)
        class Reply:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({'choices': [{'message': {'content': '{"status":"done"}'}}]}).encode()
        return Reply()

    def config(self, **extra):
        return {'llm_url': 'http://127.0.0.1:1/v1', 'llm_model': 'unit-test',
                'llm_prompt_profile': 'compact', **extra}

    def test_default_budget_preserved(self):
        with patch.object(workflow.urllib.request, 'urlopen', self.response):
            workflow._chat(self.config(), 'unit test', b'not-a-model-request')
        self.assertEqual(self.payload['max_tokens'], 768)

    def test_explicit_budget_keeps_schema_and_prompt(self):
        schema = {'type': 'object', 'properties': {'status': {'type': 'string'}}}
        with patch.object(workflow.urllib.request, 'urlopen', self.response):
            workflow._chat(self.config(llm_max_tokens=2048), 'same prompt', b'unit-test', response_schema=schema)
        self.assertEqual(self.payload['max_tokens'], 2048)
        self.assertEqual(self.payload['response_format']['json_schema']['schema'], schema)
        self.assertEqual(self.payload['messages'][0]['content'][0]['text'], 'same prompt')

    def test_invalid_budget_refused_before_transport(self):
        for value in (0, 255, 4097, True, '2048'):
            with self.subTest(value=value), patch.object(workflow.urllib.request, 'urlopen') as transport:
                with self.assertRaises(ValueError):
                    workflow._chat(self.config(llm_max_tokens=value), 'unit', b'unit')
                transport.assert_not_called()


if __name__ == '__main__': unittest.main()

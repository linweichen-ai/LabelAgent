import unittest

from tools.run_new_dataset_smoke import model_config, summarize_model_trace


class NewDatasetSmokeTests(unittest.TestCase):
    def test_model_override_is_in_memory_and_rejects_credentials(self):
        source = {"llm_url": "http://old/v1", "llm_model": "old",
                  "sam3_url": "http://sam"}
        updated = model_config(source, "http://127.0.0.1:38868/v1", "new")
        self.assertEqual(updated["llm_url"], "http://127.0.0.1:38868/v1")
        self.assertEqual(updated["llm_model"], "new")
        self.assertEqual(source["llm_model"], "old")
        with self.assertRaises(ValueError):
            model_config(source, "http://user:secret@127.0.0.1/v1", None)

    def test_trace_summary_excludes_prompt_and_model_prose(self):
        trace = [{"service": "GPT", "status": "succeeded", "elapsed_ms": 10,
                  "input": {"instruction": "private prompt", "images": [{}]},
                  "output": {"raw_text": "private model prose",
                             "parsed": {"status": "done"},
                             "local_vlm": {"quantization": "nf4"}}}]
        summary = summarize_model_trace(trace)
        self.assertEqual(summary[0]["raw_text_chars"], 19)
        self.assertNotIn("instruction", summary[0])
        self.assertNotIn("raw_text", summary[0])


if __name__ == "__main__":
    unittest.main()

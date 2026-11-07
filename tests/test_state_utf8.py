import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from source.agent import server


class StateUTF8Tests(unittest.TestCase):
    def test_unicode_state_roundtrip_does_not_depend_on_windows_default_encoding(self):
        labels = ["pallet", "\u76ee\u6807\U0001f7e6"]
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "WORK", Path(directory)):
            with patch.dict(server.state, {"video": None, "labels": labels}, clear=False):
                server.save()
                stored = json.loads(server.state_path().read_bytes().decode("utf-8"))
                self.assertEqual(stored["labels"], labels)
                server.state["labels"] = []
                server.load()
                self.assertEqual(server.state["labels"], labels)


if __name__ == "__main__":
    unittest.main()

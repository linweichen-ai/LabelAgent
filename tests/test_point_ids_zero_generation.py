"""Integration regression for a completed planner with no usable proposals."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _fixture_paths():
    here = Path(__file__).resolve()
    # Published tests use the repository's own source; only local helper runs
    # fall back to the previously frozen source snapshot.
    roots = list(here.parents)
    snapshot = next((root for root in roots if (root / "source/agent/workflow.py").is_file()), None)
    if snapshot is None:
        snapshots = (root / "task_workspaces/labelagent_point_ids_20261107/source_snapshot" for root in roots)
        snapshot = next((root for root in snapshots if (root / "source/agent/workflow.py").is_file()), None)
    generators = [here.with_name("generate_point_ids.py")]
    for root in roots:
        generators.extend([root / "tools/labelagent_experiments/generate_point_ids.py",
                           root / "tools/labelagent_live_handoff/generate_point_ids.py"])
    generator = next((path for path in generators if path.is_file()), None)
    if snapshot is None or generator is None:
        raise FileNotFoundError("A repository workflow and point-ID generator fixture are required")
    return snapshot, generator


_ISOLATED_RUN = r'''
import contextlib, importlib.util, io, json, sys
from pathlib import Path
from unittest import mock
snapshot, generator_file, base = map(Path, sys.argv[1:])
sys.path[:0] = [str(snapshot), str(generator_file.parent)]
from source.agent import workflow
assert Path(workflow.__file__).resolve() == (snapshot / "source/agent/workflow.py").resolve()
spec = importlib.util.spec_from_file_location("isolated_zero_point_ids_generator", generator_file)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
work, out = base / "work", base / "out"
with (mock.patch.object(generator, "work", work),
      mock.patch.object(generator, "out", out),
      mock.patch.object(workflow, "sam_request", side_effect=AssertionError("No SAM call is allowed")) as sam,
      mock.patch("urllib.request.urlopen", side_effect=AssertionError("No network/status call is allowed")) as network,
      contextlib.redirect_stdout(io.StringIO())):
    generator.main()
sam.assert_not_called()
network.assert_not_called()
print(json.dumps({"actual_main_called": True, "sam_calls": sam.call_count,
                  "network_calls": network.call_count, "source_root": str(snapshot)}))
'''


class PointIdsZeroGenerationTests(unittest.TestCase):
    def test_zero_plans_write_complete_empty_artifacts_without_model_or_status_requests(self):
        snapshot, generator_file = _fixture_paths()
        original_path = list(sys.path)
        original_source_modules = {key: value for key, value in sys.modules.items()
                                   if key == "source" or key.startswith("source.")}
        temporary_parent = Path(tempfile.gettempdir()).resolve()
        with tempfile.TemporaryDirectory(prefix="labelagent-zero-point-plans-", dir=temporary_parent) as name:
            base = Path(name).resolve()
            self.assertEqual(base.parent, temporary_parent)
            work = base / "work"
            out = base / "out"
            planning = out / "planning"
            work.mkdir()
            planning.mkdir(parents=True)
            (work / "config.json").write_text(json.dumps({
                "source_snapshot": str(snapshot),
                "sam_url": "http://127.0.0.1:1/forbidden-in-zero-plan-test",
            }), encoding="utf-8")
            (planning / "completed.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            (planning / "progress.json").write_text("[]\n", encoding="utf-8")

            # A fresh interpreter prevents other tests' cached source packages
            # from changing this fixture, and leaves their imports untouched.
            run = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", _ISOLATED_RUN,
                                  str(snapshot), str(generator_file), str(base)],
                                 capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            receipt = json.loads(run.stdout)
            self.assertTrue(receipt["actual_main_called"])
            self.assertEqual(receipt["sam_calls"], 0)
            self.assertEqual(receipt["network_calls"], 0)
            self.assertEqual(Path(receipt["source_root"]).resolve(), snapshot.resolve())

            destination = out / "seam_generation"
            read = lambda file: json.loads((destination / file).read_text(encoding="utf-8"))
            completed = read("completed.json")
            self.assertEqual(completed["status"], "complete")
            for field in ("real_scene_SAM_requests", "synthetic_control_requests", "valid_complete_outputs", "model_trace_records"):
                self.assertEqual(completed[field], 0, field)
            self.assertEqual(read("progress.json"), [])
            self.assertEqual(read("all_hypotheses.json"), [])
            self.assertEqual(read("blank_controls.json")["rows"], [])
            self.assertTrue(read("blank_controls.json")["not_accuracy"])
            self.assertFalse((destination / "model_trace.json").exists())
        self.assertEqual(sys.path, original_path)
        self.assertEqual({key: value for key, value in sys.modules.items()
                          if key == "source" or key.startswith("source.")}, original_source_modules)


if __name__ == "__main__":
    unittest.main()

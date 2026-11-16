"""Regression for a first inference blocked by a network-mounted default cache."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('runtime_cache_server',Path(__file__).resolve().parents[1]/'tools/local_vlm_server.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeCacheTests(unittest.TestCase):
    def test_default_leaves_process_cache_environment_unchanged(self):
        with patch.dict(os.environ,{'XDG_CACHE_HOME':'existing-network-cache'}):
            before=dict(os.environ)
            self.assertEqual(MODULE.configure_runtime_caches(None),{})
            self.assertEqual(dict(os.environ),before)

    def test_explicit_local_cache_does_not_touch_existing_network_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            old=root/'old-network-cache'
            old.mkdir()
            sentinel=old/'keep.txt'
            sentinel.write_bytes(b'preserve this cache')
            with patch.dict(os.environ,{'XDG_CACHE_HOME':str(old),'TRITON_CACHE_DIR':str(old)}):
                configured=MODULE.configure_runtime_caches(root/'new-runtime-cache')
                self.assertEqual(len(configured),7)
                for key,value in configured.items():
                    self.assertEqual(os.environ[key],value)
                    self.assertTrue(Path(value).is_dir())
                    self.assertTrue(Path(value).is_relative_to((root/'new-runtime-cache').resolve()))
                self.assertEqual(sentinel.read_bytes(),b'preserve this cache')
                self.assertEqual(sorted(item.name for item in old.iterdir()),['keep.txt'])

    def test_preexisting_cache_file_is_preserved_across_reconfiguration(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ):
            configured=MODULE.configure_runtime_caches(directory)
            sentinel=Path(configured['TRITON_CACHE_DIR'])/'compiled.bin'
            sentinel.write_bytes(b'already compiled')
            self.assertEqual(MODULE.configure_runtime_caches(directory),configured)
            self.assertEqual(sentinel.read_bytes(),b'already compiled')


if __name__=='__main__':unittest.main()

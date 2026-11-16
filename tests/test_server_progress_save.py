import ctypes
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from source.agent import server as M

class ServerProgressSaveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.work_patch=patch.object(M,'WORK',Path(self.temp.name));self.work_patch.start();self.addCleanup(self.work_patch.stop)
        self.path=Path(self.temp.name)/'last_run.json';self.path.write_text('previous',encoding='utf-8')

    def test_success_keeps_utf8_json_serialization_exact(self):
        value={'类别':'托盘','progress':[1,2,3]}
        M.save_run(value)
        expected=json.dumps(value,ensure_ascii=False,indent=2)
        self.assertEqual(self.path.read_text('utf-8'),expected)
        self.assertFalse(self.path.with_suffix('.tmp').exists())

    def test_transient_replace_denial_retries_without_rewriting_json_or_losing_previous(self):
        original=Path.replace;attempts=[]
        def replace(path,target):
            attempts.append(1)
            if len(attempts)<3:
                self.assertEqual(self.path.read_text('utf-8'),'previous')
                raise PermissionError('transient sharing violation')
            return original(path,target)
        with patch.object(Path,'replace',replace),patch.object(M.time,'sleep') as sleep:
            M.save_run({'complete':True})
        self.assertEqual(len(attempts),3);self.assertEqual(sleep.call_count,2)
        self.assertEqual(json.loads(self.path.read_text('utf-8')),{'complete':True})

    def test_permanent_denial_is_bounded_and_other_errors_are_not_hidden(self):
        with patch.object(Path,'replace',side_effect=PermissionError('denied')) as replace,patch.object(M.time,'sleep') as sleep:
            with self.assertRaises(PermissionError):M.save_run({'new':'value'})
        self.assertEqual(replace.call_count,10);self.assertEqual(sleep.call_count,9)
        self.assertEqual(self.path.read_text('utf-8'),'previous')
        self.assertTrue(self.path.with_suffix('.tmp').is_file())
        with patch.object(Path,'replace',side_effect=OSError('disk unavailable')) as replace,patch.object(M.time,'sleep') as sleep:
            with self.assertRaises(OSError):M.save_run({'new':'value'})
            self.assertEqual(replace.call_count,1);sleep.assert_not_called()

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_real_windows_reader_handle_can_release_and_allow_atomic_commit(self):
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
        handle=kernel.CreateFileW(str(self.path),0x80000000,0x1|0x2,None,3,0,None)
        self.assertNotEqual(handle,wintypes.HANDLE(-1).value)
        reached_retry=threading.Event();attempts=[];original=Path.replace
        def replace(path,target):
            attempts.append(1)
            if len(attempts)>=2:reached_retry.set()
            return original(path,target)
        with ThreadPoolExecutor(max_workers=1) as pool,patch.object(Path,'replace',replace):
            future=pool.submit(M.save_run,{'complete':'真实写入'})
            try:
                self.assertTrue(reached_retry.wait(3),'Expected a real Windows sharing denial and retry')
                self.assertEqual(self.path.read_text('utf-8'),'previous')
            finally:kernel.CloseHandle(handle)
            future.result(timeout=3)
        self.assertGreaterEqual(len(attempts),2)
        self.assertEqual(json.loads(self.path.read_text('utf-8')),{'complete':'真实写入'})


if __name__=='__main__':unittest.main(verbosity=2)

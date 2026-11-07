"""Verify opt-in allocator controls and a deadline that exits only this process."""
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

SPEC=importlib.util.spec_from_file_location('probe_limited_server',Path(__file__).resolve().parents[1]/'tools/local_vlm_server.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProbeLimitTests(unittest.TestCase):
    def test_default_backend_does_not_modify_the_allocator(self):
        call=Mock()
        torch=SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace(set_per_process_memory_fraction=call)))
        MODULE.configure_cuda_fraction(torch,None)
        call.assert_not_called()

    def test_probe_sets_cap_on_the_selected_visible_device(self):
        call=Mock()
        torch=SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace(set_per_process_memory_fraction=call)))
        MODULE.configure_cuda_fraction(torch,.35)
        call.assert_called_once_with(.35,device=0)

    def test_invalid_fraction_rejected_before_GPU_calls(self):
        call=Mock()
        torch=SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace(set_per_process_memory_fraction=call)))
        with self.assertRaises(ValueError):MODULE.configure_cuda_fraction(torch,.5)
        call.assert_not_called()

    def test_deadline_has_no_foreign_process_target_and_is_bounded(self):
        callbacks=[]
        class Timer:
            def __init__(self,seconds,callback):self.seconds=seconds;callbacks.append(callback)
            def start(self):pass
        with patch.object(MODULE.threading,'Timer',Timer),patch.object(MODULE.os,'_exit') as leave,patch.object(MODULE.sys,'stderr',io.StringIO()):
            timer=MODULE.arm_probe_deadline(120)
            self.assertTrue(timer.daemon)
            callbacks[0]()
            leave.assert_called_once_with(124)
            with self.assertRaises(ValueError):MODULE.arm_probe_deadline(3600)

    def test_changed_headroom_rejects_before_reading_images_or_running_model(self):
        cuda=SimpleNamespace(mem_get_info=lambda device:(1000*1024**2,24564*1024**2),
            memory_reserved=lambda device:8000*1024**2)
        engine=MODULE.TransformersBackend.__new__(MODULE.TransformersBackend)
        engine.torch=SimpleNamespace(cuda=cuda)
        engine.process_allocator_fraction=.35
        engine.probe_reserved_free_mib=4096
        engine.processor=Mock()
        with self.assertRaisesRegex(RuntimeError,'headroom changed'):engine.generate([],[],10)
        engine.processor.assert_not_called()
        engine.processor.apply_chat_template.assert_not_called()

    def test_runtime_probe_keeps_remaining_allocator_budget_and_reserve(self):
        cuda=SimpleNamespace(mem_get_info=lambda device:(7500*1024**2,24564*1024**2),
            memory_reserved=lambda device:7500*1024**2)
        MODULE.require_runtime_probe_headroom(SimpleNamespace(cuda=cuda),.35,4096)


if __name__=='__main__':unittest.main()

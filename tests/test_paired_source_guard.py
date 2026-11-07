import unittest
from unittest.mock import patch,Mock
from paired_source_cue import external_live


class SourceGuardTests(unittest.TestCase):
    def test_completed_child_cleared_to_none_is_terminal(self):
        with patch('paired_source_cue.old.read',return_value={'status':'complete','pid':None,'create_time':None,'child_pid':None,'child_create_time':None}),patch('psutil.Process',side_effect=AssertionError('No null PID lookup')):
            self.assertEqual(external_live(),('complete',False))
    def test_actual_matching_live_process_blocks_own_requests(self):
        p=Mock();p.is_running.return_value=True;p.create_time.return_value=100.
        with patch('paired_source_cue.old.read',return_value={'status':'running','pid':2,'create_time':100.,'child_pid':None,'child_create_time':None}),patch('psutil.Process',return_value=p):self.assertEqual(external_live(),('running',True))
    def test_reused_pid_does_not_match_old_process(self):
        p=Mock();p.is_running.return_value=True;p.create_time.return_value=200.
        with patch('paired_source_cue.old.read',return_value={'status':'complete','pid':2,'create_time':100.,'child_pid':None,'child_create_time':None}),patch('psutil.Process',return_value=p):self.assertEqual(external_live(),('complete',False))


if __name__=='__main__':unittest.main()

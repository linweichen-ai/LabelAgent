import unittest
from prepare_palette_parts import np
from native_review_feedback import describe_native,review_response

class Feedback(unittest.TestCase):
    def test_negative_center_empty_but_neighborhood_foreground_is_rejected(self):
        m=np.zeros((40,40),'uint8');m[10:14,10:14]=1;before=m.copy();d=describe_native(m,[{'x':11,'y':11,'label':1},{'x':16,'y':10,'label':0}]);self.assertFalse(d['native_prompt_passed']);self.assertFalse(d['failures'][0]['center_foreground']);self.assertGreater(d['failures'][0]['foreground_pixels_in_required_region'],0);np.testing.assert_array_equal(m,before)
    def test_input_border_is_separate_from_click_validity(self):
        m=np.zeros((40,40),'uint8');m[10:14,10:14]=1;m[0,25]=1;d=describe_native(m,[{'x':11,'y':11,'label':1}]);self.assertTrue(d['native_prompt_guard']['valid']);self.assertFalse(d['native_prompt_passed'])
    def test_missing_native_selection_never_grants_write_or_fourth_round(self):
        d=review_response({'warnings':[]},[],3);self.assertFalse(d['annotation_write_permitted']);self.assertEqual(d['next_action'],'abstain_at_limit');self.assertEqual(d['remaining_rounds'],0)
        with self.assertRaises(ValueError):review_response({},[],4)
    def test_native_pass_is_not_final_instance_acceptance(self):
        m=np.zeros((40,40),'uint8');m[10:14,10:14]=1;d=describe_native(m,[{'x':11,'y':11,'label':1}]);self.assertTrue(d['native_prompt_passed']);self.assertFalse(d['final_instance_acceptance_granted'])

if __name__=='__main__':unittest.main()

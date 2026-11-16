from copy import deepcopy
import unittest
from tools.crop_response_binding import check_crop_receipt

class CropBindingTests(unittest.TestCase):
    def setUp(self):
        self.expected={'topic':'/capture/camera_0/image','frame':15,'tile_index':0,'bounds':[0,0,20,20],
            'source_image_sha256':'image','crop_png_sha256':'crop','query':'wooden pallet','plan_sha256':'plan'}
        self.receipt={**deepcopy(self.expected),'response':{'status':'ok','results':[]}}

    def test_complete_empty_success_is_a_valid_bound_response(self):
        before=deepcopy(self.receipt);check_crop_receipt(self.receipt,self.expected);self.assertEqual(self.receipt,before)

    def test_different_query_image_crop_or_plan_is_rejected(self):
        for field in self.expected:
            value=deepcopy(self.receipt);value[field]='different'
            with self.assertRaises(ValueError):check_crop_receipt(value,self.expected)

    def test_error_or_malformed_response_is_not_replayed(self):
        for result in ({'status':'error','results':[]},{'status':'ok','results':None},None):
            value=deepcopy(self.receipt);value['response']=result
            with self.assertRaises(ValueError):check_crop_receipt(value,self.expected)

if __name__=='__main__':unittest.main()

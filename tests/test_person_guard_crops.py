from copy import deepcopy
import unittest
from source.agent.edit_agent import _rasterize
from tools.person_guard_crops import PURPOSE, check_person_guard_receipt, guard_pending_request, remap_person_guard_result


def result():
    return {'score': .8, 'parts': [
        {'points': [[0,0],[9,0],[9,9],[0,9]], 'holes': [[[3,3],[6,3],[6,6],[3,6]]]},
        {'points': [[12,1],[15,1],[15,4],[12,4]], 'holes': []}]}


class PersonGuardCropTests(unittest.TestCase):
    def test_all_parts_and_holes_shift_without_mutating_response(self):
        raw=result();before=deepcopy(raw)
        shape=remap_person_guard_result(raw,(20,30,40,50),80,80,'guard-1')
        self.assertEqual(raw,before)
        self.assertEqual(shape['label'],'person')
        self.assertEqual(shape['purpose'],PURPOSE)
        self.assertEqual(shape['parts'][1]['points'][0],[32,31])
        self.assertEqual(shape['holes'][0][0],[23,33])

    def test_internal_crop_edge_fragment_is_retained_as_guard(self):
        self.assertIsNotNone(remap_person_guard_result(result(),(20,30,40,50),80,80,'guard-1'))

    def test_holes_and_disconnected_parts_survive_rasterization(self):
        shape=remap_person_guard_result(result(),(20,30,40,50),80,80,'guard-1')
        mask=_rasterize(shape,80,80)
        self.assertTrue(mask[31,21])
        self.assertFalse(mask[34,24])
        self.assertTrue(mask[32,33])

    def test_invalid_geometry_confidence_and_bounds_are_rejected(self):
        for value in (float('nan'),1.1,None,True):
            raw=result();raw['score']=value
            with self.assertRaises(ValueError):remap_person_guard_result(raw,(20,30,40,50),80,80,'g')
        raw=result();raw['parts'][0]['points'][0]=[-1,0]
        with self.assertRaises(ValueError):remap_person_guard_result(raw,(20,30,40,50),80,80,'g')
        with self.assertRaises(ValueError):remap_person_guard_result(result(),(20,30,81,50),80,80,'g')

    def test_exact_native_person_response_binds_to_crop_and_plan(self):
        expected={'topic':'camera','frame':15,'tile_index':0,'bounds':[0,0,20,20],
                  'source_image_sha256':'image','crop_png_sha256':'crop','query':'person','plan_sha256':'plan'}
        receipt={**expected,'label':'person','purpose':PURPOSE,'response':{'results':[]}}
        check_person_guard_receipt(receipt,expected)
        for key in ('crop_png_sha256','plan_sha256','bounds'):
            bad=deepcopy(receipt);bad[key]='other'
            with self.assertRaises(ValueError):check_person_guard_receipt(bad,expected)

    def test_pallet_query_class_or_non_guard_receipt_is_rejected(self):
        expected={'topic':'camera','frame':15,'tile_index':0,'bounds':[0,0,20,20],
                  'source_image_sha256':'image','crop_png_sha256':'crop','query':'person','plan_sha256':'plan'}
        receipt={**expected,'label':'person','purpose':PURPOSE,'response':{'results':[]}}
        for key,value in (('label','pallet'),('purpose','annotation'),('query','pallet')):
            bad={**receipt,key:value}
            with self.assertRaises(ValueError):check_person_guard_receipt(bad,expected)

    def test_uncertain_missing_response_blocks_repeat_but_busy_rejection_is_safe(self):
        for status in ('awaiting_response','complete','unknown'):
            with self.assertRaises(ValueError):guard_pending_request({'status':status},False)
        guard_pending_request({'status':'rejected_busy_before_activation'},False)
        guard_pending_request({'status':'complete'},True)
        guard_pending_request(None,False)


if __name__=='__main__':unittest.main()

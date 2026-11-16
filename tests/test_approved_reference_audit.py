from copy import deepcopy
import unittest
from tools.audit_approved_reference import consistency


def rect(identifier,x0,y0,x1,y1,label='pallet'):
    return {'id':identifier,'label':label,'points':[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],'holes':[]}


class ApprovedReferenceAuditTests(unittest.TestCase):
    def test_duplicate_candidate_does_not_increase_one_to_one_coverage(self):
        reference=[rect(1,2,2,7,7)];predictions=[rect('a',2,2,7,7),rect('b',2,2,7,7)]
        row=consistency(reference,predictions,20,16)['pallet']
        self.assertEqual(row['matches']['0.5']['matched_approved_objects'],1)
        self.assertEqual(row['matches']['0.5']['unmatched_prediction_objects'],1)
        self.assertEqual(row['union_iou_against_approved'],1.)
        self.assertEqual(row['approved_objects_with_multiple_candidate_hits_at_0_5'],1)

    def test_identical_pixels_with_wrong_class_do_not_match(self):
        rows=consistency([rect(1,2,2,7,7)],[rect('a',2,2,7,7,'person')],20,16)
        self.assertEqual(rows['pallet']['matches']['0.5']['matched_approved_objects'],0)
        self.assertEqual(rows['pallet']['approved_union_coverage'],0.)
        self.assertEqual(rows['person']['approved_objects'],0)

    def test_unapproved_regions_are_counted_but_not_called_errors(self):
        rows=consistency([rect(1,2,2,5,5)],[rect('a',2,2,5,5),rect('b',12,2,15,5)],20,16)
        row=rows['pallet'];self.assertEqual(row['approved_union_coverage'],1.)
        self.assertEqual(row['prediction_pixels_outside_approved_union'],16)
        self.assertFalse(row['outside_pixels_are_verified_errors'])
        self.assertFalse(row['matches']['0.5']['unmatched_predictions_are_verified_errors'])
        self.assertNotIn('precision',row);self.assertNotIn('accuracy',row)

    def test_one_candidate_cannot_cover_two_approved_objects_in_matching(self):
        refs=[rect(1,2,2,5,5),rect(2,6,2,9,5)]
        row=consistency(refs,[rect('a',2,2,9,5)],20,16)['pallet']
        self.assertEqual(row['matches']['0.5']['matched_approved_objects'],1)
        self.assertEqual(row['matches']['0.5']['unmatched_approved_objects'],1)

    def test_holes_are_preserved_and_inputs_are_immutable(self):
        reference=[rect(1,1,1,12,12)];reference[0]['holes']=[[[4,4],[8,4],[8,8],[4,8]]]
        predictions=[rect('archived-string-id',1,1,12,12)]
        before=deepcopy((reference,predictions));row=consistency(reference,predictions,20,16)['pallet']
        self.assertEqual(row['prediction_pixels_outside_approved_union'],25)
        self.assertLess(row['union_iou_against_approved'],1.)
        self.assertEqual((reference,predictions),before)

    def test_invalid_geometry_or_labels_fail_closed(self):
        with self.assertRaises(ValueError):consistency([rect(1,1,1,5,5)],[rect(1,1,1,5,5,'unknown')],20,16)
        with self.assertRaises(ValueError):consistency([rect(1,-3,1,5,5)],[],20,16)


if __name__=='__main__':unittest.main()

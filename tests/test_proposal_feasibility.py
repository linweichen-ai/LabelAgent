"""Bounds do not become deployable reference-selected candidate exports."""
import unittest
import numpy as np
from tools.diagnose_proposal_feasibility import matching_bound


class ProposalFeasibilityTests(unittest.TestCase):
    def test_original_matching_rank_and_incremental_budget_bound(self):
        matrix=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float)
        self.assertEqual(matching_bound(matrix,[0],[1,2],0),1)
        self.assertEqual(matching_bound(matrix,[0],[1,2],1),2)
        self.assertEqual(matching_bound(matrix,[0],[1,2],5),3)
        self.assertEqual(matching_bound(matrix,[0],[0,0],5),1)

    def test_duplicate_object_columns_cannot_inflate_bound(self):
        matrix=np.array([[1,1,1],[0,0,0]],dtype=float)
        self.assertEqual(matching_bound(matrix,[0],[1,2],2),1)
        self.assertEqual(matching_bound(np.zeros((0,3)),[],[0,1],2),0)
        with self.assertRaises(ValueError):matching_bound(matrix,[0],[1],-1)


if __name__=='__main__':unittest.main()

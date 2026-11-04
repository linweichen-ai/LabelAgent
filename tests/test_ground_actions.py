import sys,io,tempfile,unittest
from pathlib import Path
import numpy as np
from source.agent.edit_agent import _valid_region,_shape_from_mask,_rasterize,apply_operations
from tools.labelagent_experiments.ground_action_review import gap_polygon,validate_action,action_schema

class GroundActionChecks(unittest.TestCase):
    def test_collinear_repair_is_rejected(self):
        self.assertFalse(_valid_region([[7,2],[7,6],[7,9]],20,20))
        self.assertFalse(_valid_region([[7,2],[7,2],[7,2]],20,20))
    def test_thin_one_pixel_gap_has_positive_bounds(self):
        self.assertTrue(_valid_region(gap_polygon({'bbox':[4,3,5,12]},20,20),20,20))
    def test_real_fill_tool_never_changes_pixels_outside_bound_reviewed_gap(self):
        mask=np.ones((40,40),np.uint8);mask[10:15,12:20]=0;mask[25:29,25:29]=0
        holemask=np.zeros_like(mask);holemask[10:15,12:20]=1
        shape=_shape_from_mask(mask,'floor',1);hole={'id':'fixed','shape_id':1,'bbox':[12,10,20,15],'mask':holemask}
        op={'operation':'fill_gap','shape_ids':[1],'hole_id':'fixed','polygon':gap_polygon(hole,40,40),'reason':'visible floor'}
        result=apply_operations([shape],[op],['floor'],40,40,gap_regions=[hole])
        after=_rasterize(result[0][0],40,40)
        changed=after!=mask
        self.assertEqual(int(changed.sum()),40);self.assertTrue(np.all(holemask[changed]))
        self.assertEqual(int(after[25:29,25:29].sum()),0)
    def test_conflicting_or_incomplete_model_actions_are_not_repaired_by_code(self):
        for value in [{'action':'fill_gap','semantic':'non_ground','reason':'equipment'},
                      {'action':'keep','semantic':'ground','reason':'floor'},
                      {'action':'fill_gap','semantic':'ground','reason':''}]:
            with self.assertRaises(ValueError):validate_action(value)
    def test_actual_decoder_grammar_enforces_action_semantics(self):
        from lmformatenforcer import JsonSchemaParser
        import json
        def accepts(value):
            parser=JsonSchemaParser(action_schema())
            for char in json.dumps(value):
                if char not in parser.get_allowed_characters():return False
                parser=parser.add_character(char)
            return parser.can_end()
        self.assertTrue(accepts({'action':'fill_gap','semantic':'ground','reason':'floor stripe'}))
        self.assertFalse(accepts({'action':'fill_gap','semantic':'non_ground','reason':'equipment'}))

if __name__=='__main__':unittest.main(verbosity=2)

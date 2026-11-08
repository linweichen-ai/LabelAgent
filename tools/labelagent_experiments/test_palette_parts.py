"""Source-only invariants: uniform negatives, real-pixel binding, immutability."""
import unittest
from prepare_palette_parts import np
from balanced_palette_part_points import propose
from native_prompt_guard import evaluate

class SourceParts(unittest.TestCase):
    def test_uniform_controls_cannot_propose_parts(self):
        for value in [127,255]:
            rgb=np.full((50,80,3),value,'uint8');mask=np.ones((50,80),'uint8');rows,_=propose(rgb,mask,mask);self.assertEqual(rows,[])
    def scene(self):
        rgb=np.full((80,120,3),140,'uint8');rgb[18:30,18:100]=[30,50,110];rgb[50:62,18:100]=[90,55,20];stock=np.ones((80,120),'uint8');observed=np.zeros_like(stock);observed[8:72,8:112]=1;return rgb,stock,observed
    def test_all_proposed_points_are_real_stock_pixels(self):
        rgb,stock,observed=self.scene();rows,_=propose(rgb,stock,observed);self.assertTrue(rows)
        for row in rows:
            for p in row['points']+row['negative_hypotheses']:self.assertTrue(stock[p['y'],p['x']] and observed[p['y'],p['x']])
    def test_inputs_not_modified_and_repeated_generation_is_deterministic(self):
        args=self.scene();before=[a.copy() for a in args];first,_=propose(*args);second,_=propose(*args);self.assertEqual(first,second)
        for a,b in zip(args,before):np.testing.assert_array_equal(a,b)
    def test_uniform_unobserved_padding_cannot_supply_parts(self):
        rgb,stock,observed=self.scene();rgb[observed==0]=[255,0,0];rows,_=propose(rgb,stock,observed)
        for r in rows:self.assertTrue(observed[r['point'][1],r['point'][0]])
    def test_original_negative_neighborhood_still_rejects(self):
        m=np.zeros((30,30),'uint8');m[10:14,10:14]=1;copy=m.copy();guard=evaluate(m,[{'x':10,'y':10,'label':1},{'x':16,'y':10,'label':0}]);self.assertFalse(guard['valid']);np.testing.assert_array_equal(m,copy)

if __name__=='__main__':unittest.main()

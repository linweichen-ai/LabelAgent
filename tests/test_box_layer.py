import unittest
import numpy as np
from boxed_capture import validate_box,BoxPredictor
from box_layer_policy import propose


class BoxTests(unittest.TestCase):
    def test_outside_box_native_pixels_remain(self):
        class Model:
            def predict_inst(self,*a,**kw):self.kw=kw;return self.returned
        model=Model();mask=np.zeros((3,20,30),np.float32);mask[:,1,1]=1;mask[:,18,28]=1;model.returned=(mask,np.ones(3),None)
        proxy=BoxPredictor(model,validate_box([5,5,15,15],30,20));result=proxy.predict_inst('state',multimask_output=True)
        self.assertIs(result,model.returned);self.assertEqual(int(result[0].sum()),6);np.testing.assert_array_equal(model.kw['box'],[5,5,15,15])

    def test_bad_boxes_rejected(self):
        for box in [[0,0,30,10],[5,5,4,9],[True,0,5,8],[0,0,float('nan'),8],[0,0,4]]:
            with self.assertRaises(ValueError):validate_box(box,30,20)

    def test_rgb_seam_and_points_without_labels(self):
        rgb=np.zeros((60,90,3),np.uint8);rgb[30:]=220;mask=np.zeros((60,90),np.uint8);mask[10:50,10:80]=1
        original=mask.copy();rows,diag=propose(rgb,mask);self.assertEqual(len(rows),2);self.assertIn(diag['RGB_seam'],[29,30]);np.testing.assert_array_equal(mask,original)
        for r in rows:
            q=r['points'][0];self.assertTrue(mask[q['y'],q['x']]);validate_box(r['diagnostic_box'],90,60)


if __name__=='__main__':unittest.main()

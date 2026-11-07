import io,unittest
import cv2,numpy as np
from PIL import Image
from soft_prior_capture import validate_prior,PriorPredictor

def png(image):
    b=io.BytesIO();image.save(b,'PNG');return b.getvalue()


class PriorTests(unittest.TestCase):
    def setUp(self):
        self.logits=np.linspace(-2,3,20*30,dtype=np.float32).reshape(20,30);self.image=png(Image.new('RGB',(30,20),'gray'));self.prior=png(Image.fromarray((self.logits>=0).astype('uint8')*255))
    def test_same_source_sign_and_geometry_required(self):
        self.assertIs(validate_prior(self.logits,self.image,self.prior),self.logits)
        for a in [self.logits.astype('float64'),self.logits[:10],np.full_like(self.logits,np.nan),-self.logits]:
            with self.assertRaises(ValueError):validate_prior(a,self.image,self.prior)
    def test_hard_input_exact_and_return_object_not_changed(self):
        class Model:
            def predict_inst(self,*a,**kw):self.kw=kw;return self.returned
        model=Model();model.returned=(np.ones((1,20,30)),np.ones(1),np.ones((1,8,8)));hard=cv2.resize(np.where(self.logits>=0,10.,-10.).astype('float32'),(8,8))[None];p=PriorPredictor(model,self.logits,'hard');r=p.predict_inst('state',mask_input=hard,multimask_output=False);self.assertIs(r,model.returned);np.testing.assert_array_equal(model.kw['mask_input'],hard);self.assertFalse(p.calls[0]['multimask_output']);model.returned[2][:]=7;self.assertTrue((p.calls[0]['native_low_res_logits']==1).all())
    def test_soft_prior_preserves_values_without_normalization_or_clipping(self):
        class Model:
            def predict_inst(self,*a,**kw):self.kw=kw;return ('raw',None,None)
        model=Model();hard=np.zeros((1,8,12),np.float32);p=PriorPredictor(model,self.logits,'soft');p.predict_inst('state',mask_input=hard,multimask_output=False);np.testing.assert_array_equal(model.kw['mask_input'],cv2.resize(self.logits,(12,8),interpolation=cv2.INTER_LINEAR)[None]);np.testing.assert_array_equal(hard,np.zeros_like(hard));self.assertGreater(len(np.unique(model.kw['mask_input'])),2)
    def test_missing_or_bad_prior_protocol_rejected(self):
        with self.assertRaises(ValueError):PriorPredictor(object(),self.logits,'other')
        with self.assertRaises(ValueError):PriorPredictor(object(),self.logits,'soft').predict_inst('state')
        with self.assertRaises(ValueError):PriorPredictor(object(),self.logits,'soft').predict_inst('state',mask_input=np.zeros((8,8)))


if __name__=='__main__':unittest.main()

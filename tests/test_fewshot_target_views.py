import unittest
from PIL import Image
from tools.fewshot_target_views import fewshot_target_card

class FewshotChecks(unittest.TestCase):
    def references(self):return [(label,Image.new('RGB',(60,40),(30,40,50))) for label in ('pallet','person','pallet','person')]
    def test_balanced_reference_classes_and_unlabeled_query(self):
        image,meta=fewshot_target_card(Image.new('RGB',(512,512),(200,100,50)),self.references())
        self.assertEqual(image.size,(1024,1024));self.assertFalse(meta['target_class_or_ID_drawn'])
        self.assertEqual([r['label'] for r in meta['reference_examples']],['pallet','person','pallet','person'])
    def test_query_is_retained_and_reference_pixels_do_not_overlay_it(self):
        q=Image.new('RGB',(512,512),(200,100,50));image,meta=fewshot_target_card(q,self.references());refs=[(l,Image.new('RGB',(60,40),(1,2,3))) for l,_ in self.references()];changed,_=fewshot_target_card(q,refs)
        self.assertEqual(image.crop((0,288,1024,1024)).tobytes(),changed.crop((0,288,1024,1024)).tobytes())
        self.assertEqual(meta['query_fit']['display_size'],[736,736]);self.assertEqual(image.getpixel((512,640)),(200,100,50))
    def test_class_imbalanced_examples_rejected(self):
        with self.assertRaises(ValueError):fewshot_target_card(Image.new('RGB',(20,20)),[('pallet',Image.new('RGB',(20,20)))]*4)

if __name__=='__main__':unittest.main()

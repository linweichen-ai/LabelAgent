import hashlib
import unittest
from PIL import Image
from tools.source_companion import unmarked_companion, source_companion_payload
from tools.ascii_target_recheck import ascii_card_payload

class CompanionTests(unittest.TestCase):
    def test_outside_pixels_cannot_enter_query(self):
        source=Image.new('RGB',(30,20),(255,0,0));source.paste((10,60,130),(7,5,15,14))
        changed=Image.new('RGB',source.size,(0,255,255));changed.paste((10,60,130),(7,5,15,14))
        a,ma=unmarked_companion(source,[7,5,15,14]);b,mb=unmarked_companion(changed,[7,5,15,14])
        self.assertEqual(a.tobytes(),b.tobytes());self.assertEqual(ma,mb)
        self.assertEqual(ma['source_size'],[8,9])

    def test_thin_source_preserves_entire_extent(self):
        source=Image.new('RGB',(1,10),(0,0,0));source.putpixel((0,0),(255,0,0));source.putpixel((0,9),(0,0,255))
        image,meta=unmarked_companion(source,[0,0,1,10]);x,y=meta['offset'];w,h=meta['display_size']
        self.assertEqual(image.getpixel((x,y)),(255,0,0));self.assertEqual(image.getpixel((x+w-1,y+h-1)),(0,0,255))

    def test_invalid_bounds_rejected(self):
        for box in ([-1,0,2,2],[0,0,3,2],[1,0,1,2],[0,0,2.0,2]):
            with self.assertRaises(ValueError):unmarked_companion(Image.new('RGB',(2,2)),box)

    def fixture(self):
        a,b=b'clean-jpeg',b'marked-jpeg'
        refs=[dict(path=n,sha256=hashlib.sha256(v).hexdigest()) for n,v in [('a',a),('b',b)]]
        f=dict(model='test',categories=['pallet','person','uncertain'],temperature=0,max_tokens=192,
               prompts={'overview':'neutral target query'},image_token_budgets={'overview':256},guard_image=refs[0],guard_prompt='guard')
        return f,{'a':(a,'image/jpeg'),'b':(b,'image/jpeg')},dict(id='private-id',expected_category='pallet',views={'overview':refs})

    def test_image_order_and_no_truth_leakage(self):
        f,a,c=self.fixture();body,h=source_companion_payload(f,a,c,'overview')
        changed={**c,'id':'different','expected_category':'person','human_category':'person','role':'other'}
        self.assertEqual(source_companion_payload(f,a,changed,'overview')[1],h)
        content=body['messages'][0]['content'];self.assertEqual(len(content),3)
        self.assertTrue(content[1]['image_url']['url'].endswith('Y2xlYW4tanBlZw=='))
        self.assertTrue(content[2]['image_url']['url'].endswith('bWFya2VkLWpwZWc='))

    def test_default_guard_is_identical_single_image(self):
        f,a,c=self.fixture()
        self.assertEqual(source_companion_payload(f,a,c,'guard',True),ascii_card_payload(f,a,c,'guard',True))
        body,_=source_companion_payload(f,a,c,'guard',True)
        self.assertNotIn('image_token_budget',body)
        self.assertEqual(len(body['messages'][0]['content']),2)

if __name__=='__main__':unittest.main()

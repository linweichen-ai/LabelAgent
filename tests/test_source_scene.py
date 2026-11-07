import hashlib,json,unittest
from tools.source_scene import scene_payload,reviewed_category
class SceneTests(unittest.TestCase):
    def data(self):
        assets={'clean':(b'original-source','image/jpeg'),'marked':(b'guides','image/jpeg')}
        refs=[{'path':k,'sha256':hashlib.sha256(v[0]).hexdigest()} for k,v in assets.items()]
        return {'model':'local'},assets,{'id':'PRIVATE','views':{'overview':refs}}
    def test_clean_only_no_markers_or_truth(self):
        f,a,c=self.data();body,h=scene_payload(f,a,c);s=json.dumps(body)
        self.assertEqual(len(body['messages'][0]['content']),2);self.assertNotIn('PRIVATE',s)
        before=scene_payload(f,a,c);c.update(human_category='SECRET',seed=[1,2]);c['views']['overview'][1]['path']='missing';self.assertEqual(before,scene_payload(f,a,c))
        self.assertNotIn('SECRET',json.dumps(scene_payload(f,a,c)[0]))
    def test_single_image_budget_and_no_class_options(self):
        f,a,c=self.data();body,h=scene_payload(f,a,c);self.assertEqual(body['image_token_budget'],256)
        enum=body['response_format']['json_schema']['schema']['properties']['category']['enum'];self.assertNotIn('pallet',enum);self.assertNotIn('person',enum)
    def test_review_preserves_raw_when_scene_present(self):
        for c in ['pallet','person','uncertain']:self.assertEqual(reviewed_category(c,'structured_scene'),c)
    def test_review_abstains_on_flat_or_unclear(self):
        for scene in ['flat_field','uncertain']:
            for raw in ['pallet','person','uncertain']:self.assertEqual(reviewed_category(raw,scene),'uncertain')
        with self.assertRaises(ValueError):reviewed_category('pallet','made_up')
if __name__=='__main__':unittest.main()

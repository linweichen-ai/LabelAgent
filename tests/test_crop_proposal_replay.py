import hashlib,io
from copy import deepcopy
import unittest
from PIL import Image
from tools.replay_crop_proposals import replay_image
from tools.tiled_candidate_utils import tile_grid

class CropReplayTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(40,40),'white')
        self.context={'topic':'/capture/camera_0/image','frame':15,'source_image_sha256':'image','query':'wooden pallet',
            'plan_sha256':'plan','grid_spec':{'divisions':2,'crop_fraction':.6},'internal_edge_margin_pixels':2}
        self.receipts=[]
        for index,bounds in enumerate(tile_grid(40,40)):
            buffer=io.BytesIO();self.image.crop(bounds).save(buffer,format='PNG')
            self.receipts.append({'topic':self.context['topic'],'frame':15,'tile_index':index,'bounds':list(bounds),
                'source_image_sha256':'image','crop_png_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),
                'query':'wooden pallet','plan_sha256':'plan','response':{'results':[]}})

    def test_all_nontruncated_candidates_survive_without_an_early_budget(self):
        def result(left,right):return {'score':.9,'parts':[{'points':[[left,3],[right,3],[right,8],[left,8]],'holes':[]}]}
        self.receipts[0]['response']['results']=[result(3,5),result(8,10),result(13,15),result(21,23)]
        seed=[{'id':'original','label':'person'}];before=deepcopy(self.receipts)
        values,stats=replay_image(seed,self.image,self.receipts,self.context)
        self.assertEqual(values[:1],seed);self.assertEqual(len(values),4)
        self.assertEqual(stats,{'raw_proposals':4,'edge_rejected':1,'nontruncated_proposals':3})
        self.assertEqual(self.receipts,before)

    def test_wrong_prompt_cannot_be_replayed(self):
        self.receipts[1]['query']='pallet'
        with self.assertRaises(ValueError):replay_image([],self.image,self.receipts,self.context)

    def test_missing_crop_response_is_rejected(self):
        with self.assertRaises(ValueError):replay_image([],self.image,self.receipts[:-1],self.context)

if __name__=='__main__':unittest.main()

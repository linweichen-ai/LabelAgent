"""A separate variant keeps natural photo context around tiny targets."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent
src=(helpers/'generate_semantic_candidates.py').read_text(encoding='utf-8')
src=src.replace("out=ROOT/'outputs/labelagent_semantic_recovery_20261105/instance_generation'","out=ROOT/'outputs/labelagent_semantic_recovery_20261105/instance_generation_contextual'")
old="views=[classify_neutral(workflow,config,png(Image.fromarray(im)),out/'visual_cache') for im in [isolated,raw]]"
new="""tight=[max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(1920,int(xx.max())+9),min(1536,int(yy.max())+9)]
    margin=max(32,int(max(xx.max()-xx.min(),yy.max()-yy.min())*.5))
    wider=[max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(1920,int(xx.max())+margin+1),min(1536,int(yy.max())+margin+1)]
    photos=[image.crop(tuple(b)) for b in [tight,wider]]
    for i,photo in enumerate(photos):
     # Pixel interpolation is declared; this adds no source detail. It avoids
     # a very small input silently receiving only the encoder's 64-token min.
     scale=max(1,512/max(photo.size));photos[i]=photo.resize((round(photo.width*scale),round(photo.height*scale)),Image.Resampling.LANCZOS)
    views=[classify_neutral(workflow,config,png(im),out/'visual_cache') for im in photos]"""
assert old in src;src=src.replace(old,new)
src=src.replace("'crop':box","'crop':box,'actual_source_crops':[tight,wider],'interpolation':'Lanczos to minlongedge512; no new source detail'")
src=src.replace('neutral center-object caption on isolated original pixels AND unmarked source crop; target category list withheld from visual encoder','neutral center-object caption on unmarked tight and natural-context photo crops; minlongedge512 interpolation disclosed, no target class list')
(helpers/'generate_contextual_candidates.py').write_text(src,encoding='utf-8')
print('Separate contextual candidate variant prepared')

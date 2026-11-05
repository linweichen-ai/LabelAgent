"""Two source-bound cues: all target pixels retained, no outside-region input."""
import io,copy,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw
INSTANCE_SCHEMA={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':160},'category':{'type':'string','enum':['pallet_base','person','machine','cargo','floor','wall_or_rack','other','uncertain']},'instance_status':{'type':'string','enum':['single','repeated','not_base','uncertain']},'evidence':{'type':'string','minLength':1,'maxLength':120}},'required':['object','category','instance_status','evidence'],'additionalProperties':False}
INSTANCE_PROMPT=('Name the actual visible material and physical object indicated by the pointer or corner locator. '
 'The locator only identifies the target region; it is never evidence that an object exists. '
 'Use the source photo, not the question, to choose category. '
 'For instance_status, single means exactly one distinguishable visible support base with a deck/slats and feet or an identifiable deck under occlusion. '
 'Repeated stacked bases are repeated, an unrelated object is not_base, and an isolated edge/foot without enough identity evidence is uncertain. '
 'Judge only the indicated target, not adjacent or background objects. Do not invent hidden structure. '
 'If the image is uniform or structure is absent, say so. Return JSON keys object, category, instance_status, evidence. '
 'Give concrete visible evidence rather than repeating a category or referring only to the locator.')
def png(im):
 b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
def target_views(source,mask):
 source=source.convert('RGB');mask=np.asarray(mask,dtype=bool)
 if mask.shape!=(source.height,source.width) or not mask.any():raise ValueError('Exact nonempty target required')
 yy,xx=np.where(mask);box=[max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(source.width,int(xx.max())+9),min(source.height,int(yy.max())+9)];photo=source.crop(tuple(box));local=mask[box[1]:box[3],box[0]:box[2]];assert int(local.sum())==int(mask.sum())
 scale=min(496/photo.width,496/photo.height);tw,th=max(1,round(photo.width*scale)),max(1,round(photo.height*scale));ox,oy=(512-tw)//2,(512-th)//2;canvas=Image.new('RGB',(512,512),(127,127,127));canvas.paste(photo.resize((tw,th),Image.Resampling.LANCZOS),(ox,oy));ys,xs=np.where(local);idx=int(np.argmin((xs-xs.mean())**2+(ys-ys.mean())**2));tip=(int(xs[idx]),int(ys[idx]));display=lambda x,y:(round(ox+x*tw/photo.width),round(oy+y*th/photo.height));tx,ty=display(*tip);tail=(max(2,min(509,tx+(110 if tx<256 else -110))),max(2,min(509,ty+(90 if ty<256 else -90))));pointed=np.array(canvas).copy();cv2.arrowedLine(pointed,tail,(tx,ty),(255,150,0),3,tipLength=.1)
 boxed=canvas.copy();draw=ImageDraw.Draw(boxed);left,top=display(int(xs.min()),int(ys.min()));right,bottom=display(int(xs.max())+1,int(ys.max())+1);length=max(5,min(18,(right-left)//4,(bottom-top)//3))
 for x,y,dx,dy in [(left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)]:draw.line([(x+dx*length,y),(x,y),(x,y+dy*length)],fill=(255,0,255),width=3)
 return {'pointer':Image.fromarray(pointed),'corners':boxed},{'source_box':box,'original_target_pixels':int(mask.sum()),'retained_target_pixels':int(local.sum()),'pointer_source_pixel':list(tip),'pointer_in_target':bool(local[tip[1],tip[0]]),'outside_source_pixels_used':False,'semantic_location_verified':False}
def single_physical_base(decision):
 from semantic_subject_guard import compatible_pallet_subject
 return (decision.get('category')=='pallet_base' and decision.get('instance_status')=='single' and compatible_pallet_subject({'object':decision.get('object','')}) and isinstance(decision.get('evidence'),str) and bool(decision['evidence'].strip()))

"""Retrospective 3D-reference cards; never input to proposal generation."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
src=Path(__file__).with_name('diagnose_subset_limit.py').read_text(encoding='utf-8')
src=src.replace("from PIL import Image","from PIL import Image,ImageDraw,ImageFont")
src=src.replace("out=ROOT/'outputs/labelagent_efficiency_20261104/diagnosis/subset_upper_bound.json'","out=ROOT/'outputs/labelagent_recovery_20261105/miss_diagnosis/upper_bound.json'")
src=src.replace("original=ROOT/'outputs/labelagent_hybrid_verified_20261104';rows=[]","original=ROOT/'outputs/labelagent_hybrid_verified_20261104';rows=[]\n folder=ROOT/'outputs/labelagent_recovery_20261105/miss_diagnosis';folder.mkdir(exist_ok=True)\n font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)")
needle="coverage=float(hit[selected].mean());rows.append"
insert="""uv=pixels[selected];x1,y1=uv.min(0);x2,y2=uv.max(0);pad=72;box=(max(0,int(x1)-pad),max(0,int(y1)-pad),min(w,int(x2)+pad+1),min(h,int(y2)+pad+1))
   source_image=Image.open(original/'inputs'/(case+'.png')).convert('RGB');overlay=np.array(source_image).copy()
   cv2.drawContours(overlay,cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(30,220,220),2)
   for x,y in uv:cv2.circle(overlay,(int(x),int(y)),4,(245,60,50),-1)
   canvas=Image.new('RGB',(1600,950),'#f1f5fa');draw=ImageDraw.Draw(canvas)
   draw.text((20,15),case+' / 参考对象 '+str(b['obj_id'])+' / 原始漏检诊断',font=font,fill='#142536')
   canvas.paste(source_image.crop(box).resize((800,760)),(0,70));canvas.paste(Image.fromarray(overlay).crop(box).resize((800,760)),(800,70))
   draw.text((20,850),'左：原图；右：稀疏参考点（红）与原SAM掩码边界（青）。不是二维真值。',font=font,fill='#142536')
   draw.text((20,895),'仅作事后解释；不向候选生成器提供对象位置或参考点。',font=font,fill='#142536')
   canvas.save(folder/(case+'_obj'+str(b['obj_id'])+'.jpg'),quality=94)
   coverage=float(hit[selected].mean());rows.append"""
assert needle in src
src=src.replace(needle,insert)
exec(compile(src,str(Path(__file__)), 'exec'),{'__name__':'__main__','__file__':__file__})

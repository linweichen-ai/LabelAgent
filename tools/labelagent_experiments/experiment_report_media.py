"""Render saved real experiment stages, then fully decode every replay video."""
import textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import save,sha

class Report:
    def __init__(self,out):
        self.out=Path(out)
        if self.out.exists():raise FileExistsError('Preserve completed or interrupted report')
        (self.out/'assets').mkdir(parents=True);self.cards=[];self.font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);self.small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    def canvas(self,title):
        c=Image.new('RGB',(1920,1080),'#edf2f6');ImageDraw.Draw(c).text((28,22),title,font=self.font,fill='#193a50');return c
    def panel(self,c,k,photo,title,notes,grid=False):
        x=28+(k%3)*630;y=90+(k//3)*480 if grid else 110;d=ImageDraw.Draw(c);d.text((x,y),title,font=self.small,fill='#193a50');photo=ImageOps.contain(photo,(595,330 if grid else 520));c.paste(photo,(x,y+42));yy=y+380 if grid else 720
        for note in notes:
            for line in textwrap.wrap(str(note),54 if str(note).isascii() else 29):d.text((x,yy),line,font=self.small,fill='#193a50');yy+=24
    def add(self,group,name,c):
        path=self.out/'assets'/(name+'.jpg');c.save(path,quality=93);self.cards.append((group,path));return path
    def text_card(self,group,name,title,lines):
        c=self.canvas(title);d=ImageDraw.Draw(c)
        for i,line in enumerate(lines):d.text((40,130+i*105),str(line),font=self.font,fill='#193a50')
        return self.add(group,name,c)
    def videos(self,groups):
        result=[]
        for group in groups:
            photos=[p for g,p in self.cards if g==group];name=group+'.mp4'
            if not photos:continue
            with av.open(str(self.out/name),'w') as container:
                stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.options={'crf':'23','preset':'veryfast','threads':'2'}
                for file in photos:
                    arr=np.asarray(Image.open(file).convert('RGB'))
                    for _ in range(40):
                        for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            cap=cv2.VideoCapture(str(self.out/name));count=0
            while True:
                ok,_=cap.read()
                if not ok:break
                count+=1
            cap.release();assert count==40*len(photos);result.append({'path':name,'frames':count,'seconds':count/10,'sha256':sha(self.out/name)})
        save(self.out/'media_verification.json',{'passed':True,'pictures':len(self.cards),'videos':result,'fully_decoded_frames':sum(r['frames'] for r in result),'saved_stage_replay_not_screen_recording':True});return result
    def index(self,title,text,videos):
        (self.out/'index.html').write_text('<meta charset="utf-8"><title>'+title+'</title><h1>'+title+'</h1><p>'+text+'</p><a href="summary.json">实测汇总</a> · <a href="independent_readback.json">独立回读</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="assets/'+p.name+'"></p>' for _,p in self.cards),encoding='utf-8')

def overlay(photo,mask):
    a=np.asarray(photo.convert('RGB')).copy();mask=np.asarray(mask)!=0;assert mask.shape==a.shape[:2];a[mask]=(a[mask]*.47+np.array([25,210,235])*.53).astype('uint8');return Image.fromarray(a)

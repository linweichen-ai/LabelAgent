"""Balanced approved examples above an unlabeled, separately framed query."""
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
from tools.evaluate_target_cards import fit_panel

def fewshot_target_card(query,references):
    if len(references)!=4 or Counter(label for label,_ in references)!={'pallet':2,'person':2}:
        raise ValueError('Exactly two approved examples of each class required')
    canvas=Image.new('RGB',(1024,1024),(127,127,127));draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=18)
    reference_meta=[]
    for n,(label,photo) in enumerate(references):
        panel,fit=fit_panel(photo,236,236);x=n*256+10;canvas.paste(panel,(x,20));draw.text((x,0),'REF '+label,font=font,fill=(255,255,255))
        reference_meta.append(dict(label=label,fit=fit))
    panel,fit=fit_panel(query,1024,736);canvas.paste(panel,(0,288));draw.text((12,264),'QUERY (unlabeled)',font=font,fill=(255,255,255))
    return canvas,dict(query_panel=[0,288,1024,1024],query_fit=fit,reference_examples=reference_meta,
                       target_class_or_ID_drawn=False,query_whole_input_retained=True,reference_and_query_panels_separate=True)

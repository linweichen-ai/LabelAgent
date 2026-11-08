"""Part proposals exclude a dominant color blob; final masks are never edited."""
from palette_part_points import propose as all_components

def propose(rgb,stock,observed,max_components=2):
    _,diag=all_components(rgb,stock,observed,max_components);records=diag.get('component_hypotheses',[]);eligible=[r for r in records if r['area_fraction']<=.4];selected=[]
    for row in sorted(eligible,key=lambda r:(-r['score'],r['point'][1],r['point'][0],r['color'])):
        x,y=row['point']
        if any((x-r['point'][0])**2+(y-r['point'][1])**2<12**2 for r in selected):continue
        selected.append(row)
        if len(selected)>=max_components:break
    output=[]
    for row in selected:
        x,y=row['point'];others=[]
        for other in records:
            nx,ny=other['point'];delta=ny-y
            if other is row or abs(delta)<8 or (nx-x)**2+(ny-y)**2<16**2:continue
            left=max(row['bbox'][0],other['bbox'][0]);right=min(row['bbox'][0]+row['bbox'][2],other['bbox'][0]+other['bbox'][2])
            if right-left<min(row['bbox'][2],other['bbox'][2])*.25:continue
            others.append((abs(delta),-other['area'],ny,nx,other))
        neg=[]
        for sign in [-1,1]:
            options=[r for r in others if (r[4]['point'][1]-y)*sign>0]
            if options:neg.append(min(options,key=lambda r:r[:4])[4])
        output.append(row|{'points':[{'x':x,'y':y,'label':1}],'negative_hypotheses':[{'x':r['point'][0],'y':r['point'][1],'label':0} for r in neg],'negative_component_records':neg,'negative_physical_identity_unconfirmed':True})
    return output,diag|{'selected_components':len(output),'dominant_blob_source_part_max_fraction':.4,'dominant_blob_excluded_source_components':len(records)-len(eligible),'source_only_generation_variant_not_final_acceptance_change':True}

"""Apply frozen sensor-only height proposals to every eligible trained R15 stock PAL."""
import shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_palette_parts import ROOT,BASE
from run_rgb_layers import read,save
from deliver_context_graph import sha
from m2f_query_replay import probability
import unlabelled_height_bands as sensor
OUT=BASE/'labelagent_MF_stock_sensor_parts_20261108'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve sensor-source cohort')
    assert shutil.disk_usage(BASE).free>192*2**20
    grid=BASE/'labelagent_registered_fixed_grid_20261108';g=read(grid/'generation_completed.json');old=read(ROOT/'outputs/labelagent_unlabelled_height_bands_20261108/frozen_plan.json');protected={str(grid/'generation_completed.json'):sha(grid/'generation_completed.json'),str(Path(__file__)):sha(__file__),str(Path(sensor.__file__)):sha(sensor.__file__)};OUT.mkdir();views=[];veto=[]
    for r in g['rows']:
        if r['model']!='R15_PAL_candidate' or r['is_synthetic_control']:continue
        folder=Path(r['folder']);rgb=np.asarray(Image.open(folder/'panoptic.png')).astype('uint32');ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];records=read(folder/'query_class_records.json');raw=np.load(folder/'raw_queries.npz',allow_pickle=False);protected[str(folder/'panoptic.png')]=sha(folder/'panoptic.png');protected[str(folder/'raw_queries.npz')]=sha(folder/'raw_queries.npz');protected[r['input']]=sha(r['input']);protected[r['original_source_image']]=sha(r['original_source_image']);border=lambda m:bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
        # The actual stock winner is recovered from the same frozen logits,
        # rather than inferred from a reference instance or target rectangle.
        from m2f_query_replay import replay
        rp=read(grid/'frozen_plan.json');replayed,segs,_,winner=replay(raw['class_logits'],raw['mask_logits'],ids.shape,rp['postprocessing']);np.testing.assert_array_equal(replayed,ids);assert segs==r['segments_info']
        for rank,s in enumerate(sorted([s for s in segs if s['category_id']==1],key=lambda s:(-s['score'],s['id']))[:3],1):
            stock=ids==s['id'];q=np.unique(winner[stock]);assert len(q)==1;native=probability(raw['mask_logits'],int(q[0]),ids.shape)>=.5
            if border(stock) or border(native):veto.append({'teacher_key':r['teacher_key'],'segment':s['id'],'reason':'unchanged_native_or_stock_input_border'});continue
            key=r['teacher_key']+'_segment'+str(s['id']);prior=OUT/'priors'/(key+'.png');prior.parent.mkdir(exist_ok=True);Image.fromarray(stock.astype('uint8')*255).save(prior);protected[str(prior)]=sha(prior);views.append({'teacher_key':key,'case':r['case'],'source_view':r['source_view'],'source_image':r['original_source_image'],'input':r['input'],'box':r['source_virtual_box'],'prior':str(prior),'model':'R15_PAL_candidate','stock_segment':s,'query':int(q[0]),'source_rank':rank,'original_teacher_key':r['teacher_key'],'no_GT_for_source_selection':True})
    assert len(views)*2<=32
    plan={k:v for k,v in old.items() if k not in ['views','protected']};plan.update(views=views,protected=protected,source='All R15 stock PAL, at most top3 per frozen view, with unchanged native/stock border guards; no reference based ROI selection',source_vetoes=veto,new_MF_calls=0,new_model_calls=0,prior_all_stock_foreground_retained=True,maximum_total_positive_proposals=32,reference_files_opened=0,not_same_input_as_RGB_only_baseline=True);save(OUT/'frozen_plan.json',plan);print({'source_stock_views':len(views),'vetoes':len(veto),'new_model_calls':0},flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:sensor.OUT=OUT;sensor.run()

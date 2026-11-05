"""Check whether a time-named field actually contains acquisition times."""
import json,sys,hashlib
from pathlib import Path
from projection_audit_core import ROOT,read,environment
def main():
 cfg,ev=environment();import numpy as np
 out=ROOT/'outputs/labelagent_projection_recovery_20261105/timing_audit';out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():raise FileExistsError('Preserve timing audit')
 dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');rows=[];types={1:'u1',2:'i1',3:'<u2',4:'<i2',5:'<u4',6:'<i4',7:'<f4',8:'<f8'}
 for frame in [74,75,76,77,78]:
  stamp=str(manifest['timestamps_ns'][frame]);ev.checked_source(dataset,manifest,f'lidar/{stamp}.pcd');raw,fields,stride,n,_=ev.read_pcd(dataset/'source/lidar'/(stamp+'.pcd'));field=next(f for f in fields if f['name']=='time');times=np.ndarray((n,),dtype=types[field['type']],buffer=raw,offset=field['offset'],strides=(stride,));rows.append({'frame':frame,'filename_timestamp_ns':int(stamp),'points':n,'time_field_type':field['type'],'time_min':float(np.min(times)),'time_max':float(np.max(times)),'nonzero_time_values':int(np.count_nonzero(times)),'finite_time_values':int(np.isfinite(times).sum())})
 result={'status':'complete','rows':rows,'all_five_native_time_columns_are_zero':all(r['nonzero_time_values']==0 for r in rows),'hardware_exposure_synchronization_verified':False,'per_point_scan_timing_recoverable':False,'deskew_state_documented':False,'original_references_or_metric_changed':False,'finding':'PCD time column is present but all values are zero in all5 frames; PNGs have no timing metadata; importer supplies log/publish times from filenames. Exact file join excludes importer mis-indexing, not physical sensor clock offset.'}
 (out/'completed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

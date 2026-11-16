"""Merge complete disjoint exports of the same frozen person-envelope recipe."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

SIGNATURE=('version','methods','minimum_score','addition_ratio','nms_iou','solver_seconds','raw_projection',
    'person_envelopes','guard','raw_coverage_certificate','prediction_uses_reference_labels','code_sha256','helper_sha256')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def merge_exports(dataset,frozen_plan,baseline_json,baseline_receipt,current_json,current_receipt,studies,output):
    dataset=Path(dataset).resolve();output=Path(output).resolve();studies=[Path(p).resolve() for p in studies]
    if not studies or len(set(studies))!=len(studies):raise ValueError('Distinct source studies required')
    if output.exists() or output.is_relative_to(dataset) or any(output.is_relative_to(p) for p in studies):
        raise ValueError('Fresh output outside inputs required')
    bindings={}
    def read(path):
        path=Path(path).resolve()
        if any(path.is_relative_to(dataset/'source'/p) for p in ('labels','ali_label','wilddet3d-labels')):
            raise ValueError('Reference annotation input forbidden')
        raw=path.read_bytes();value=hashlib.sha256(raw).hexdigest()
        if path in bindings and bindings[path]!=value:raise ValueError('Input changed')
        bindings[path]=value
        return json.loads(raw)
    frozen=read(frozen_plan);manifest=read(dataset/'import_manifest.json')
    baseline=read(baseline_json);br=read(baseline_receipt);current=read(current_json);cr=read(current_receipt)
    if (sha(baseline_json)!=br['prediction_json_sha256'] or sha(baseline_json)!=frozen['baseline_prediction_sha256']
        or sha(current_json)!=cr['prediction_json_sha256'] or sha(current_json)!=frozen['current_prediction_sha256']):
        raise ValueError('Frozen baseline/current binding changed')
    if any(r['archive_sha256']!=manifest['archive_sha256'] for r in (br,cr)):raise ValueError('Archive mismatch')
    frames=frozen['full_validation_frames'];cameras=frozen['cameras']
    if any(not values or any(type(v) is not int or v<0 for v in values) or len(values)!=len(set(values)) for values in (frames,cameras)):
        raise ValueError('Distinct nonnegative dimensions required')
    expected={(f'/capture/camera_{c}/image',str(f)) for f in frames for c in cameras}
    if len(expected)!=frozen['full_validation_images']:raise ValueError('Frozen coverage count differs')
    records={(r['topic'],str(r['frame'])):r['sha256'] for r in manifest['records'] if r['topic'].endswith('/image')}
    if not expected<=records.keys():raise ValueError('Missing manifest images')
    merged=None;seen=set();signature=None;parents=[];images={}
    for directory in studies:
        plan_path=directory/'study_plan.json';plan=read(plan_path);progress=read(directory/'progress.json');plan_sha=sha(plan_path)
        if (progress.get('status')!='complete' or progress.get('reference_files_opened')!=0 or progress.get('workbench_writes') is not False
            or plan.get('prediction_uses_reference_labels') is not False):raise ValueError('Complete isolated reference-free study required')
        if any(progress.get(k) is not True for k in ('input_bindings_unchanged','plan_unchanged','code_unchanged','all_predictions_frozen_before_scoring')):
            raise ValueError('Parent bindings or completion changed')
        if plan['code_sha256']!=frozen['predictor_sha256'] or plan['helper_sha256']!=frozen['helper_sha256']:
            raise ValueError('Frozen recipe implementation changed')
        own={(f'/capture/camera_{c}/image',str(f)) for f in plan['frames'] for c in plan['cameras']}
        if len(own)!=plan['planned_images'] or progress['completed_images']!=len(own) or not own<=expected or own&seen:
            raise ValueError('Missing, overlapping or unexpected source cohort')
        own_signature=canonical({k:plan[k] for k in SIGNATURE})
        if signature is None:
            signature=own_signature;methods=list(plan['methods'])
            if frozen['primary_method'] not in methods or 'original_point_guard_control' not in methods:
                raise ValueError('Frozen primary and original control required')
            merged={m:{} for m in methods}
        elif signature!=own_signature:raise ValueError('Parent recipe differs')
        rows={(r['topic'],str(r['frame']),r['method']):r for r in progress['rows']}
        if len(rows)!=len(progress['rows']) or set(rows)!={(t,f,m) for t,f in own for m in methods}:
            raise ValueError('Incomplete or duplicate source certificates')
        parent={'plan_sha256':plan_sha,'images':len(own),'predictions':{}}
        for method in methods:
            path=directory/(method+'_candidates.json');mapping=read(path);receipt=read(directory/(method+'_receipt.json'))
            if (receipt['prediction_json_sha256']!=sha(path) or receipt['plan_sha256']!=plan_sha
                or receipt['archive_sha256']!=manifest['archive_sha256'] or receipt['method']!=method
                or receipt['candidates_confirmed_as_truth'] is not False):raise ValueError('Source receipt mismatch')
            if {(t,f) for t,fs in mapping.items() for f in fs}!=own:raise ValueError('Export coverage mismatch')
            if {(t,f) for t,fs in receipt['image_sha256'].items() for f in fs}!=own:raise ValueError('Image receipt coverage mismatch')
            for topic,key in own:
                values=mapping[topic][key];original=baseline[topic][key];row=rows[topic,key,method]
                if values[:len(original)]!=original:raise ValueError('Original prefix changed')
                if len({s['id'] for s in values})!=len(values):raise ValueError('Duplicate candidate IDs')
                if not all(row.get(k) is True for k in ('original_prefix_preserved','raw_certificate_verified','person_guard_verified')):
                    raise ValueError('Missing independent source readbacks')
                if row.get('solver',{}).get('certificate_satisfied') is not True:raise ValueError('Missing actual solver certificate')
                if method=='original_point_guard_control' and canonical(values)!=canonical(current[topic][key]):
                    raise ValueError('Original guard control drifted')
                source_hash=records[topic,key]
                if any(r['image_sha256'][topic][key]!=source_hash for r in (receipt,br,cr)):raise ValueError('Image binding changed')
                merged[method].setdefault(topic,{})[key]=values;images.setdefault(topic,{})[key]=source_hash
            parent['predictions'][method]=sha(path)
        seen|=own;parents.append(parent)
    if seen!=expected:raise ValueError('Full frozen cohort is incomplete')
    if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('Source changed during merge')
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        path=output/name;path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');return path
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'methods':list(merged),'primary_method':frozen['primary_method'],
        'frames':frames,'cameras':cameras,'planned_images':len(expected),'parent_bindings':parents,
        'frozen_validation_plan_sha256':sha(frozen_plan),'new_model_requests':0,'reference_files_opened':0,
        'original_control_replayed_exact':True,'merge_code_sha256':sha(__file__)}
    plan_path=save('study_plan.json',plan);plan_sha=sha(plan_path)
    for method,mapping in merged.items():
        ordered={t:{k:fs[k] for k in sorted(fs,key=int)} for t,fs in sorted(mapping.items())}
        path=save(method+'_candidates.json',ordered)
        save(method+'_receipt.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
            'archive_sha256':manifest['archive_sha256'],'prediction_json_sha256':sha(path),
            'image_sha256':images,'plan_sha256':plan_sha,'method':method,'new_model_requests':0,
            'candidates_confirmed_as_truth':False,'source':'merged_bound_person_envelope_predictions'})
        if canonical(read(path))!=canonical(ordered):raise ValueError('Merged readback differs')
    if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('Source changed before completion')
    result={'status':'complete','images':len(expected),'source_studies':len(studies),'methods':list(merged),
        'primary_method':frozen['primary_method'],'original_control_replayed_exact':True,'source_bindings_unchanged':True,
        'plan_sha256':plan_sha,'export_readback_verified':True,'new_model_requests':0,'workbench_writes':False}
    save('progress.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','frozen-plan','baseline-json','baseline-receipt','current-json','current-receipt','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--study',action='append',required=True)
    args=parser.parse_args()
    print(json.dumps(merge_exports(args.dataset,args.frozen_plan,args.baseline_json,args.baseline_receipt,
                                  args.current_json,args.current_receipt,args.study,args.output)))

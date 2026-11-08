"""Review all sensor-prompt native-valid outputs under the unchanged RGB gates."""
import shutil,sys
from pathlib import Path
from run_MF_stock_sensor_SAM import OUT
from prepare_palette_parts import BASE
from run_rgb_layers import read,save
from deliver_context_graph import sha
import MF_first_semantics as prepare

def main():
    prepare.OUT=OUT;prepare.S=OUT/'semantic'
    if sys.argv[1]!='prepare':prepare.runner.OUT=OUT;prepare.runner.S=prepare.S;prepare.runner.run();return
    prepare.prepare();s=prepare.S;p=read(s/'frozen_plan.json');wanted={r['request_sha256']:r for r in p['requests']};cache=read(s/'exact_prior_cache.json');seen={r['request_sha256'] for r in cache['rows']}
    for relative in ['labelagent_MF_first_parts_20261108/semantic','labelagent_MF_first_parts_20261108/native_pool/semantic']:
        path=BASE/relative/'frozen_plan.json';old=read(path);assert old['model_config_sha256']==p['model_config_sha256']
        for row in old['requests']:
            key=row['request_sha256'];response=path.parent/'responses'/(key+'.json')
            if key in seen or key not in wanted or row['payload']!=wanted[key]['payload'] or not response.exists():continue
            reply=read(response);stats=reply['local_vlm'];assert stats['model_config_sha256']==p['model_config_sha256'] and stats['actual_image_token_counts']==[256]*len(row['images']);dest=s/'responses'/response.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(response,dest);assert sha(dest)==sha(response);seen.add(key);cache['rows'].append({'request_sha256':key,'source_plan':str(path),'source_plan_sha256':sha(path),'source_response':str(response),'source_response_sha256':sha(response),'exact_payload_model_and_actual_tokens':True})
    cache['unique_exact_prior_reuses']=len(seen);save(s/'exact_prior_cache.json',cache);p['protected'][str(Path(__file__))]=sha(__file__);save(s/'frozen_plan.json',p);print({'selected_before_reference':len(p['selected']),'exact_prior_reuses':len(seen),'new_model_calls':0},flush=True)

if __name__=='__main__':main()

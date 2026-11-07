"""Reuse exact parent control captions; classify only new geometric inputs."""
import sys
from pathlib import Path
import review_soft_prior_controls as base
from run_prior_strength_control import OUT
PARENT=OUT.parent;base.OUT=OUT;base.D=OUT/'semantic_controls'


def prepare():
    base.prepare();file=base.D/'frozen_plan.json';p=base.read(file);parent=base.read(PARENT/'semantic_controls/completed.json');cache={r['request_sha256']:r for r in parent['rows']}
    for r in p['requests']:
        if r['request_sha256'] in cache:
            old=cache[r['request_sha256']];response=base.read(old['response']);assert response['local_vlm']['model_config_sha256']==p['model_config_sha256'];r.update(reuse_response=old['response'],reuse_response_sha256=base.sha(old['response']));p['protected'][old['response']]=base.sha(old['response'])
    p['actual_wrapper_sha256']=base.sha(__file__);p['max_new_requests']=len({r['request_sha256'] for r in p['requests'] if 'reuse_response' not in r});base.save(file,p);print({'max_fresh_strength_control_captions':p['max_new_requests']},flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        p=base.read(base.D/'frozen_plan.json');assert base.sha(__file__)==p['actual_wrapper_sha256'];base.run()

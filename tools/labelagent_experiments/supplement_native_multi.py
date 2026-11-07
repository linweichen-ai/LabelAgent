"""Distinguish pooled diagnostics from individually feasible unchanged pairs."""
import shutil
from pathlib import Path
from run_native_multi import ROOT,OUT,read,save,sha

audit=read(OUT/'support_area_audit.json');summary=read(OUT/'final_summary.json');proof=[]
for case in audit['cases']:
    r=next(x for x in case['radii'] if x['radius']==3)
    for candidate in audit['candidates']:
        if candidate['case']!=case['case'] or not(candidate['valid'] and candidate['area_fits'] and not candidate['person_conflict']):continue
        own=next(x for x in r['new_mask_ownership'] if x['key']==candidate['key'])['ownership']
        for target in r['remaining_targets']:
            found=next((x for x in own if x['object_id']==target['object_id']),None)
            # No old edge reaches threshold for this row, so keeping every
            # original matched pair and assigning this new column is feasible.
            if target['old_best_support']<.5 and found and found['qualified_support']>=.5:
                assert audit['base_union']+candidate['extra_pixels']<=audit['cap']
                proof.append({'case':case['case'],'candidate':candidate['key'],'object_id':target['object_id'],'candidate_alone_added':True,'all_old_matched_pairs_can_be_kept_exact':True,'old_matched_count':r['old_matches'],'feasible_matched_count':r['old_matches']+1,'extra_pixels':candidate['extra_pixels'],'qualified_support':found['qualified_support'],'semantic_neutral_passed':False})
save(OUT/'individual_budget_matching_proof.json',{'source_audit_sha256':sha(OUT/'support_area_audit.json'),'pool_gated_counts_are_diagnostic_not_joint_budget_admission':True,'proof':proof,'adopted':False})
summary['individual_budget_matching_proof']=proof
summary['pool_counts_are_not_joint_budget_admission']=True
save(OUT/'final_summary.json',summary)
repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
public=read(repo/'docs/test-results/native-multi.json');public.update(individual_budget_matching_proof=proof,pool_counts_are_not_joint_budget_admission=True);save(repo/'docs/test-results/native-multi.json',public)
shutil.copy2(__file__,repo/'tools/labelagent_experiments/supplement_native_multi.py')
with (OUT/'本轮结论.txt').open('a',encoding='utf-8') as f:f.write('\n面积/匹配补充：候选池按单个面积筛过不等于整个池满足共同预算。individual_budget_matching_proof.json单独证明上述两个候选各自新增73/58、各自可保留全部原匹配再增加目标42；两份都语义失败，不接纳。\n')
shutil.copy2(OUT/'本轮结论.txt',repo/'docs/test-results/native-multi.md')
print({'individually_budget_feasible_new_matches':len(proof),'adopted':False},flush=True)

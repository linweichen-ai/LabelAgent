"""Complete D evidence and verified C media entry after C reserve interruptions."""
import os,shutil,stat,sys
from pathlib import Path
import deliver_anchor_gap as base
from run_rgb_layers import ROOT,read,save,sha
C_DISPLAY=base.TARGET
FULL=base.r.BASE/'LabelAgent_完整交接_锚点间隙与覆盖缺口_20261108'
def prepare():
    base.TARGET=FULL;base.prepare();manifest=read(FULL/'complete_evidence_manifest.json')['files'];checked=[]
    assert C_DISPLAY.resolve()==(ROOT/'deliverables/LabelAgent_实时交接/09_补充成果/跨裁剪锚点与新实例间隙及覆盖缺口_20261108').resolve()
    for file in C_DISPLAY.rglob('*'):
        if not file.is_file():continue
        relative=file.relative_to(C_DISPLAY).as_posix();assert relative in manifest and sha(file)==manifest[relative]['sha256'] and sha(FULL/relative)==manifest[relative]['sha256']
        if file.relative_to(C_DISPLAY).parts[0]=='report':continue
        file.resolve().relative_to(C_DISPLAY.resolve());checked.append({'path':str(file),'relative_path':relative,'verified_complete_D_copy':str(FULL/relative),'sha256':manifest[relative]['sha256'],'bytes':file.stat().st_size})
    record={'reason':'Two C copies stopped at original96MiB reserve; every partial file is already byte-exact in complete independent D packet','all_targets_are_owned_unsealed_partial_copies_not_original_outputs_or_annotations':True,'files':checked,'all_original_inputs_and_model_outputs_still_retained':True,'before_C_free':shutil.disk_usage('C:/').free}
    save(base.C/'anchor_gap_partial_C_retirement_plan_20261108.json',record);save(FULL/'engineering_failures/verified_C_partial_retirement_plan.json',record)
    for row in checked:
        file=Path(row['path']);file.resolve().relative_to(C_DISPLAY.resolve());assert sha(file)==row['sha256'] and sha(row['verified_complete_D_copy'])==row['sha256']
        try:file.unlink()
        except PermissionError:file.chmod(stat.S_IREAD|stat.S_IWRITE);file.unlink()
    for folder in sorted((p for p in C_DISPLAY.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        folder.resolve().relative_to(C_DISPLAY.resolve())
        if not any(folder.iterdir()):folder.rmdir()
    for name in ['index.html','summary.json','media_verification.json','本轮结论.txt']:shutil.copy2(FULL/name,C_DISPLAY/name)
    html=(C_DISPLAY/'index.html').read_text('utf-8');html=html.replace('</h1>','</h1><p style="padding:16px;background:#fff1b8">C盘保留全部27图4视频；完整原始证据另存D盘。请同时备份 <a href="完整证据备份位置.txt">完整证据目录</a>。</p>',1);(C_DISPLAY/'index.html').write_text(html,encoding='utf-8');(C_DISPLAY/'完整证据备份位置.txt').write_text('C盘空间波动，两次复制触发原96MiB余量保护。完整普通文件证据已在D盘逐文件核验：\n'+str(FULL)+'\n请整体备份上述D盘目录。C盘本目录只包含全部27张图、4段回放及摘要，不包含本轮全部原答/数组/原参考副本。原始输出和原标注未删除或修改；回收的仅是已经在D盘完整核验过的未封存C复制副本。\n',encoding='utf-8');receipt=read(base.C/'anchor_gap_bundle_verification.json');receipt.update(C_media_entry=str(C_DISPLAY),C_media_complete_private_evidence_on_D=True,C_copy_is_display_only=True,C_partial_copies_retired_after_full_D_hash_readback=True);save(base.C/'anchor_gap_bundle_verification.json',receipt);save(C_DISPLAY/'完整证据位置与核验.json',receipt);save(C_DISPLAY/'交接SHA256.json',{'files':[{'path':p.relative_to(C_DISPLAY).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in C_DISPLAY.rglob('*') if p.is_file()]});record['after_C_free']=shutil.disk_usage('C:/').free;record['retirement_complete']=True;save(base.C/'anchor_gap_partial_C_retirement_completed_20261108.json',record);save(FULL/'engineering_failures/verified_C_partial_retirement_completed.json',record);save(FULL/'交接SHA256.json',{'files':[{'path':p.relative_to(FULL).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in FULL.rglob('*') if p.is_file() and p.name!='交接SHA256.json']})
    note=base.HOME/'00_本轮完整证据备份位置.txt';old=note.read_text('utf-8') if note.exists() else '';note.write_text('新增本轮27张图、4段回放：完整原始证据目录另在\n'+str(FULL)+'\n请同时备份此D盘整个目录，C盘交接仅含本轮可播放报告与摘要。\n\n此前仍需保留的证据位置：\n'+old,encoding='utf-8');print({'D_complete_members':receipt['evidence_members_verified'],'C_display_pictures':27,'C_display_videos':4,'C_free':record['after_C_free']},flush=True)
if __name__=='__main__':
    base.TARGET=FULL
    {'prepare':prepare,'publish':base.publish,'record':base.record}[sys.argv[1]]()

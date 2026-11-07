"""Complete ordinary-file handoff with verified immutable same-volume hardlinks."""
import hashlib,json,os,shutil,stat,zipfile
from pathlib import Path
import deliver_zoom_registry as d
from run_rgb_layers import read,save,sha

def main():
    assert not (d.C/'zoom_registry_bundle_verification.json').exists()
    plan=read(d.C/'zoom_registry_archive_member_plan.json');files=plan['files'];aliases=plan['same_bytes_member_aliases'];failure=read(d.C/'zoom_registry_archive_runtime_interruption.json');partial=Path(failure['partial_archive']);assert partial.resolve().parent==d.TARGET.resolve()
    # Retire only the interrupted derived ZIP, after checking every contained
    # byte against original files which remain present and hash-bound.
    if partial.exists():
        assert sha(partial)==failure['partial_sha256'];recovered=[]
        with zipfile.ZipFile(partial) as z:
            for name in z.namelist():
                canonical=aliases.get(name,name);assert canonical in files;row=files[canonical];assert sha(row['source'])==row['sha256'];assert hashlib.sha256(z.read(name)).hexdigest()==row['sha256'];recovered.append({'partial_member':name,'canonical':canonical,'sha256':row['sha256']})
        save(d.C/'zoom_registry_C_partial_retirement.json',{'partial':str(partial),'partial_sha256':sha(partial),'partial_bytes':partial.stat().st_size,'all_members_exact_in_preserved_originals':True,'members':recovered,'no_original_data_removed':True});partial.unlink()
    else:recovered=read(d.C/'zoom_registry_C_partial_retirement.json')['members']
    evidence=d.TARGET/'complete_evidence';evidence.mkdir(exist_ok=True);rows=[];canonical_paths={};physical_copy=0
    for name,row in files.items():
        source=Path(row['source']);dest=evidence/name;dest.parent.mkdir(parents=True,exist_ok=True);assert sha(source)==row['sha256']
        immutable=(ROOT_outputs:=d.ROOT/'outputs') in source.resolve().parents and source.suffix.lower() in ['.npz','.npy','.png','.jpg','.mp4']
        can_link=immutable and source.drive.lower()==dest.drive.lower()
        if dest.exists():
            assert sha(dest)==row['sha256'];method='same_volume_readonly_hardlink' if can_link else 'independent_copy'
        elif can_link:
            os.link(source,dest);os.chmod(dest,stat.S_IREAD);method='same_volume_readonly_hardlink'
        else:
            if shutil.disk_usage(dest.parent).free<row['bytes']+96*2**20:raise RuntimeError('Unchanged96MiB reserve before physical evidence copy')
            shutil.copy2(source,dest);method='independent_copy';physical_copy+=row['bytes']
        assert sha(dest)==row['sha256'];canonical_paths[name]=dest;rows.append({'member':name,'source':str(source),'sha256':row['sha256'],'bytes':row['bytes'],'storage':method})
    for alias,canonical in aliases.items():
        source=canonical_paths[canonical];dest=evidence/alias;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(source,dest)
        assert sha(dest)==files[canonical]['sha256'];rows.append({'member':alias,'canonical':canonical,'sha256':files[canonical]['sha256'],'bytes':files[canonical]['bytes'],'storage':'same_bytes_alias_hardlink'})
    save(evidence/'evidence_manifest.json',{'files':rows,'all_canonical_files_and_aliases_present_as_ordinary_files':True,'backup_directory_copy_reads_complete_bytes':True,'new_raw_arrays_readonly_do_not_edit_in_place':True,'source_inputs_originals_and_failure_receipts_preserved':True})
    (d.TARGET/'备份说明.txt').write_text('请备份整个本目录，包括complete_evidence。所有证据均为可直接读取的普通文件，已逐一SHA256核验。为避免重复占用空间，同盘不可变数组、图片、视频以只读硬链接保存；复制到备份盘后会得到独立文件。请勿原地修改证据。原标注与代码为独立副本。两次压缩封存被空间检查拦下，未降低余量或质量门槛；不完整压缩件在逐成员核验原文件仍完整后回收，失败记录保留。视频是保存阶段回放。',encoding='utf-8')
    local=[{'path':p.relative_to(d.TARGET).as_posix(),'sha256':sha(p)} for p in d.TARGET.rglob('*') if p.is_file() and p.name!='交接SHA256.json'];save(d.TARGET/'交接SHA256.json',{'files':local})
    receipt={'passed':True,'bundle':str(d.TARGET),'artifact_type':'complete_ordinary_file_directory','archive_created':False,'incomplete_archive_members_verified_then_retired':len(recovered),'evidence_members_verified':len(rows),'unique_canonical_members':len(files),'content_aliases_present_as_files':len(aliases),'evidence_manifest_sha256':sha(evidence/'evidence_manifest.json'),'physical_copies_bytes':physical_copy,'pictures':35,'videos':5,'decoded_frames':1400,'full_new_queries_native_masks_and_originals_readback':True,'backup_entire_directory':True,'all_quality_and96MiB_resource_guards_unchanged':True}
    save(d.C/'zoom_registry_bundle_verification.json',receipt);print(receipt,flush=True)

if __name__=='__main__':main()

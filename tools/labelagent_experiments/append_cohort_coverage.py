"""Append newly verified source-cohort coverage witness to the real report."""
from PIL import ImageFont
from run_rgb_layers import read,save
from experiment_report_media import Report
from finish_anchor_gap_report import OUT
def main():
    evidence=read(OUT/'source_cohort_coverage.json');assert evidence['camera1_left_two_targets_not_covered_by_recent_six_teacher_protocol']
    if (OUT/'assets/source_cohort_coverage.jpg').exists():raise FileExistsError('Preserve appended coverage witness')
    report=Report.__new__(Report);report.out=OUT;report.font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);report.small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);report.cards=[]
    for p in sorted((OUT/'assets').glob('*.jpg')):
        group='anchors' if p.name.startswith('anchor_') else 'gaps' if p.name.startswith(('gap_source_','gap_native_')) else 'review' if p.name.startswith(('gap_semantic_','structure_layout_')) else 'ledger';report.cards.append((group,p))
    report.text_card('ledger','source_cohort_coverage','覆盖缺口已确认：最近六源框不能评估camera1左侧两个遗漏',['这只针对最近固定六源框及其锚点/间隙派生分支，不否定原整帧基线','camera1唯一源框为[928,359,1246,606]，位于画面中部','camera1参考obj47：11个可见点进入该源框0个','camera1参考obj49：6个可见点进入该源框0个','这两个遗漏没有进入该分支的源候选生成，不能据此断言模型漏检能力','后续改用两相机整帧及固定四象限；明确真实上下文和边界留白','参考坐标仅作本次事后覆盖诊断，不用于新生成的框、点或先验'])
    oldmedia=read(OUT/'media_verification.json');new=report.videos(['ledger']);videos=[v for v in oldmedia['videos'] if v['path']!='ledger.mp4']+new;media={'passed':True,'pictures':len(report.cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True};save(OUT/'media_verification.json',media);summary=read(OUT/'summary.json');summary.update(pictures=len(report.cards),videos=len(videos),recent_fixed_ROI_coverage_hole_verified=True,recent_camera1_source_ROIs_cover_neither_left_target=True,coverage_witness_not_generation_reference_input=True);save(OUT/'summary.json',summary);text=(OUT/'本轮结论.txt').read_text('utf-8')+' 新增覆盖审计确认：最近六源框协议的camera1唯一源框位于中部，左侧obj47/49的11/6个可见参考点都不在该框内，因此该局部分支的结果不能用于推断模型对它们的检测能力。后续用两相机整帧和固定四象限，不按参考坐标选框或选点。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');report.index('跨裁剪锚点、源层次新实例与结构复核诊断',text,videos);html=(OUT/'index.html').read_text('utf-8').replace('</h1>','</h1><p><a href="source_cohort_coverage.json">新增：固定源框覆盖缺口核验</a> · <a href="four_remaining_case_ledger.json">剩余四例逐项诊断</a></p>',1);(OUT/'index.html').write_text(html,encoding='utf-8');print({'pictures':len(report.cards),'videos':len(videos),'decoded_frames':media['fully_decoded_frames']},flush=True)
if __name__=='__main__':main()

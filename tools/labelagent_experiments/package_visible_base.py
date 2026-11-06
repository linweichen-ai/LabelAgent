"""Freeze all true, failed, reused and unexecuted evidence without conflation."""
from pathlib import Path
p=Path(__file__).with_name('package_instance_separation.py');s=p.read_text(encoding='utf-8').replace('labelagent_instance_separation_20261105','labelagent_visible_base_20261107').replace('遮挡粘连实例分离_20261105','完整可见基座与原面积预算_20261107').replace('instance_separation_ui_verification','visible_base_ui_verification').replace('instance_separation_regression','visible_base_regression').replace('instance_separation_bundle_verification','visible_base_bundle_verification').replace("['masks']==37","['masks']==52").replace('106张','85张')
start=s.index("notes=f'''遮挡粘连实例分离");end=s.index("(home/'交接说明.txt')",start)
s=s[:start]+'''notes=f\'''完整可见基座与原面积预算：本轮交接

先打开01_主报告与全部试跑/index.html：{s['pictures']}张浏览图片、5段1080p视频，{s['fully_decoded_video_frames']}帧全部解码核验。视频为实际保存阶段回放，原Agent阶段明确标为复用，不是本轮新推理或连续录屏。
仍13/17、490/579点，原6/6结果不变，4漏检未恢复。完整8289像素的5/6纯度候选只有一个连通部分，新增3132在主体内，原余量209.25，不能靠删零散部分过关。新完整外形覆盖目标13点却覆盖邻实例50点，纯度0.206，未过0.67；邻实例原本已匹配，不能当新实例恢复。
16次真实VLM结构规划全部坐标绑定失败；坐标输入修正仅准备，因共享模型忙未取得回答，未当作实际推理。另有8次真实同权重CPU前向、9条新完整查询、8次SAM文字请求的空返回；本轮只新增2次预算内语义请求，无新增Agent调用。

02为本轮冻结比较与复用身份收据；03保留原六组真实结果。全部7类初始掩码逐项一致时才复用六组原结果，不宣称新测得6/6。三轮、每轮3项、累计32点和原面积/数量不变，原27托盘及3人员掩码保持，原标注写入0、训练更新0。
04含四相机五邻帧PNG、PCD、原3D标签、标定、位姿及原人工参考的独立副本；07是同一真实Mask2Former独立权重包，SHA256 {expected}。08补全复用查询和旧SAM的原响应。05含隔离源码、实际工具与配置，恢复时需按路径重绑；完整MCAP、SAM3/Qwen权重及系统依赖未重复复制，不宣称离线一键重跑完整环境。
06含93项工程检查、52条完整候选的逐像素证明、原始文件保护与浏览器核验；输入规划的未生效输出上限声明和共享服务繁忙失败均保留，不作为成功验证。

已冻结的结果与源码使用同盘硬链接，备份时按内容正常复制即可；归档内不要原地修改，请另存。原始数据和Mask2Former权重为独立复制。交接SHA256.json逐文件回读一致。公共Git只有代码及去标识化汇总，真实媒体、标签和权重只留本地。
\''';'''+s[end:]
insert=""" reuse=home/'08_复用查询与旧SAM原响应';reuse.mkdir()
 for source,label in [(ROOT/'outputs/labelagent_instance_separation_20261105/query_generation','原30查询'),(ROOT/'outputs/labelagent_instance_separation_20261105/seam_generation','原7SAM及失败请求'),(ROOT/'outputs/labelagent_target_identity_20261106/seam_generation','上轮6SAM及失败请求')]:shutil.copytree(source,reuse/label,copy_function=frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
"""
s=s.replace(" files=[{'path':p.relative_to(home).as_posix()",insert+" files=[{'path':p.relative_to(home).as_posix()")
exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

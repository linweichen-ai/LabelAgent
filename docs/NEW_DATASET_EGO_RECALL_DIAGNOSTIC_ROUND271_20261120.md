# Round 271：11 月 4 日新数据实测与 ego 召回诊断

日期：2026-11-20
状态：新数据隔离推理和候选诊断完成；没有修改生产推理、人工真值、训练集或模型。

## 1. 新数据主流程是否能跑通

使用批次 B 的两个问题目录各一张 640×512 图像，在隔离工作区执行正式 LabelAgent 流程，类别为车辆、pallet、person、floor、slope、ego、other_robot、cargo_floor，最多一轮 Agent。所有结果保持机器候选态。

| 图像 | 候选 | 实测耗时 | 模型调用 | 人工复核队列 |
|---|---:|---:|---:|---:|
| 003 | 16：floor 1、pallet 13、person 2 | 44.885 秒 | 10 | 2 |
| 004 | 13：floor 1、pallet 12 | 25.478 秒 | 7 | 3 |

VLM 请求统计新增 6 次 miss/store，busy rejection、coalesce timeout 和 coalesce failure 均为 0。候选审核复用分别为 7.818 ms 和 5.796 ms，复用前后 Agent 输出与模型调用记录一致。隔离工作区运行后自动移除，正式 CVAT Task3 状态没有被替换。

VLM 在两张图中都识别到可见 ego，但 SAM3 没有生成 ego 候选；系统按 `vlm_visible_label_missing_candidate` 进入复核队列，没有把漏检伪装成完整结果。

## 2. 为什么不是换一个文本就能解决

对两张图使用一次图像特征提取，并并行检查七个不依赖颜色的文本：`ego robot`、`forklift`、`forklift body`、`forklift chassis`、`robot chassis`、`forklift hood`、`camera robot`。十四个图文组合全部返回 0 个候选。

这说明当前问题不是“橙色提示”或某个英文单词写错，而是 SAM3 文本分支对画面底部被裁切、只露出局部的自车结构召回不足。

## 3. VLM 坐标提示为什么没有直接上线

第一版让 VLM输出 ego 正负点：004 的单点能够得到完整底盘，但 003 只分出控制面板小块；第二版要求多个分散正点时，两图又都错误判断 ego 不存在。该接口在相同语义、不同画面下不稳定，因此没有部署。

## 4. 固定底部点扫描的结果

诊断工具对每张图使用 18 个固定底部点，一次编码图像、批量执行点分割，再由纯几何门禁保留两种形态：

- `wide_bottom_chassis`：宽、贴底、较扁的前向底盘；
- `centered_fork_carriage`：居中、贴底、较高的后向货叉架。

第一轮错误地把四张 Capture 图像当作负例；查看原图后确认三张 camera_3 实际含居中货叉架，一张 camera_1 含被人严重遮挡的右侧车体，因此撤回“6/6 正负验证”的旧说法。没有用错误标签调宽门槛。

修正后的第二轮包含：

- 5 张 assistant-visible 正例：003、004 和 camera_3 连续三帧；
- 4 张明确裁去底部区域并恢复原尺寸的合成负例，仅用于几何门禁验证；
- camera_1 严重遮挡侧车体不进入自动正例，继续保持 unresolved/human review。

结果为 9/9：五张正例均生成位置正确的自车掩膜，四张合成负例均无候选。640×512 两张图端到端诊断耗时为 0.547–0.609 秒；SAM 服务内部单次批处理约 0.30–0.32 秒。拼图人工复核确认没有把地面、人员或托盘误画成 ego。

该结果仍不是生产泛化证据：负例是合成裁切图，正例数量小，且没有人工像素真值或独立未曝光数据。因此本轮只提交可复现诊断工具和门禁测试，不接入正式 `/api/run`。

## 5. 测试与运行恢复

- 新增几何门禁与固定点测试：6 项通过；
- 正式 `PYTHONUTF8=1 pytest -q tests`：1,050 项通过，另有 186 个子测试通过；
- 初次全量测试在导入 PyTorch 时因 Windows 分页文件不足失败，未误报为代码失败；
- 在确认工作台无在途任务后，仅短暂停止本机页面和隧道，远端模型保持热启动，释放提交空间后完成全量测试；
- 测试后工作台和隧道全部恢复，首页 HTTP 200、状态 idle、主 SAM3、5090 SAM 和主 Qwen VLM 正常；可选 5090 VLM 仍未启用。

本轮额外对一份 58.08 MiB 的昨日豆包日志启用可逆 NTFS 压缩，物理释放约 39.02 MiB；没有删除或移动文件。C 盘最终仍只有约 0.25 GiB，属于持续风险，正式批量运行仍须保留保存空间门禁。

## 6. 证据与边界

本机证据：

- `artifacts/new-dataset-smoke-round271/summary.json`
- `artifacts/ego-sam-prompt-round271/results.json`
- `artifacts/ego-sam-point-round271/results.json`
- `artifacts/ego-bottom-point-round271-v2/results.json`
- `artifacts/ego-bottom-point-round271-v2/contact-sheet.jpg`
- `artifacts/ego-bottom-point-round271-negative-crops/synthetic-negative-contact-sheet.jpg`

这些产物含厂内图像或本机绝对路径，只保留在本机 D 盘联接的实验目录，不随源码上传。源码仓库只提交诊断工具、确定性测试、说明和无图结构化证据。

隔离边界保持不变：`candidate_only=true`、`human_truth=false`、`approved_for_training=false`、`production_deployed=false`。本轮没有启动 SAM3 微调、后训练或全量新数据推理。

## 7. 下一步

在接入生产前补充真实无 ego 负例、侧向/遮挡 ego 正例和不同相机姿态；如果独立门禁仍通过，再把点扫描限制为“VLM 已发现 ego、文本 SAM 无候选”时的候选补救，并继续经过类别盲审与人工复核，不自动写成真值。

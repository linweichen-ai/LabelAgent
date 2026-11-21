# Round389：ego 逐像素人工真值验收门禁（2026-11-22）

## 结论

Round388 已证明四路影子评测可以安全收集诊断，但尚没有逐像素人工真值，不能把目视复核当作
精度结论。本轮新增只读验收工具 `tools/evaluate_ego_shadow_pixel_truth.py`，把下一步需要的人工
验收变成可复现、不可由助手或机器结果冒充的门禁。

本轮没有创建人工真值，也没有宣称 ego 质量门禁已经通过；只是把合规输入、指标和拒绝条件固定下来。

## 输入契约

工具只接受 validation/test 划分，并要求：

- `truth_provenance.kind=human_pixel_truth`；
- `human_confirmed=true`，同时 `assistant_generated=false`、`machine_generated=false`；
- 人工确认收据文件真实存在，收据 SHA-256 与清单一致；
- 收据逐样本绑定 ego、person、floor 三类真值掩膜 SHA-256；
- 候选与三类真值都是同尺寸、非空且严格二值的本地掩膜；
- 所有路径必须位于清单目录内，拒绝绝对路径和目录逃逸。

因此，仅复制一个“人工确认”字段、伪造一个 64 位字符串、修改收据或替换任一真值掩膜，都不能通过。

## 指标与门禁

逐样本输出：

- ego IoU；
- ego precision 与 recall；
- 候选落入 person 真值的比例；
- 候选落入 floor 真值的比例；
- 候选及三类真值掩膜 SHA-256。

汇总输出整体与逐相机统计，并按清单显式阈值检查：平均 IoU、最低单样本 IoU、person/floor
最大误纳率、每路相机最低样本数及必测相机列表。即使全部指标通过，报告仍固定
`approved_for_training=false`、`eligible_for_candidate_merge=false`；通过只表示完成评测门禁，不能
自动部署或写训练集。

模板建议先采用：平均 IoU 不低于 0.85、单样本不低于 0.75、person/floor 误纳率均不高于
2%，四路相机各至少 5 帧。阈值会记录在报告中，不使用隐藏默认值。

## 验证

- 新增 6 项主测试和 2 个子测试，覆盖完全匹配、低 IoU、person/floor 误纳、助手/机器真值拒绝、
  路径逃逸、非二值掩膜、训练划分、相机缺样本、重复样本、收据篡改和真值掩膜替换；
- 完整 Python：`1230 passed, 219 subtests passed`；
- 网页 Node：`167 passed`；
- 正式工作台保持 idle，SAM、主 VLM、快速 VLM 均 ready；
- 四份正式状态/历史 SHA-256 未变化。

## 下一步

在本地隔离目录为四路各选至少 5 帧，人工逐像素确认 ego/person/floor 掩膜并生成绑定收据，再运行
该工具。助手目视复核可以用于排队和提示，但不能生成或签署人工真值收据。

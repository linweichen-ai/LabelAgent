# Round762：斜坡外部专家预评测队列冻结

## 目的

把已有斜坡专项包整理成一个不含图片路径、不复制图片、不可误用为人工真值的外部专家预评测队列。该队列用于明确后续人工复核工作量和分层结构，尚不能直接进行真实质量 A/B。

## 现有证据

来源是 Round160 的斜坡复核包，绑定：

- 数据 SHA-256：`ab84b8894be2fad0ea4064e0779715377a303bb1b7b988e6184cde4e082eb6a0`
- 来源清单 SHA-256：`1c67cb8dc22b225a7b3b0d149539ec210b7386c0a2a4f1016b7384160092a79c`
- 相机话题：`/CVAT/Task3/images`
- 32 帧：8 个 `positive_proxy`、24 个 `negative_cohort`
- 人工复核：32/32 待处理
- 人工参考：0/32
- 训练批准：0/32

`positive_proxy` 只是历史机器预标注线索，不是斜坡人工真值；`negative_cohort` 也只是待人工确认的困难负样本。Round160 的助手审阅意见继续保持 assistant-only。

## 新增工具与输出

- 新增 `tools/build_expert_preab_manifest.py`。
- 工具只保留数据哈希、话题、帧号、帧哈希、分层、候选数、来源摘要和复核状态。
- 删除源 MCAP 本地路径、原图路径和复核图路径；不复制任何图片。
- 拒绝 `test24`、重复帧、非 pending 状态，以及任何人工真值、训练批准或训练回流标记漂移。
- 每帧生成由数据哈希、话题、帧号和帧哈希共同绑定的稳定 sample ID。
- 当前输出固定 `ready_for_quality_ab=false`、`external_upload_authorized=false`、`human_reference_count=0`。

结构化清单：`docs/evidence/workbench-stability-20261117/round762-slope-expert-preab-manifest.json`。

## 验证结果

- 预评测清单新增 5 项测试，与人工参考 A/B 评测器合计 12/12 通过。
- Python 编译和 `git diff --check` 通过。
- 对真实 Round160 清单只读执行：32 帧完整冻结，8/24 分层保持不变，图片复制 0、图片路径 0、`test24` 使用 0、外部上传授权 0。

## 下一步

先在本地复核界面完成人工判断，至少确认斜坡是否存在；需要评估分割边界时，还必须完成人工轮廓修订。只有人工回执与帧哈希一致后，才能生成 `synthetic_fixture=false` 的真实 A/B 输入。即使后续 API 连通，未经用户单独授权也不得把这 32 张工厂图片发送给外部模型。

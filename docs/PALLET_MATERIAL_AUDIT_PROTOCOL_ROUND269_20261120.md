# Round 269：栈板材质复核与候选审核协议修复

日期：2026-11-20

状态：实现、测试、生产部署和线上候选复核均已完成；机器复核仍保持候选态，不写人工真值，不批准训练或部署。

## 1. 本轮解决的问题

Round 268 证明了栈板边界 v5 在 Capture_20261103 的 160 个分层候选上不会误修，但没有回答两个语义问题：

1. 蓝色塑料栈板、木栈板和严重遮挡栈板能否被复核模型稳定区分；
2. 明显纸箱被 SAM3 标成 pallet 时，复核记录是否自洽并能可靠进入人工复核队列。

本轮没有继续放宽几何清理门槛，而是把“边界好不好”和“类别对不对”分开处理。

## 2. 12 个大候选原图复核

从 Round 268 的 160 个 Capture_20261103 分层候选中，按像素面积选取最大的 12 个并生成原图与隔离视图。人工目视结果显示，多数为真实的蓝色塑料或木质栈板，人物、遮挡物和邻近背景能够被隔离视图排除。

两个需要单独处理的样本为：

- camera_0 第 72 帧 `sam3-camera_0-72-pallet-8`：明显是实心纸箱，属于 pallet 误检；
- camera_1 第 92 帧 `sam3-camera_1-92-pallet-0`：只剩严重遮挡的碎片，证据不足，不应自动确认。

本地证据：

- `artifacts/pallet-material-review-round269/capture-top12-original-isolated.jpg`
  - SHA-256：`65A58BAAC264C000B6937F760A43503FA4DEB7A966B275454262E2C7CB2FDBD0`
- `artifacts/pallet-material-review-round269/capture-top12-review.json`
  - SHA-256：`726E9C9AF1A6EB0D3D22EDBEED3D4D970113E4779FD79C50F4FFC1930B9B728E`

这些文件含厂内图像或本机路径，只保留本机，不上传代码托管平台。

## 3. 暴露出的协议矛盾

旧提示在蓝色塑料栈板上给出了正确视觉证据，但字段互相冲突：`classification=conflict`，同时又输出 `observed_label=pallet`、`candidate_supported=true` 和 `reason_code=visible_known_label`。这不是识别失败，而是协议输出不一致；若直接保存，会让后续 Agent、人工复核界面和训练门禁得到相互矛盾的证据。

失败样本保存在本机：

- `artifacts/pallet-material-review-round269/blue-plastic-protocol-failure.json`
  - SHA-256：`6112B2B710312F77C4D4571B098C7A77F6762F43B62A052555070FFCC65D94F8`

## 4. 实现变化

`source/agent/candidate_class_audit.py` 现在包含三层保护：

1. 首次提示明确字段组合：已知类别必须给出允许标签、支持候选并使用 `visible_known_label`；不确定或冲突必须保持 `observed_label=null`、`candidate_supported=false`，且不能使用 `visible_known_label`。
2. 仅当返回违反协议时，最多进行一次盲重试。重试不回显错误答案、不泄露当前绑定类别，仍使用相同图像和允许标签。
3. 对唯一一种窄矛盾做确定性修复：当类别、标签和支持字段已经一致地表示“不支持”，但原因码误写为 `visible_known_label` 时，只把原因码改为 `ontology_conflict` 或 `insufficient_visual_evidence`。不修改类别、标签、支持、置信度、可见证据、候选或路由。

每条审核记录新增 `protocol_retry_count` 和可选的 `protocol_repair`，便于界面和后续统计区分模型原始合规、重试合规与窄修复。

实现提交：`fe07aac350afeaff75ab5b9ac1f03aaf9edad9b9`。

## 5. 独立三类实测

在 Capture_20261103 的隔离候选上执行同一套新协议：

| 样本 | 结果 | 请求数 | 实测耗时 |
|---|---|---:|---:|
| 蓝色塑料栈板，camera_3 第 47 帧 | 保留 pallet，0.98 | 1 | 9.168 秒 |
| 清晰木栈板，camera_1 第 117 帧 | 保留 pallet，0.98 | 1 | 6.877 秒 |
| 严重遮挡碎片，camera_1 第 92 帧 | 进入人工复核，0.95 | 1 | 10.544 秒 |

三项 `protocol_retry_count` 均为 0。证据文件：

- `artifacts/pallet-material-review-round269/material-three-case-audit-v3.json`
  - SHA-256：`D0A8A5678549C2067E047E6ED63C56DCB2EDF65ED46797948065A4B039F225D2`

纸箱负例在已有精确请求缓存上回放时为 55.24 ms，使用一次模型结果并经窄原因码修复进入人工复核；该数值是缓存路径，不冒充冷启动延迟。证据文件：

- `artifacts/pallet-material-review-round269/camera0-frame72-box-audit-v3.json`
  - SHA-256：`7F76908D13635CD92B1A5E47C92329495BC0D8E1C09020C6B1151BE0BCE2D196`

## 6. 测试与生产部署

定向测试覆盖：首次合规、一次重试成功、第二次仍不合规时失败关闭、窄原因码修复以及其他矛盾不被自动修复。

正式验收：

- `pytest -q tests`：1,044 项通过；
- 子测试：186 项通过；
- 未为 SAM3 的可选演示依赖污染项目环境。

部署信息：

- 生产文件：`D:\Apps\LabelAgent\source\agent\candidate_class_audit.py`；
- 部署后 SHA-256：`2839A559E6AAA21E656CE4938D89B61B474D26704CBA3956C949FEBF3AE8A87E`；
- 回滚目录：`D:\Apps\LabelAgent\.local\backups\candidate-audit-protocol-round269-20261119-235846`；
- 回滚清单 SHA-256：`4856ADCEDEDD62A5AF96E2C15B9DFA58F8ED2B6B7A5F3ED8ABF3BE43CF5F5249`；
- 生产协议 SHA-256：`3c9a7bb6491dc6b64f1441c7860019290759922ee2de97763ca8d7313937364d`。

只重启了工作台；4090 安全通道和 5090 SAM 通道保持原进程，未中断 SAM3 或主 VLM 服务。5090 可选 VLM 仍未启用。

## 7. 生产线上验收

在正式工作台当前 CVAT Task3 数据集第 132 帧执行候选复核：

- 旧审核协议 SHA-256：`d9f1da2316ae44787612acecc1dd12b371a2bfae2a78ea177c0b396bdfbff706`；
- 新协议 SHA-256：`3c9a7bb6491dc6b64f1441c7860019290759922ee2de97763ca8d7313937364d`；
- 旧记录因协议不一致被正确失效，未命中缓存；
- 8.552 秒完成；
- 候选 9 判为实心箱体、非 pallet，置信度 0.98，路由 `human_review_required`；
- `protocol_retry_count=0`，`protocol_repair=non_known_reason_code_normalized`；
- 候选总数 14→14；
- 候选产物 SHA-256 前后均为 `7d88c20852f706ee41f7e92ea49d058fd3225e7548ae3b20d20fa2730327619c`；
- 人工真值前后均不存在；
- `candidate_only=true`、`human_truth=false`、`approved_for_training=false`；
- 审核回执 SHA-256：`232d6819d523d099a9639fead171c5d1f5701ee5b2c9551caf43b67642f3c251`。

这证明新协议已在正式工作台真实生效，并保持候选、人工真值和训练批准隔离。

## 8. 边界与下一步

- 12 个大候选是定向目视复核，不是随机准确率统计；
- 三类样本证明了蓝塑料、木质、严重遮挡和箱体负例的路径，但不能外推到未来所有材质和光照；
- 本轮修复的是候选语义审核协议，没有重新训练 SAM3，也没有把 VLM 结果自动提升为人工真值；
- 下一步应优先清理磁盘余量、补充独立人工复核样本，并在不放宽候选门禁的前提下统计更多材质的误检率和复核耗时。

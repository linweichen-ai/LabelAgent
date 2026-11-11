# 固定工厂 holdout 垂直链回执（2026-11-11）

## 结论

固定 640×512 工厂 holdout 已完成 `SAM 候选 → 原图+叠图 VLM 首审 → Agent 动作 → 原图+输出叠图 VLM 二审`。最终状态为 `candidate_final`，只表示机器候选链路完成，不是人工真值、训练数据或分割质量验收。

- `input_id`：`bc706b57eedaea59e3deda1e25eb5a20002e90b61089083e84d2f98130c5600b`
- `frame_id`：`1789897506782450197`，`camera_0`
- 原图 SHA-256：`34e5094d24db487b3fcd5c458ac810cde5d58293f89fe0b92a2933c72a146812`
- 训练隔离：`holdout_excluded_from_training=true`
- 点 `(390,220)`：`assistant_selected_prompt / operator_seed`，不是人工选择，也不是人工真值。

## 各环节结果

1. SAM 候选哈希为 `0be59d...2593`；可读叠图哈希为 `dcd2aa...7323`。
2. 首轮 VLM 确实同时看到原图和叠图：判定 `slope`、已知类、非 unknown、候选受支持；漏分/多分/粘连均为 `none`，置信度 0.95。它建议 `split`，但与自身“无缺陷”判断矛盾。
3. Agent 明确选择 `no_op`：当前只有一个连通坡面，没有可拆分实例；输入 SAM artifact 哈希 `0be59d...2593`，本次历史运行的规范化输出候选哈希 `2495ee...7a5b`。两者解析后的候选内容一致；哈希差异只来自 JSON 序列化。工具随后已收紧为未来 `no_op` 逐字节保留输入，并有回归测试，本次既有回执不覆盖。
4. 第一次二审虽为 HTTP 200，但输出在 384 tokens 处截断，原始响应哈希 `f56c39...506b`，已按失败证据保留，未当作结构化成功。
5. 收紧类别枚举和理由长度后，独立二审再次判定 `slope`、无漏分/多分/粘连、置信度 1.0、`no_op`，并路由到 `candidate_final`。请求哈希 `aa50b0...35ab`，响应哈希 `80d114...bac6`。

最终回执内部哈希 `acbe7139...2f9d6` 已独立重算一致；本地回执文件哈希为 `9a8c9a2b...f37b7`。

## 人工门禁

本轮同步完成并验证：数据集/版本/相机/topic/帧/候选哈希绑定；状态与审计的事务式写入及故障恢复；驳回后候选移出活动区并归档；同一被拒候选不得原样接受；显式 `confirmed_empty`；`reviewer_declared_id + identity_assurance=self_declared`；只有服务端持久化回读 `verified=true` 才提示成功。

Python 3.11 环境相关测试 57/57 通过；schema 专用环境 12/12 通过；`node --check source/web/app.js` 通过。故障注入测试中预期出现的 500 traceback 是测试主动注入保存失败，不是未处理回归。

## Shipping-container smoke 收口

该 `/segment_text` 只能记为“服务推理通过、质量未验收”。主 part 约为 `x=299..1182, y=98..679`，另有两个仅 5/6 顶点、约 5×6 像素的疑似伪小块，不能据此声称分割质量良好。

## 业务边界

训练/标注/专项评测继续保留 `floor` 与 `slope` 两类；下游映射为 `traversable_area = floor UNION slope`。本轮没有 `human_accepted`、`truth_committed` 或 `training_ready`，也没有启动监督微调。

原始叠图、请求、响应、失败响应和最终回执位于：

`C:/Users/seer/Documents/ChatGPT/Seer2clw/outputs/sfl_cpd15y_stage2_20261111/fixed_holdout_chain/`

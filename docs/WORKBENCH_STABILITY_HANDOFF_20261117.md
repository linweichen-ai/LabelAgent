# LabelAgent 工作台稳定性修复交接（2026-11-17）

## 结论

正式工作台已切换到本轮修复，地址仍为 `http://127.0.0.1:8765`。本轮先解决“能否顺畅跑通”，不把机器候选当成人工真值，也没有启动训练或改变模型权重。

## 已解决的问题

1. 点击开始后长期显示 0%。前端此前没有识别 `claiming`、`discovering`、`discovering_retry` 等真实后台阶段。现在预检、VLM 类别识别、描述、SAM 生成、VLM 观察、Agent 编辑和保存均有明确状态；运行任务最低显示 1%，VLM 类别识别显示约 6%，并使用动态进度条表示后台仍在工作。
2. 模型离线后仍进入长请求。开始批量标注前先检查 SAM 与 VLM；未就绪时明确报出是哪一项失败，并且不会提交推理任务。
3. 长阶段像死机。后台状态现在提供任务总耗时、当前阶段耗时、状态序号和观察时间。单阶段超过 60 秒时，界面说明后台仍在线并可安全停止。
4. 历史状态文件过大。原 `state.json` 为 265,256,582 字节，逐行缩进且读写时复制整段文本。改为流式读写和紧凑 JSON 后，同一真实状态为 34,220,380 字节，减少 87.1%；隔离副本加载 2.79 秒、保存 6.30 秒，9 个历史文件会话全部保留。
5. 停止结果不透明。隔离任务在 VLM 识别阶段停止后约 12 秒进入 `stopped`，目标帧未写入候选，也未改动人工真值。

## 真实单帧验收

- 数据：`batchA_0920_000_clean_review.mcap`，相机话题 `/capture/camera_1/image`，第 31 帧。
- 总耗时：约 168 秒。
- 阶段序列：`claiming -> discovering -> describing -> generating -> observing -> finished -> completed`。
- 状态序号从 1 递增到 17；VLM 识别、SAM 生成、VLM 观察均可从状态接口独立观察。
- 生成 11 个机器候选：`floor` 1 个、`person` 2 个、`pallet` 8 个。
- VLM 还判断 `ego` 与 `other_robot` 可见，但没有对应 SAM 候选，因此保留为人工复核项。
- `slope` 在该帧被判断为不可见；这不是斜坡效果验收样本。
- 所有结果仍是 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`。

## 图像证据

- [原始帧](evidence/workbench-stability-20261117/frame31-original.jpg)
- [11 个候选叠加](evidence/workbench-stability-20261117/frame31-all-candidates.jpg)
- [人物候选盲审图](evidence/workbench-stability-20261117/frame31-person-audit.jpg)

## 自动化验证

- Python 服务端回归：94/94 通过。
- 候选审核前端回归：60/60 通过。
- 进度与轮询前端回归：6/6 通过。
- JavaScript 语法检查与差异空白检查通过。
- 正常完成与主动停止两条真实模型路径均通过。
- 正式工作台读回：状态 `idle`，进度契约版本 2，SAM/VLM 均 ready。
- 正式状态文件与隔离压缩副本 SHA-256 一致：`0B4D4284E887F3A664F7DF81DB8F1384699BA2944F6BB236132A74264437204F`。

## 回滚与协作

- 原始生产状态备份：`D:\Apps\LabelAgent\backup\state-before-workbench-stability-20261117.json`。
- 原始状态 SHA-256：`73A74432246316DE525DD2CDB22D28C5142955B702F47FA90AB90E7A569106A1`。
- 开发分支：`codex/workbench-stability-20261117`。
- 另一个 LabelAgent 对话应基于本文件和该分支继续，不要重复修复 0% 进度问题；下一优先级是批量连续运行、失败帧重试和斜坡专项评测。

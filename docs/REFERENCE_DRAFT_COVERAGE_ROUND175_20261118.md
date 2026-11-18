# 参考帧两阶段首标补漏（Round 175）

## 问题

旧的推荐模式先让 VLM 判断当前帧可见类别，只把被判为“可见/不确定”的类别交给 SAM。这样速度快，但 VLM 一旦漏看用户已经选择的类别，该类别首轮就完全没有轮廓。

## 本轮改动

- 新增“自动识别 + 已选类别补漏”可选模式；Round 177 真实 A/B 后默认恢复为快速模式。
- 第一阶段保留 VLM 可见类别识别与新类别发现。
- 第二阶段将未被第一阶段覆盖的已选类别作为独立 SAM 补漏探测。
- 保留“仅自动识别可见类别（更快）”和原来的逐类/单类别模式。
- 记录 `recall_probe_labels` 与 `probe_missing_selected`，可追溯哪些类别来自补漏。
- 补漏结果仍是 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`，必须人工复核后才能成为真值。

## 验证

- Python 全量回归：934/934 通过。
- 浏览器静态行为回归：46/46 通过。
- 正式服务读回：主页 HTTP 200，新模式文案和脚本均已生效。
- 模型状态：SAM ready、VLM ready（`labelagent-vlm8b`）。
- 连续推理预检：ready。
- 正式部署前备份：`D:\Apps\LabelAgent\backup\reference-coverage-v1-before-20261118T1735`。

## 当前边界

这项改动提供了参考帧首轮“补查已选类别”的能力，不代表每个补漏轮廓都正确。Round 177 的三帧真实 A/B 没有观察到新增有效轮廓，因此它保留为人工按需开启的实验模式，不再作为默认。系统继续保留人工复核、Agent 精修和高风险类别复核三道门禁。

# Round761：外部视觉专家人工参考 A/B 评测器

## 目的

在外部模型连通后，用同一批固定开发集困难帧比较不同模型的复核质量、稳定性、速度和成本。评测器只读取结果并生成报告，不修改候选掩膜、标签体系、人工真值或训练集。

## 评测门禁

- 输入必须明确 `human_reference=true`、`reviewer_confirmed=true`，否则拒绝生成任何质量指标。
- 分区名称和输入标记均不得包含或启用 `test24`；`test24` 继续只用于最终独立评测。
- 每条模型结果必须保持 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`。
- `passed` 结果必须逐一覆盖人工参考中的候选 ID；错 ID、重复 ID、遗漏候选或非法 verdict 均失败关闭。
- `failed` 和 `fallback` 结果不得携带伪造的复核内容，也不会被计入候选判断一致率的分母；但会降低结构化复核成功率，避免用少量成功样本掩盖失败。
- 工具不会自动选择赢家，也不会产生部署决定。

## 指标

- 结构化复核成功率、失败次数和回退次数；
- 候选 verdict 与人工参考的严格一致率；
- 已知类别漏检集合的微平均 Precision / Recall；
- 新类别名称大小写归一后的精确匹配 Precision / Recall；
- 单模型延迟中位数、P95 和预计总成本。

这里的“新类别名称精确匹配”不是语义等价判定，不能覆盖同义词、上下位词或类别体系判断；这类冲突仍需人工/助手复核。

## 实现与使用

- 核心：`source/agent/expert_benchmark.py`
- 命令入口：`tools/expert_vlm_ab_report.py`
- 测试：`tests/test_expert_benchmark.py`

```powershell
.\.venv\Scripts\python.exe tools\expert_vlm_ab_report.py --input <人工参考与模型结果.json>
```

## 验证结果

- A/B 评测器新增 7 项测试；连同矩阵、合成连通和复核协议共 23/23 通过。
- Python 编译和 `git diff --check` 通过。
- 使用 2 个程序构造样本和 2 个虚拟模型生成一份结构化报告，验证失败/回退、严格候选一致率、漏检、新类别、延迟和成本可以同时呈现；报告固定标记 `reference_mode=synthetic_fixture`、`human_reference=false` 和 `real_world_quality_evidence=false`，不能冒充人工或真实效果。
- 合成报告：`docs/evidence/workbench-stability-20261117/round761-expert-ab-synthetic-report.json`。

## 当前限制与下一步

当前报告只使用合成参考和虚拟结果，不能证明任何真实模型质量。实际 A/B 必须等外部模型完成合成图连通后，从非 `test24` 的固定开发困难集中选择人工确认样本，并冻结输入哈希、模型版本、提示词、参数和成本口径，再运行真实比较。真实模式要求 `synthetic_fixture=false`；合成模式必须明确 `human_reference=false`、`reviewer_confirmed=false`。

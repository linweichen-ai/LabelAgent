# 困难帧严格评测参考补全（Round 787）

## 本轮解决的问题

Round786 的 34 帧页面能复核候选、斜坡、类别冲突和托盘质量，但不能直接作为三臂评测参考：它没有逐帧确认全部八类是否存在，也没有记录可见托盘实例数。若用空值或模型结果补齐，会把机器判断伪装成人工参考。

Round787 保留原 34 帧与原图哈希，新增：

- 车辆、pallet、person、floor、slope、ego、other_robot、cargo_floor 八类存在性；
- 可见托盘实例数及存在性一致性检查；
- `uncertain` 选项，允许诚实记录看不清，但带不确定项的记录不会进入严格三臂评测；
- 托盘孔洞与粘连不再只汇总模型自报数量，而是与逐帧人工托盘质量裁决计算 precision、recall 和 F1；
- 从人工导出记录生成三臂输入骨架的只读转换工具。

旧 Round786 未覆盖、未删除，仍作为前一版工作记录保留。

## 使用方法

1. 打开 `artifacts/round787-diverse-hard-adjudication-benchmark-ready-20261124/reviewer.html`。
2. 完成 34 帧全部必选项并导出 `mixed-hard-adjudication.json`。
3. 使用原始 Round779 数据包和导出文件进行验证；只有 `benchmark_reference_ready=true` 才能进入三臂运行。
4. 使用 `tools/prepare_three_arm_reference.py` 生成评测输入骨架；骨架的 `runs` 初始为空，不会触发模型调用。
5. 后续仅在三臂使用相同 SAM 候选哈希、覆盖同一批样本后，运行 `tools/three_arm_benchmark.py`。

服务器只读入口：`http://127.0.0.1:18765/hard-review/round787/`（需要既有 SSH 转发）。

更顺滑且字段完全相同的后续入口为 Round789：`http://127.0.0.1:18765/hard-review/round789/`。它增加立即保存、完成计数和 J/K 翻页，但不自动预填人工答案。

## 安全边界

该复核是“声明式类别与实例数参考”，不是像素级人工真值。生成物仍固定：

- `candidate_only=true`
- `human_truth=false`
- `approved_for_training=false`
- `pixel_truth=false`

三臂输入中的 `human_reference=true` 仅表示这些明确字段经过声明式人工确认，不能解释为掩膜真值或训练授权。`test24` 没有进入该流程。

## 验证

- 复核工作区、旧格式兼容、存在性与托盘计数门禁：8 项通过；
- 三臂评测安全、计分协议与人工托盘质量对照：7 项通过；
- 合计定向测试：15/15 通过；
- 新工作区：34 帧、68 张图、0 个图片缺失，所有文件继续由收据哈希约束。

本轮只补齐真实评测入口，没有调用外部模型、没有运行三臂推理，也没有产生可宣称的模型效果提升。

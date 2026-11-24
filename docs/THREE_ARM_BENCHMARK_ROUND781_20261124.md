# Round781：本地 / 全外部 / 混合路由三臂评测器

## 目的

混合模型是否值得部署，不能只看几张效果图或模型回答一致率。本轮建立严格的同集三臂评测协议：

1. `local_only`：只使用本地 Qwen8B；
2. `all_external`：每帧都调用外部视觉专家；
3. `hybrid`：普通帧本地处理，仅困难帧调用外部专家。

三种方案必须使用相同样本和相同 SAM 候选哈希，避免通过重新分割不同候选来制造不公平对比。

## 指标

新增 `source/agent/three_arm_benchmark.py` 与命令行入口 `tools/three_arm_benchmark.py`，统一输出：

- 八个项目类别各自的 precision、recall、F1；
- micro precision、recall、F1；
- 斜坡正例召回；
- 仅在“人工确认 floor 存在且 slope 不存在”的帧上统计斜坡误检率；
- 托盘参考实例数、支持率、漏实例、多实例、孔洞错误和粘连错误；
- 车辆/ego/other_robot 类别冲突数；
- 每臂成功、失败、超时、回退；
- 平均延迟和 P95 延迟；
- 外部调用数、外部调用比例和预计成本；
- 三臂是否复用了同一个 SAM 候选。

斜坡与地面在评测中保持独立，同时明确记录下游允许合并为可通行区域。

## 失败关闭规则

评测器会拒绝：

- test24 或名称中包含 test24 的分区；
- 真实评测缺少人工确认参考；
- 三臂不是严格同样本覆盖；
- 同一样本在三臂间使用不同 SAM 候选哈希；
- `local_only` 偷调外部模型；
- `all_external` 没有调用外部模型；
- 无外部调用却声称发生回退或超时；
- 失败运行夹带预测；
- 机器运行写成人工真值或批准训练。

## 当前实测范围

输入模板位于 `docs/templates/three_arm_eval_input.example.json`。合成协议链路已跑通，报告 SHA-256 为：

`f1cf0b3cae821e9fe528c5164d8c2180d27a11e8abeb4c5ad18ff5ba0788200e`

合成报告仅验证计算和安全门，固定标记：

- `synthetic_fixture=true`
- `human_reference=false`
- `real_world_quality_evidence=false`
- `automatic_winner_selected=false`
- `deployment_decision=not_evaluated`

因此其中的数值不代表真实模型效果，也不能用于选择或部署模型。

## 验证结果

- 三臂覆盖、同一 SAM 候选复用、失败惩罚、斜坡/地面口径、调用契约和安全边界：6/6 通过。
- 项目主测试集：1881 通过、1 跳过、286 个子测试通过。
- 输入模板和结构化证据可解析，`git diff --check` 通过。

## 当前边界与下一步

Round779 的 34 张困难集尚未完成声明式人工裁决，所以真实三臂质量报告仍被正确阻断。下一步是完成这 34 张的人工类别存在性、托盘实例和错误类型确认，然后在保持同一 SAM 候选的前提下运行三臂影子对比。外部调用仍需按真实数据外发授权执行；未通过全部门槛前不部署。

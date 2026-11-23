# 外部视觉专家接入第三阶段：困难帧影子分流（2026-11-24）

## 目标

让 LabelAgent 能识别“哪些帧值得交给更强模型复核”，但在获得 API Key、完成公开图连通性测试和费用确认之前，绝不自动上传工厂图像。

## 已实现

- 新增确定性的困难帧判定器，并接入参考帧草稿与普通批量推理的持久化结果。
- 默认关闭；只有 `llm_expert_shadow_routing_enabled=true` 才记录路由建议。
- 当前触发证据包括：
  - 新类别候选或标签表冲突；
  - 可见类别不确定或置信度低于 0.65；
  - Agent `error`、`max_rounds`、`no_progress` 或延迟复核；
  - 复查轮数达到阈值，默认 4 轮；
  - 协议修复、JSON/协议失败或人工复核队列；
  - 类别复核冲突。
- `pallet`、`slope` 本身不会无条件触发昂贵模型；只有同时存在困难证据时才提高上下文优先级。
- 保存结果只包含原因代码、数量、类别和阈值，不包含图片、坐标、未知类别的原始描述或 API Key。

## 输出边界

每个影子判定都明确保存：

- `dispatch_performed=false`
- `external_upload_performed=false`
- `candidate_only=true`
- `human_truth=false`
- `approved_for_training=false`

因此，本阶段只是“分流建议”，不是外部模型调用，更不是自动审批。

## 配置

```json
{
  "llm_expert_shadow_routing_enabled": true,
  "llm_expert_shadow_round_threshold": 4
}
```

阈值只允许 1～100。模型就绪接口会显示影子分流是否启用及当前轮数阈值，但不会连接外部服务。

## 验证

- 新增默认关闭、正常帧不升级、复合困难证据升级、原因去重与确定性测试。
- 真实服务器路径测试确认：参考帧草稿的 `no_progress` 会保存专家复核建议；普通成功帧保存 `local_result_sufficient`。
- `tests/test_agent.py` 全量 293 项通过。
- Python 编译和 `git diff --check` 通过。

## 当前限制与下一步

- 尚未真实调用外部模型，因此没有外部模型质量、延迟或费用数据。
- 尚未自动执行分流；后续即使启用真实调用，也必须先保持影子模式并限制公开/脱敏图片。
- 下一步在用户选定供应商并安全配置本机环境变量后，执行一次公开图片连通性测试；随后对固定困难样本比较本地结果与专家建议，仍不写人工真值。

# Round760：外部视觉专家多模型合成图矩阵

## 目的

把单模型合成图连通性测试扩展为最多 8 个模型的可比矩阵。它用于统一检查协议、认证、结构化输出、成功/失败状态和单次延迟，为后续固定困难样本 A/B 做准备；它本身不评估分割精度，也不能产生部署结论。

## 工具行为

- 新增 `tools/expert_vlm_matrix.py`，输入只允许模型显示名、协议、URL、模型名和密钥环境变量名。
- 默认 `dry_run`，不联网、不上传；执行模式必须同时提供 `--execute` 与 `--acknowledge-synthetic-upload`。
- 拒绝 `api_key`、`key`、`secret`、`token`、`authorization`、`password` 等原始凭据字段，拒绝任何包含 `test24` 的配置，并拒绝图片路径等额外字段。
- 每个模型仍复用 Round756 的确定性合成图，不打开数据集、MCAP 或工作台图片。
- 单模型失败不会抹掉其他模型的回执；汇总只记录成功、失败、实际连通数量和合成图上传数量。
- 如果下游回执或异常意外包含环境变量中的凭据值，矩阵会将该条结果判为失败并删除凭据内容。
- 结果固定声明 `factory_data_uploaded=false`、`test24_used=false`、`candidate_only=true`、`human_truth=false`、`approved_for_training=false`、`deployment_decision=not_evaluated` 和 `quality_improvement_claimed=false`。

## 使用示例

配置文件仅保存非秘密信息：

```json
{
  "profiles": [
    {
      "name": "qwen-priority",
      "provider": "openai_chat",
      "url": "https://供应商地址/v1",
      "model": "qwen3-vl-plus",
      "key_env": "LABELAGENT_QWEN_API_KEY"
    }
  ]
}
```

默认离线预检：

```powershell
.\.venv\Scripts\python.exe tools\expert_vlm_matrix.py --profiles <配置文件.json>
```

只有完成本机环境变量配置并明确允许上传程序生成的合成图片后，才执行：

```powershell
.\.venv\Scripts\python.exe tools\expert_vlm_matrix.py `
  --profiles <配置文件.json> `
  --execute `
  --acknowledge-synthetic-upload
```

## 验证结果

- 新增矩阵测试 6 项；与现有合成连通和专家复核协议合计 16/16 通过。
- Python 编译和 `git diff --check` 通过。
- 四模型占位配置完成真实 dry-run：4/4 配置通过，实际连通 0、外部上传 0、工厂数据上传 0、`test24` 使用 0。
- dry-run 证据：`docs/evidence/workbench-stability-20261117/round760-expert-matrix-dry-run.json`，配置清单 SHA-256 为 `71f6ab148e0ec06bb7c80331e3fa434b27b5613305ae890997f7970d2243855b`。

## 当前限制与下一步

四个示例 URL 中使用 `.invalid` 占位域名，除 `qwen3-vl-plus` 外其他模型名也仍是占位符；本轮没有真实 API Key、网络调用或模型效果证据。拿到第一个供应商信息后，先只替换对应配置并执行单模型合成图连通性测试；通过后再进入固定开发集困难帧影子复核，不能用 `test24` 调参。

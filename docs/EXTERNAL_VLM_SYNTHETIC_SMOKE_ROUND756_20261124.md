# 外部视觉模型合成图连通性验收工具（Round756，2026-11-24）

## 用途

在提供 API Key 后，先用程序生成的合成工厂图验证外部模型协议、认证、结构化输出和延迟。该工具不会打开数据集、MCAP、工作台图片或任何真实工厂文件。

## 默认安全行为

- 默认只做 `dry_run`，不连接网络。
- 合成图固定为 640×512 JPEG，当前 SHA-256 为 `0a309a256896a8dbcd31c65624d62c9e522df38b171652d8b22066ab26cd6dd0`。
- 真正执行必须同时提供 `--execute` 和 `--acknowledge-synthetic-upload`。
- Key 只从 `--key-env` 指定的环境变量读取，不接受命令行明文 Key。
- 回执不保存 Key、图片 Base64 或模型原始输出，只记录供应商、模型、主机、合成图哈希、耗时和响应哈希。
- 回执始终明确 `factory_data_uploaded=false`、`candidate_only=true`、`human_truth=false`、`approved_for_training=false`。

## 使用方式

先做不联网检查：

```powershell
.\.venv\Scripts\python.exe tools\expert_vlm_smoke.py `
  --provider openai_responses `
  --url https://api.openai.com/v1 `
  --model <模型名> `
  --key-env LABELAGENT_EXPERT_API_KEY
```

确认供应商、模型和环境变量后，才执行合成图连通性测试：

```powershell
.\.venv\Scripts\python.exe tools\expert_vlm_smoke.py `
  --provider openai_responses `
  --url https://api.openai.com/v1 `
  --model <模型名> `
  --key-env LABELAGENT_EXPERT_API_KEY `
  --execute `
  --acknowledge-synthetic-upload
```

## 当前验证

- 4 项工具测试通过：默认不联网、执行双确认、缺少密钥失败关闭、回执不泄露密钥、内嵌 URL 凭据拒绝、异常响应拒绝。
- 本地真实命令完成 dry-run，确认没有 API Key 时 `ready_to_execute=false`，没有网络调用。
- 本轮没有真实外部请求、图片上传、训练、推理、部署或真值修改。

# 外部视觉专家模型接入第二阶段（2026-11-23）

## 本阶段完成了什么

在第一阶段安全门禁上增加三种原生协议，并保留原有兼容协议：

| 配置值 | 对应接口 | 认证方式 |
|---|---|---|
| `openai_chat` | OpenAI Chat Completions 兼容接口 | Bearer 请求头 |
| `openai_responses` | OpenAI Responses | Bearer 请求头 |
| `anthropic_messages` | Claude Messages | `x-api-key` 请求头 |
| `gemini_generate_content` | Gemini `generateContent` | `x-goog-api-key` 请求头 |

四种接口的图像、结构化 JSON 约束、Token 用量和结束原因统一转换为 LabelAgent 已有的内部格式。现有本地 Qwen8B、快速 VLM、SAM3 和 Agent 逻辑未改变。

## 安全和失败处理

- 外部专家仍默认关闭，只能显式启用。
- API Key 只从 `llm_expert_key_env` 指定的本机环境变量读取，不保存到配置、文档或 Git。
- Gemini Key 使用请求头，不进入 URL、日志或浏览器历史。
- 外部请求默认 120 秒超时，可通过 `llm_expert_timeout_seconds` 在 10～300 秒内调整。
- 响应为空、结构未知或没有文本时失败关闭，不把空结果当作有效 JSON。
- 只有 `openai_chat` 保留原有结构化解码兼容重试；其他原生协议不会套用错误的 Chat Completions 回退格式。
- 外部结果仍是候选：`candidate_only=true`、`human_truth=false`、`approved_for_training=false`。

## 离线验证

- 7 项专家通道定向测试通过。
- `tests/test_agent.py` 全量 290 项通过。
- Python 编译检查通过。
- 测试只拦截并检查 HTTP 请求对象，没有连接任何模型服务、没有 API Key、没有上传图像。

## 当前没有做的事

- 没有真实调用 OpenAI、Claude、Gemini 或阿里百炼。
- 没有做模型质量、延迟和费用对比。
- 没有自动困难帧路由、训练、微调或部署。
- 没有修改人工真值、训练集或 `test24`。

## 下一步安全接入流程

1. 选定一个供应商与模型；不要在聊天中粘贴 Key。
2. 在运行 LabelAgent 的本机设置专用环境变量，例如 `LABELAGENT_EXPERT_API_KEY`。
3. 先用公开或脱敏图片做一次连通性测试，记录请求耗时、Token、费用和错误格式。
4. 用固定困难样本做影子 A/B；外部模型只生成助手候选，不写人工真值。
5. 只有协议成功率、质量和成本均通过门禁，才接入低置信度困难帧自动路由。

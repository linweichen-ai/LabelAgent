# Round 856：外部专家仅预检模式

## 目的

在获得真实图片外发授权后，操作人员需要先确认整批输入、目标模型、凭据和账本状态，而不是运行命令后立即开始上传。组合执行器因此新增 `--preflight-only`。

## 行为

仅预检会完整执行以下本地检查：

- v4 明确授权及授权人；
- 34 张源图与 17 张缩放图的队列、manifest、receipt 和逐图 SHA-256；
- `candidate_only`、非人工真值、非训练批准及 `test24` 隔离；
- provider、HTTPS endpoint 与模型精确匹配；
- API Key 环境变量存在且格式有效；
- 持久账本是否未使用，或是否属于同一输出目录的合法恢复。

成功时只输出安全摘要，明确标记：

- `ledger_claim_created=false`；
- `external_call_performed=false`；
- 最大调用上限 51；
- 当前已有占位数量；
- 目标模型和样本数量。

失败时返回结构化 `status=refused`，不显示 Python 堆栈，不输出凭据，不创建输出目录或账本。

## 真实冻结模板拒绝验证

当前 Round 853 模板仍为 `authorized=false`。使用真实 34/17 队列、真实 manifest/receipt、当前 Astra 配置执行仅预检，结果为：

- exit code：2；
- status：`refused`；
- 原因：缺少匹配的明确外发授权；
- 账本创建：否；
- 输出目录创建：否；
- 网络请求：0；
- 图片上传：0。

这证明未授权模板不会因为“仅预检”功能而绕过外发门禁。

## 测试

- 授权、账本、凭据与仅预检定向测试：42/42 通过；
- 全量回归：`2081 passed, 1 skipped, 294 subtests passed`；
- 正式工作台部署：否；
- `test24` 使用：0。

## 下一步

用户明确授权后，先运行仅预检；预检为 `ready` 后再运行 8 张 source canary。canary 的成功率、延迟和 Token 门禁通过后，才继续剩余样本。

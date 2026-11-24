# Round 853：外部专家授权精确绑定

## 发现的问题

旧版组合授权已经绑定两份外部队列及每张源图哈希，但没有把复核清单、文件收据、目标 API 地址和模型名纳入授权内容。若仅修改旧模板的 `authorized` 字段，理论上仍可能把同一授权误用于另一份清单或另一目标模型。

## 本轮修复

外部上传授权升级为 `labelagent.external-upload-authorization.v4`，授权文件现在同时绑定：

- 34 帧复核清单 SHA-256；
- 复核包文件收据 SHA-256；
- 全外部 34 帧队列和混合 17 帧队列 SHA-256；
- 两种上传图像规格和各自调用上限；
- 规范化后的 provider、无凭据 HTTPS 地址和模型名；
- 总调用上限 51 次；
- `candidate_only=true`、`human_truth=false`、`approved_for_training=false`、`test24_used=false`。

执行器会在任何网络请求之前核对上述信息。复核清单、收据、队列、接口或模型任一不一致，都会失败关闭；带查询参数、内嵌用户名/密码或非 HTTPS 的目标地址也会被拒绝。旧 v3 组合授权不再允许进入组合执行器。

## 当前冻结模板

新模板位于本地 `round853-strict-expert-authorization-template-20261125/authorization-template.json`，其关键绑定为：

- manifest：`4cd7eadb244cadad83208b4f2d9f304da457830b839070dd796edf77f53dad50`；
- receipt：`0d1b15ccc09f3b138a2c1e3639b0740a09dd4438e35775af4cd0d4434470ebfd`；
- all34 队列：`63638ee8754b0b274562cf279e564f5fcbf6ba72b4af16a8c58c87fb589b782c`；
- hybrid17 队列：`39e12f5f7fa5823ec925a364c0b630333a59e72dc9c8d3e0d1347320e012df9b`；
- provider：`openai_chat`；
- 模型：`openai/gpt-6-astra`；
- 状态：`authorized=false`。
- 模板 SHA-256：`01582e26d0560ff229fd59a878051011056164ccdbcd127dc65b4f6fca7c553e`。

模板不含 API Key，也没有发出任何工厂图像请求。

## 验证结果

- 授权构建、输入加载、准备状态审计、组合执行与错误目标拒绝定向测试：31/31 通过；
- 项目全量回归：`2070 passed, 1 skipped, 294 subtests passed`；
- 实际冻结输入只读审计：v4 `safe_template=true`、`authorized=false`，身份绑定全部匹配；
- 外部 API 调用：0；
- `test24` 使用：0；
- 正式部署：否。

## 当前限制

34 帧人工复核仍为 0/34，外部图片授权仍未取得。因此三路质量比较和后训练都保持关闭，本轮只完善安全门禁，不宣称模型效果提升。

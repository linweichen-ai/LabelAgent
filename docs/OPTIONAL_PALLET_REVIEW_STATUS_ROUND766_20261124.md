# Round766：可选托盘复核状态无错误启动

## 问题与修复

工作台启动时会探测可选的托盘候选复核页。此前在没有配置该复核包时，状态接口返回 HTTP 404；功能本身会正确隐藏入口，但浏览器仍把它记录为资源错误，干扰验收和问题排查。

现在仅在环境变量未配置的正常场景返回 HTTP 200，并明确给出 `available=false`、`read_only=true`、`production_deployed=false`、`human_truth_writes=0` 和 `training_data_writes=0`，同时禁止缓存。已经配置但目录、清单或图片损坏时仍保持原来的失败关闭响应，不会掩盖真实安全故障。

## 验证

- 托盘候选复核测试：4/4 通过，包括默认关闭、已验证只读包、资源篡改和缓存失效。
- 斜坡专项复核 API：7/7 通过。
- 网页主逻辑：95/95 通过。
- 隔离端口 18771 真实 Edge 复测通过：页面错误、失败请求、HTTP 错误、被忽略的可选 HTTP 错误和真值写请求全部为 0。
- 复测后端口已关闭；正式工作台未部署，人工真值、训练数据、模型和 `test24` 均未修改。

## 证据

- 回执：`artifacts/round766-optional-review-status-browser/result.json`
  - SHA-256：`370a5378054ce06aaf61573690fbf8f522da025357fad257154cdfd41ad9d1ac`
- 截图：`artifacts/round766-optional-review-status-browser/review-packet-priority-queue.png`
  - SHA-256：`6a12507e6957bebf8ad1bd064fc1197ad13eca8f871a28a26f464c4fb84c6af5`

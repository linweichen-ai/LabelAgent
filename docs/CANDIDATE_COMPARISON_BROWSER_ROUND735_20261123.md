# 候选对比隔离浏览器验收（Round735）

## 结论

通用候选对比能力已在隔离的真实工作台和真实 Edge 中通过端到端验收。验收只读取正式工作台状态的本地副本和原数据，不调用推理、不占用 GPU、不写人工真值、不写训练集，也不修改正式 `D:\Apps\LabelAgent\work`。

本轮仍未部署正式工作台。Round733 原包继续只作为冻结本地证据；正式服务重启会改变文件载入代次，未来若部署，必须在重启后按新代次重新生成候选包，不能放宽门禁。

## 隔离环境

- 隔离端口：`http://127.0.0.1:8877/`
- 隔离工作目录：`artifacts/round735-candidate-comparison-isolated/work/`
- 数据路径：沿用正式当前数据的只读绝对路径
- 隔离文件载入代次：`1838475676874012`
- 请求帧：0、76、151
- 生成候选对比：33 项、133 个文件
- 候选包 manifest SHA-256：`4822f86e838a7263dd22e69b474d02c1b39c223c85454e594211ef7c3adcc737`
- 候选包文件树 SHA-256：`b2cb58f8d9362a9da6cf49398d3dc6f48b36204714c491466a7781f5025e84ab`

这些哈希与 Round733 不同仅因为 manifest 绑定了新的隔离服务载入代次；候选内容仍由同一确定性生成器产生。

## 浏览器验收结果

真实 Edge 检查通过：

- 页面 HTTP 200；
- 自动发现 1 个对比包；
- 读取 33 条对比记录；
- 当前候选和边界清理候选两张预览均成功载入；
- 会话查看偏好写入当前 `sessionStorage`，同一会话刷新后保留；
- 新浏览器会话不继承旧偏好；
- 模拟候选目录刷新为空后，旧包、旧记录、旧卡片和旧文字全部清空；
- 候选对比 API 非 GET 请求为 0；
- 人工真值、人工审核和参考帧受保护写请求为 0；
- 页面脚本错误为 0，意外 HTTP 错误为 0。

页面会按现有设计探测可选的 `/api/pallet-candidate-review`；隔离环境未配置该独立能力，所以出现 6 次预期 GET 404。验收单独列出并仅豁免这一精确 URL，没有忽略其他 HTTP 错误。

验收收据和截图保留在本地：

- `artifacts/round735-candidate-comparison-isolated/browser-acceptance-v3/acceptance.json`
- `artifacts/round735-candidate-comparison-isolated/browser-acceptance-v3/candidate-comparison.png`

## 本轮发现并修复的问题

1. 独立审阅发现：目录刷新失败或变空时，旧预览可能残留。现统一清空 packet、row、scope、图片状态、选择框和内容区；真实 Edge 以空目录响应验证失败关闭。
2. 第一次隔离生成在 Windows 深目录触发路径长度限制。生成器现使用更短的临时目录名，并在输出本身位于 `candidate_comparison_packs` 时把临时根上移一层；临时目录创建和资源写出也全部纳入 `finally` 清理边界。
3. 新增候选 ID 门禁：非整数或负数 pallet ID 在写出任何资源前拒绝，避免候选 ID 参与文件名时形成路径风险。

## 回归

- Python 全量：1733 passed，1 skipped，272 subtests passed；
- Node 全量：182/182 passed；
- 候选对比定向：13 passed，10 subtests passed；
- 隔离真实 Edge：passed=true；
- `git diff --check`、Python 编译和 JavaScript 语法检查将在提交前再次执行。

## 发布边界

Round733/734/735 原始 manifest 含本机和数据绝对路径，全部保留在未跟踪本地 artifacts，不提交 GitHub/CNB。仓库只提交通用生成器、验证器、浏览器验收工具、测试和不包含绝对数据路径的总结文档。

当前能力仍是只读比较，尚未提供“采用替代 mask”。通用读取器会验证路径、文件大小、解码格式、尺寸上限和哈希，但不会把第三方包的 mask 与当前候选重新栅格化逐像素比对；大包详情和图片请求也会重复完成整包校验。若未来增加采用/签核入口，必须先补二值 mask、原帧尺寸、当前候选栅格一致性和缓存性能门禁，不能直接复用当前会话偏好作为写入授权。

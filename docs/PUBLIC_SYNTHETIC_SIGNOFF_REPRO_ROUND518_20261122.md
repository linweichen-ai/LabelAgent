# Round518：公开合成离线签字浏览器复现

## 为什么补这一轮

Round517 已证明真实81项私有复核页可以顺畅导出、刷新、恢复和正式导出，但复现命令依赖本机 `.repro` 页面，外部克隆仓库的人没有私有图片和候选身份，无法独立运行同样的验收。本轮增加公开合成入口，让复现者无需任何真实数据即可验证浏览器工作流与授权边界。

## 实现

- `tools/verify_synthetic_signoff_browser.py` 固定生成81项 `synthetic-000`～`synthetic-080`，五个身份/哈希绑定字段均由公开固定字符串确定。
- 证据图为内嵌1像素 PNG，不读取厂内图片、MCAP、test24、人工签字、状态文件或凭证。
- 临时 HTTP 服务只绑定 `127.0.0.1` 随机端口；浏览器只允许访问该端口与内嵌 `data/blob`，其他请求会被拦截并使自检失败。
- 真实浏览器依次执行部分草稿下载、刷新、导入恢复、坏哈希拒绝、完整合成草稿导入与正式 JSON 下载。
- 输出目录必须不存在；浏览器、HTTP 服务和临时下载在成功或失败后清理。截图仅在 `--screenshot` 时生成。

## 实测结果

同一入口连续运行两次：

- 页面 SHA-256 均为 `86c07e2415929ba165610b86f9e9419a988e7b1ec7a59319da0426b8d09ffc00`；
- 部分草稿 SHA-256 均为 `7a6fa3c8fad3975839cb0f0750b8ace83196d94de0de424c677c95b75b476d1c`；
- 正式合成导出 SHA-256 均为 `59f019bd9a28df7b91fe22616547c656cc92afd1f7ddff9051b5bcb7c40d0162`；
- 外部请求0、页面错误0；第二次公开证据运行耗时2,265 ms；
- `training_authorized=false`、`automatic_approval=false`、`approved_for_training=false`；
- 真实人工决定写入0。

另外，合成fixture的确定性、81项唯一身份、SHA-256字段和人工字段为空共2项单元测试通过。Round517关联的真实浏览器、Node草稿攻击和Python签字门禁保持通过。

## 公开产物

- `artifacts/round518-public-synthetic-signoff-browser-20261122/receipt.json`
- `artifacts/round518-public-synthetic-signoff-browser-20261122/synthetic-reviewer.png`

截图中只有公开合成身份和黑色占位图，可以随 Git 发布。它不是人工标注效果展示。

## 复现命令

```powershell
# setup_windows.ps1 已默认安装浏览器验收依赖
.\.venv\Scripts\python.exe tools\verify_synthetic_signoff_browser.py `
  --output .repro\synthetic-signoff-browser-check --screenshot
```

## 结论边界

本轮把“只有原开发机能验收”推进为“公开仓库克隆后可独立验收离线复核工作流”。它不包含真实图像、真实候选或人工真值，不证明分割质量，也不授权训练；真实81项仍须由人工逐项签字并通过既有失败关闭导入门禁。

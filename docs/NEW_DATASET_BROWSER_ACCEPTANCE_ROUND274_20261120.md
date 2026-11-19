# Round 274：新数据 ego 补救真实浏览器验收与完成态提示修复

日期：2026-11-20

状态：新数据在隔离工作台的真实浏览器流程已通过；发现并修复“后台已完成但顶部仍显示处理中”的体验问题，修复已部署到正式工作台。

## 1. 验收范围

在独立端口的空白工作台中上传批次 B 003 的一张 640×512 图像，使用真实 Edge 页面完成：

1. 打开数据管理器并上传图片；
2. 加载图片并通过兼容检查；
3. 打开参考帧准备；
4. 点击自动起稿；
5. 等待首批候选显示；
6. 等待 Agent 与高风险类别后台复核完成；
7. 检查 ego 点扫描安全提示、候选边界与人工真值隔离；
8. 保存整页截图和结构化回执。

正式 CVAT Task3 未切换，所有写入仅发生在隔离工作目录。隔离端口在验收后停止，远端 SAM/VLM 通道保持不变。

## 2. 第一轮发现的问题

第一轮真实浏览器结果：

- 上传约 31 ms，加载约 172 ms；
- 首批 17 个候选约 18.032 秒显示；
- 后台复核约 75.813 秒完成；
- 候选为 ego 1、floor 1、pallet 13、person 2；
- 页面无 JavaScript 错误；
- `candidate_only=true`、`human_truth=false`、`approved_for_training=false`；
- 没有创建人工真值或参考帧。

截图中图层与原图对齐，ego 位于画面底部真实宿主底盘区域，安全提示明确说明这是文本分割漏检后的高风险点扫描候选。但顶部向导仍显示“模型正在处理第 0 帧，请等待结果”，而底部已经显示“高风险类别后台复核已完成”。这会让用户误判为流程卡住。

## 3. 修复

在参考帧后台精修和高风险审核的每个终态增加一次 `labelagent:candidate-state-synced` 事件：

- 审核成功；
- 审核失败但候选保留；
- Agent 精修成功；
- 人工已经开始编辑，因此不覆盖人工草稿；
- 精修失败或过期。

顶部向导原本已经监听该事件，本轮只补齐终态通知，不改变候选、人工真值、模型调用或保存门禁。人工编辑导致旧后台结果失效的保护仍然保留，并更新了对应回归断言。

## 4. 第二轮浏览器复验

重新启动全新的隔离工作区，使用同一图像复验。模型缓存已热，因此速度只代表热缓存演示路径，不能冒充冷启动性能：

- 上传约 32 ms，加载约 109 ms；
- 首批 17 个候选约 1.625 秒显示；
- 后台复核约 2.750 秒完成；
- 顶部向导在后台完成时立即显示“已载入 17 个候选，正在逐个检查”；
- 向导完成态等待为 0 ms；
- ego 安全提示可见；
- 页面无 JavaScript 错误；
- 没有创建人工真值或参考帧。

截图经人工查看：候选层与 640×512 原图贴合，ego、人员、托盘和地面均正常显示；底部复核面板与安全提示没有遮住主要画面区域。

## 5. 测试、依赖与部署

为避免 Codex 运行时升级导致旧 Node Playwright 绝对路径失效，新增 Python 浏览器验收工具 `tools/browser_ego_fallback_acceptance.py`。浏览器依赖单独记录在 `tools/requirements-browser-qa.txt`：

`playwright==1.65.2`

该依赖安装在 `D:\Apps\LabelAgent\.venv`，只驱动本机已有 Edge，没有下载新浏览器，也未修改系统 Python。

测试结果：

- 全量 Python：1,057 项通过，另有 186 个子测试通过；
- 全量前端 Node：143 项通过；
- `node --check source/web/app.js` 通过；
- 浏览器验收脚本编译与两轮真实执行通过。

正式 `app.js` 部署前备份位于：

`D:\Apps\LabelAgent\.local\backups\round274-guide-state-predeploy-20261120\source\web\app.js`

旧 SHA-256 为 `F476AFD2...B4549A`，新版本、生产文件和 HTTP 实际返回内容的 SHA-256 均为 `0154B671...DC04A`。

## 6. 证据与边界

本地图像证据：

- `artifacts/round274-ego-ui-v2/screenshot.png`
- `artifacts/round274-ego-ui-v2/result.json`

无图结构化证据：

- `docs/evidence/workbench-stability-20261117/new-dataset-browser-ego-round274.json`

正式工作台最终读回 HTTP 200、`idle`，当前仍是 CVAT Task3 和 `/CVAT/Task3/images`。本轮没有修改正式候选、人工真值、训练集或模型权重。

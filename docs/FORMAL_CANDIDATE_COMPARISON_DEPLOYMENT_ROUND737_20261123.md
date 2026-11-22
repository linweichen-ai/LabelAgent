# 正式候选对比部署与验收（Round737）

## 结果

通用只读候选对比能力已部署到正式 `http://127.0.0.1:8765/`，并基于重启后的新文件载入代次重新生成正式候选包。页面和真实 Edge 验收通过；正式人工真值、文件库和会话状态内容未变化。

正式页面现在可在候选复核区域打开“候选对比”，查看 33 个栈板当前边界与确定性边界清理候选。这里没有采用、保存、自动审批或训练写入入口。

## 回滚与部署

- 部署前运行状态：idle；
- 回滚备份：`D:\Apps\LabelAgent\.local\deploy-backups\candidate-comparison-20261123-064527`；
- 备份文件：4 个原代码文件、4 个正式状态文件，共 8 个，均已记录并读回 SHA-256；
- 新增正式代码：`candidate_comparison.py`、`candidate-comparison-core.js`、`candidate-comparison.js`；
- 更新正式代码：`server.py`、`app.js`、`index.html`、`style.css`；
- 正式重启通过现有受管启动器完成，未启动训练或常驻占卡任务；
- 启动后 SAM3 优先 5090 通道验收通过，主 VLM 预热 0.076 秒。

## 新代次候选包

- 文件载入代次：`1838476875584112`；
- 包 ID：`pallet-boundary-live-20261123`；
- 帧：0、76、151；
- 对比项：33；
- 文件：133；
- manifest SHA-256：`85cf54bef4e6892e29cd3fee41be66a7e2b63c00114b32d1c43dc832ce738a2e`；
- 文件树 SHA-256：`a3fe81eaa223d2b852bf82a24a148cd7b448072b1ec9c6ee8ee3d856672f4004`；
- `candidate_only=true`；
- `human_truth=false`；
- `approved_for_training=false`；
- `automatic_approval=false`；
- 生成期间服务器方法仅 GET。

旧 Round733/734/735 包没有复制到正式 WORK；正式包是在服务重启后针对新代次重新生成的。

## 正式 Edge 验收

- HTTP 200；
- 对比包 1 个、记录 33 条、当前/替代预览 2 张；
- 同一会话刷新后偏好保留，新会话不继承；
- 模拟目录变空后 pack/row/card 均为 0，旧内容为空；
- 候选对比 API 非 GET：0；
- 人工真值、人工审核和参考帧受保护写请求：0；
- 请求失败：0；
- 页面错误：0；
- 意外 HTTP 错误：0；
- 结果：`passed=true`。

本地收据与截图：

- `artifacts/round737-candidate-comparison-formal-browser/acceptance.json`
- `artifacts/round737-candidate-comparison-formal-browser/candidate-comparison.png`

可选的旧托盘候选复核 API 未配置，因此页面启动期间出现精确 URL `/api/pallet-candidate-review` 的预期 GET 404；验收仅豁免该精确端点，没有忽略其他错误。

## 正式状态读回

- `state.json` SHA-256：`9ef08118dd2ba0760cc93cf57bbeaae4b3bed062d383df4e6a454c389d4ab115`；
- `files.json` SHA-256：`7edb40a9bdfcf5577aea4c2bb183ef556a2b7d70b24564c49eeb46c5d9a8c5c1`；
- `file_sessions.json` SHA-256：`94e8bee324628585e7532e595348293df1da7b24041a1d4df1c2ec9fb734e02e`；
- 工作台状态：idle；
- SAM 状态：ok；
- 候选索引：1 包 / 33 项；
- 索引声明：`human_truth_writes=false`、`training_data_writes=false`。

前三个状态哈希与部署备份完全一致，证明本次重启和验收没有改写既有正式状态内容。新增的正式候选包仍是本地只读候选证据，不是人工真值或训练数据。

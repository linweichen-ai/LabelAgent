# 第二百三十九轮：模型配置、启动器与新数据验收链路统一绑定

日期：2026-11-19

状态：已部署、已完成全量回归与正式 HTTP 验收

工作台版本：UI 153、`app.js?v=191`

## 通俗结论

以前网页、启动器、VLM 连接工具都能修改模型配置，但它们没有携带“这是哪个文件、哪个相机、哪次载入、基于哪版配置”的完整凭证。旧网页或较慢返回的工具可能覆盖较新的配置；同时，隔离工作台和新数据集验收脚本仍沿用旧类别写法，已经不兼容前面新增的安全门禁。

现在所有正式配置写入都先读取服务器当前绑定，再提交：

- 当前数据路径；
- 当前图像话题；
- 当前文件载入代次；
- 完整配置状态 SHA-256。

其中配置指纹会包含密钥的变化，但浏览器和状态接口不会返回 `llm_key` 或 `llm_fast_key` 的原文。旧绑定返回 HTTP 409；自动标注运行中返回 HTTP 503；落盘失败会恢复内存中的原配置，并返回可读 JSON 错误。

## 同步修复的辅助链路

- 正式双服务器启动器：启动后读取当前绑定，再更新 SAM/VLM 路由；遇到一次并发 409 时重新读取绑定，不复用旧基线。
- 隔离工作台启动：配置和类别写入都改用当前绑定；修复了类别接口升级后隔离启动脚本仍发送旧格式的问题。
- `connect_local_vlm.py`：真实 CUDA VLM 探针通过后，绑定当前工作区写配置，并核对成功回执。
- `run_new_dataset_smoke.py`：先绑定类别版本，再绑定配置版本，恢复新数据集独立验收能力。
- 浏览器配置弹窗：提交时冻结数据路径、相机、载入代次和配置 SHA，只在回执仍属于同一页面时更新本地状态。

## 测试与正式验收

- Python 全量：994/994，通过；
- 浏览器 JavaScript：69/69，通过；
- 定向配置、启动器、VLM连接工具测试：7/7，通过；
- Python 编译、JavaScript 语法和 `git diff --check`：通过。

正式工作台第一次配置验收时，恰有一项真实自动标注任务在运行，接口按设计返回503，没有抢占或覆盖任务。该任务完成后重新验收：

- 当前绑定无变化的配置保存：HTTP 200；
- 成功回执的旧 SHA、路径、相机和载入代次全部匹配；
- 故意使用错误配置 SHA：HTTP 409；
- 错误配置 `must-not-apply` 未写入；
- bootstrap 不含两个密钥字段；
- UI 153、`app.js?v=191` 和客户端绑定字段均从正式服务读回；
- 模型总就绪、SAM 就绪、主 VLM 就绪均为 true。

## 部署与回滚

部署前备份：

`D:\Apps\LabelAgent\.local\backups\config-write-binding-round239-20261119-125631`

部署文件：

- `source/agent/server.py`
- `source/web/app.js`
- `source/web/index.html`
- `.local/manage_remote.py`
- `tools/connect_local_vlm.py`
- `tools/run_new_dataset_smoke.py`

实现提交：`ae9c75d4cc01d761d74f982fdb836f63fa9902ad`

代码、测试、报告与机器证据已同步到 GitHub 和 CNB；两端分支均精确读回提交 `a5622f6eff19b02fa7e92e82faef0c2d1f990837`。

## 安全边界

本轮配置验收没有写人工真值、参考帧或训练批准。正式状态随后因操作者真实运行自动标注和切换数据而按预期变化，不能把部署前后的 `state.json` 差异误报为部署污染；`files.json` 保持不变。微调或后训练没有因本轮配置修复自动启动。

机器证据见 [`evidence/workbench-stability-20261117/config-write-binding-round239.json`](evidence/workbench-stability-20261117/config-write-binding-round239.json)。


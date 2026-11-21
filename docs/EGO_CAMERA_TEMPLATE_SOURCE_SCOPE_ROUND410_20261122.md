# Round410：ego 相机模板数据身份隔离（2026-11-22）

## 结果

Round408–409 证明 Round385 的 camera0 固定正负点模板不能直接从 0920 开发序列复制到
1104 批次。本轮把这一实验边界变成运行时门禁：模板仅允许在路径组件精确包含
`Capture_20261103` 时进入只读影子诊断。

`SFL_CPD15Y_20261104_smoke`、其他工厂目录、相似后缀或只是在文件名中包含相近字符，
都会在调用 SAM 前返回 `source_scope_mismatch`。报告只保存输入身份的 SHA-256，不保存完整
路径。直接调用诊断函数且不提供来源身份仍保留给隔离研究工具；正式工作台始终传入当前文件
身份，因此不会绕过门禁。

## 安全属性

- 功能默认关闭，正式配置仍为未启用；
- 即使以后误开启，也只能在 `Capture_20261103` 数据目录运行；
- 仍不返回可合并掩膜，不写候选、人工真值或训练批准；
- 数据身份不匹配时不调用 SAM；
- 取消、超时和服务错误仍按原协议显式暴露。

## 测试与部署

- 定向回归：13 项通过，另有 10 个子测试；
- 完整 Python：1284 项通过，另有 225 个子测试；
- 运行版 1104 路径探针返回 `source_scope_mismatch`，模型调用断言未触发；
- 部署前备份：
  `D:\Apps\LabelAgent\.local\backups\ego-shadow-scope-round410-20261122`；
- `server.py` 与新增 `ego_shadow.py` 的运行版哈希和已测试源码完全一致；
- 受控重启后页面 HTTP 200、运行状态 `idle`；
- 主 VLM、快速 VLM 均返回 `labelagent-vlm8b`；
- point-SAM 为 `labelagent.remote-sam3.v3-point-groups`，模型已加载、在途请求为 0；
- 四个正式状态文件哈希与 Round409 前完全一致。

结构化证据：
`docs/evidence/workbench-stability-20261117/ego-camera-template-source-scope-round410.json`。

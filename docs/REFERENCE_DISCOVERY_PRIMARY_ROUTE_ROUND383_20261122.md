# Round383：参考帧类别发现固定权威主路（2026-11-22）

## 背景与结果

Round382 证明 5090 快路在 24 张 640×512 图像上虽然中位快 32.97%，但会相对
主路漏 `other_robot` 和 `floor`，不适合作为参考帧权威类别发现。正式工作台当前的
1920×1536 图像本来就走主路；风险出现在以后直接载入 640×512 新数据时，通用小图
路由可能自动选择快路。

本轮在 `discover_reference_labels()` 内对配置做内存副本，并移除副本中的
`llm_fast_*` 字段。这样只改变参考帧类别发现：

- 640×512 和更小的参考帧也固定走权威主 VLM；
- 结构化响应重试仍继续走同一主路；
- 原项目配置不被修改；
- 已通过门禁的单/双候选类别复核仍可使用 5090 快路；
- 不改 SAM、VLM 权重，不改人工真值和训练批准。

## 验收

新增两个单测分别验证主路强制和重试不回到快路。完整回归为 Python 1201 项及
190 个子测试、网页 159 项，全部通过。

部署后用一张真实 640×512 常熟图只读探测，回执明确为 `model_route=primary`、
`fast_fallback_reason=null`。该请求命中了已有响应缓存，489.962 ms 仅用于证明路由，
不能作为冷启动速度结果。

正式运行目录 `server.py` SHA-256 为
`E56E7B02EA7C7D1F1B9CD48FDCD5BCCE0A795D826300BCD08D4AA733600F0BF6`；
首页、bootstrap、模型就绪、运行状态和状态摘要均 HTTP 200，工作台 `idle`，SAM、
主 VLM、快速 VLM 均 `ready`。四份正式状态/历史文件哈希与 Round382 完全一致。

回滚文件位于
`D:\Apps\LabelAgent\.local\backups\reference-discovery-primary-route-round383-20261122\server.py`，
SHA-256 为
`749314B0DED546F2B7EFD4F305D827E554A70583E1DA2963E77A36FC912F2F54`。

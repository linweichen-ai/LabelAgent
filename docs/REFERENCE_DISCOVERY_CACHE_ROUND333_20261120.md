# Round333：参考帧发现结果严格复用

## 为什么优化

参考帧准备弹窗会提前完成一次VLM可见类别识别，但旧实现只依赖模型进程内部的响应缓存。
正式点击起稿时，服务器仍会再次进入VLM调用；一旦模型缓存被其他请求挤出，就会重复约15秒的
大图识别，也更容易出现共享模型繁忙。

## 本轮改动

工作台新增一个仅驻内存、最多16项、有效期5分钟的发现结果缓存。缓存键同时绑定：

- 当前文件载入代次和视频对象；
- 文件路径、相机、帧号；
- 解码后图像SHA-256；
- 完整标签表；
- 完整模型配置SHA-256。

任何一项变化都不会复用。缓存只保存机器发现证据，不写人工真值、不写训练批准，也不跨进程
持久化。正式起稿回执和Agent记录新增`discovery_prefetch_cache_hit`，便于验收和追溯。

## 真实隔离验收

在Capture真实MCAP的camera3第2帧执行完整“后台预取→正式起稿”：

| 项目 | 实测结果 |
|---|---:|
| 后台准备总耗时 | 17.405秒 |
| 其中VLM发现 | 15.898秒 |
| 正式点击到候选 | 2.622秒 |
| 候选数 | 4 |
| 服务器发现缓存 | 命中 |
| SAM精确缓存 | 命中 |

VLM选择`车辆/floor/slope/ego/other_robot`进入候选探测；这只是机器输出，不是类别真值。
预取前后隔离状态哈希不变；正式结果全部保持`candidate_only=true`、`human_truth=false`、
`approved_for_training=false`。结构化证据见
[acceptance.json](evidence/reference-discovery-cache-round333/acceptance.json)。

## 回归、部署和回滚

- 定向缓存测试：5/5通过；
- 完整Python：1,125项及186项子测试通过；
- 完整JavaScript：148/148通过；
- Python编译与差异检查通过；
- 正式源码部署前哈希：`C929B5D77F047D4C71348C3A3F0B3E6459874B6802FE13A0B9FFE822CC991BCE`；
- 部署后哈希：`75C6E1A8B76EE0A22C69AFC3E68F9F49C07587EC723B5276DC9F44E4243D4F17`；
- 回滚备份：`D:\Apps\LabelAgent\.local\backups\reference-discovery-cache-round333-20261120-095021`。

正式工作台重启后HTTP 200，4090/5090的SAM与VLM通道均在线。正式`state.json`哈希在部署
前后均为`CBA73B2DBFF671BE73EFFA6B67579C290337C755464EB18D66C0CC789BC7CD64`，
`files.json`哈希均为`7EDB40A9BDFCF5577AEA4C2BB183EF556A2B7D70B24564C49EEB46C5D9A8C5C1`；
现有1帧人工真值、149帧机器候选和25帧Agent记录没有变化。

## 边界

本轮减少的是“预取已经完成后又重复请求VLM”的冗余。第一次看到一张新大图时，4090 BF16
可见类别识别仍约15–16秒，只是计算被提前到弹窗准备阶段。它不是模型精度提升、SAM微调、
VLM后训练或跨数据集泛化证明。

# Round 272：新数据 ego 点扫描补救与生产部署

日期：2026-11-20
状态：条件触发的 ego 候选补救已完成隔离实测、回归测试、生产备份和部署；仍保持机器候选与人工真值隔离。

## 1. 本轮解决的问题

Round 271 已确认：11 月 4 日新数据的 003、004 图像中，VLM 都能看见画面底部的叉车自身结构（ego），但 SAM3 文本分割对七组不依赖颜色的提示均返回零候选。问题不是“橙色识别”，而是文字分割对贴边、局部可见的自车结构召回不足。

本轮把已验证的 18 个底部点批量扫描接入正式参考帧起稿流程，但只在以下条件全部满足时触发：

1. 配置显式启用 `ego_bottom_point_fallback`；
2. VLM 判断 ego 可见且置信度不低于 0.90；
3. 正常文本 SAM 没有生成任何 ego 候选；
4. 点服务证明图像只编码一次、18 个点全部返回且点位完全一致；
5. 候选几何满足经人工看图确认的 `wide_bottom_chassis` 或 `centered_fork_carriage` 之一。

任何调用失败、覆盖不完整或几何不合格都会保持无候选，并进入人工复核队列；不会用宽松轮廓冒充成功。

## 2. 新数据隔离实测

使用正式流程但隔离工作区，对批次 B 的 003、004 各运行一张 640×512 图像。结果均保持 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`。

| 图像 | 原候选 | 补救后候选 | 新增类别 | 端到端耗时 | 点服务内部耗时 |
|---|---:|---:|---|---:|---:|
| 003 | 16 | 17 | ego 1 | 38.512 秒 | 约 0.316 秒 |
| 004 | 13 | 14 | ego 1 | 13.328 秒 | 约 0.411 秒 |

两张新增 ego 均由后续独立类别盲审保留，置信度为 1.0。前后对照拼图已经人工查看，补出的轮廓位于真实可见的叉车底盘区域，没有覆盖地面、人员或托盘。

同一叉车相机样本的四路图像底部都含不同程度的宿主硬件，因此当前没有可称为“同序列真实无 ego”的自然负例。该事实没有被隐藏：本轮依靠“VLM 高置信度可见 + 文本 SAM 漏检 + 严格几何门禁”的组合限制，而不是把合成负例宣传成充分泛化证据。

## 3. 生产集成与操作边界

- 点提示服务使用独立配置 `sam3_point_url`，不会误走文字 SAM 快速路由；
- 合并候选后，复核队列和界面元数据中的候选 ID 会跟随最终 ID 映射；
- 工作台会显示安全提示：这是 VLM 看见、文本分割漏检后由点扫描补出的高风险候选；
- 用户仍需逐个检查，只有通过既有人工真值门禁后才能成为参考帧；
- 未自动修改人工真值、训练集、模型权重或历史候选。

部署前确认工作台 `status=idle`，没有在途任务。仅短暂停止本机页面和隧道，远端 SAM/VLM 保持热启动。生产文件逐个备份到：

`D:\Apps\LabelAgent\.local\backups\round272-ego-fallback-predeploy-20261120`

发布文件为：

- `source/agent/workflow.py`
- `source/agent/server.py`
- `source/web/app.js`
- `.local/manage_remote.py`

四个文件部署后 SHA-256 均与工作树源文件一致；旧版本备份哈希与新版本哈希均记录在结构化证据中。

## 4. 测试与上线后读回

- ego 补救、管理器和新数据工具定向测试：36 项通过；
- 正式 `PYTHONUTF8=1 pytest -q tests`：1,057 项通过，另有 186 个子测试通过；
- 全部前端 Node 测试：142 项通过；
- `node --check source/web/app.js` 通过；
- `git diff --check` 通过；
- 上线后首页 HTTP 200，运行状态 `idle`；
- 4090 SAM 点服务 `/health`：`model_loaded=true`；
- 主 VLM `/v1/models`：`labelagent-vlm8b` 在线；
- `/api/bootstrap` 读回 `ego_bottom_point_fallback=true`，点服务端口为 28766；
- 原 CVAT Task3 数据路径、相机 topic、150 帧候选状态及唯一人工真值帧 44 保持不变。

## 5. 证据与剩余风险

本机图像证据：

- `artifacts/new-dataset-smoke-round272-ego-fallback/before-after-contact-sheet.jpg`
- `artifacts/new-dataset-smoke-round272-ego-fallback/003/agent-overlay.png`
- `artifacts/new-dataset-smoke-round272-ego-fallback/004/agent-overlay.png`
- `artifacts/ego-bottom-point-round272-real-negative-review/four-camera-sample-contact-sheet.jpg`

结构化无图证据：

- `docs/evidence/workbench-stability-20261117/new-dataset-ego-fallback-round272.json`

当前主要运行风险仍是本机空间。上线后最低点为 C 盘约 0.256 GiB、D 盘约 0.413 GiB。已对旧 Mask2Former 恢复实验做可逆 NTFS 压缩，文件路径与内容均保留，但模型权重和图片本身压缩率低，只释放约 40.96 MiB。随后又原位压缩了 68 个已过期的远程桌面 ETL 诊断文件、29 个已停止的隔离验收目录和当前 artifacts 证据目录，分别释放约 487.68、86.51、236.95 MiB。最终读回约为 C 盘 0.250 GiB、D 盘 1.200 GiB，工作台仍为 HTTP 200、idle，SAM 与 VLM 正常；对照拼图 SHA-256 压缩前后保持 `DB196AFC...C1F1`。没有删除任何旧实验、人工真值、训练数据或证据。

因此，当前版本适合单帧演示与小范围候选生成；虽然 D 盘已恢复到约 1.2 GiB，但 C 盘仍极低，在未进一步释放可恢复空间前，不应发起大批量下载、训练或全量推理。

## 6. 下一步

1. 对 D 盘旧隔离运行目录做逐项可恢复归档，先释放至少 3–5 GiB；
2. 收集真正无宿主结构的跨设备负例和侧向、严重遮挡正例，独立验证门禁；
3. 在不污染人工真值的前提下，让新数据通过工作台参考帧页面完成一次人工验收；
4. 继续优化托盘凹凸边界，但保留原始 SAM 掩膜和一键回滚，不以视觉平滑牺牲孔洞或叉齿结构。

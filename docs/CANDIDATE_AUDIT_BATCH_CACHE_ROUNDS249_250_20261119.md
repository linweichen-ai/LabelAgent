# LabelAgent 多候选盲审与批处理缓存优化（Round 249–250）

时间：2026-11-19 15:06（Asia/Shanghai）

## 1. 问题与原则

Round 248 的真实新数据复跑包含 5 个高风险人物候选。旧实现只在“刚好 3 个”候选时使用独立批处理，5 个候选因此全部逐个调用 VLM，冷路径长尾明显。

历史受控实验证明：3 个独立请求共同批处理可保持语义并平均节省约 28%；2 个批处理没有稳定收益。因此本轮没有把多个候选拼成一个提示，也没有修改类别判定门槛，而是把长度大于 3 的列表拆成完整三元组，剩余 1～2 个继续走原逐个审。

所有盲审只产生候选审计证据，不修改候选类别、不写人工真值、不批准训练或部署。

## 2. Round 249：完整三元组调度

5 个候选现在按以下顺序执行：

1. 前 3 个使用已验证的独立 batch3；
2. 剩余 2 个继续逐个审；
3. 三元组中任一输出不合规、低置信、不一致或服务不可用时，只丢弃该三元组的批量结果并逐个重跑；
4. 前面已经通过的三元组不受后续失败影响，候选顺序和进度编号保持不变。

部署后的同图隔离实测确认 5 个候选执行模式为：

`independent_batch_3 × 3 + sequential × 2`

5 个审计均为 `candidate_preserved`。该次运行耗时 28.66 秒，比旧的全缓存 7.94 秒慢，进一步定位到批处理端点没有复用单请求响应缓存：批处理三元组被重新推理，而其他 4 个请求命中缓存。

## 3. Round 250：批处理逐请求精确缓存

主 VLM 的批处理端点现在以每个独立请求的完整请求体计算与单请求接口相同的 SHA-256 缓存键：

- 三项均命中时直接按原顺序返回，不占用推理槽；
- 部分命中时，只计算未命中项；
- 等待共享模型槽后会再次检查缓存，避免等待期间已有相同请求完成却重复计算；
- 只有一个未命中项时走单请求生成，2～3 个未命中项才走真正独立批处理；
- 批处理生成的每一项也分别写入原单请求缓存，后续单请求可直接复用；
- 响应继续报告命中数、生成数、每项缓存键和批处理位置，不存提示词或图像到仓库。

## 4. 同一真实新数据冷/热验收

图像：`Capture_20261103/source/cameras/camera_1/1789897506782450197.png`，1920×1536。

| 运行 | 总耗时 | VLM 缓存 | 候选 | 审计模式 |
|---|---:|---|---|---|
| Round 243 旧冷基线 | 85.48 秒 | 旧冷路径 | 17 | 5 个逐个审 |
| Round 250 新冷路径 | 70.73 秒 | 0 hit / 7 miss / 7 store | 17 | 3 个批处理 + 2 个逐个审 |
| Round 250 紧接热路径 | 7.84 秒 | 7 hit / 0 miss / 0 store | 17 | 3 个批处理 + 2 个逐个审 |

与旧冷基线相比，新冷路径缩短 14.74 秒，约 17.25%。热路径相对本轮冷路径缩短约 88.92%。冷/热两次均满足：

- 17 个候选：1 floor、11 pallet、5 person；
- 几何 SHA-256 均为 `4d9d9f1adeefbda85d369d06559b139209d1e50f1b945b3ef902978efe436fa6`；
- 5 个高风险候选全部 `candidate_preserved`；
- Agent 一轮、`no_progress`；
- busy rejection、coalesce timeout、复用失败均为 0；
- Round 248 的地面大矩形继续被 `unreferenced_global_surface_large_add_requires_human_review` 拦截；
- `ego` 可见但没有候选的问题仍明确保留，没有用错误地面扩张掩盖。

这些数据证明同一张真实图上的调度与精确缓存生效；它不是跨数据集精度结论，也不能保证每张新图都固定提速 17.25%。

## 5. 测试、部署与回滚

- Python：1011 passed，另有 186 subtests passed；
- 网页：145/145 passed；
- Round 249 工作台审查文件 SHA-256：`2FCC6FD5092568DBF00CAC926CEA3CB73A3F56E2C7D74E757A943B576D42B1B7`；
- Round 250 4090 VLM 服务文件 SHA-256：`2EEA52D7D7FF4A8717AC1937053C89921FDD464EB145BE8D4F6265630D2D6880`；
- 本机回滚：`D:\Apps\LabelAgent\.local\backups\candidate-audit-full-triples-round249-20261119-145317`；
- 4090 回滚：`/home/clw/labelagent_remote_20261110/backups/batch-cache-round250-20261119-150131`；
- 主 VLM 重启后模型正常加载、inflight 0；工作台、SAM3、主 VLM、安全通道均正常；
- 实现提交：`a651d16`（完整三元组调度）、`6fc2866`（批处理精确缓存）。

本地真实图像和候选产物不推送远端：

- `D:\Apps\LabelAgent\.local\relocated-worktrees\labelagent_stability_worktree_20261117-artifacts\new-dataset-smoke-round250-cold\`；
- `D:\Apps\LabelAgent\.local\relocated-worktrees\labelagent_stability_worktree_20261117-artifacts\new-dataset-smoke-round250-warm\`。

## 6. 下一步

当前冷路径仍约 70.7 秒，主要剩余项是可见类别发现、Agent 复查和两个人物顺序盲审。下一步优先解决 `ego` 可见但无候选的召回问题；速度优化继续以“结果几何一致、审计路由不放宽、人工真值不自动写入”为门禁。

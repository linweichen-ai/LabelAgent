# LabelAgent 栈板边界继续优化（Round 244–247）

时间：2026-11-19 14:27（Asia/Shanghai）

## 1. 用户反馈与判断

用户给出的近景栈板结果已经识别到正确实例，但木纹、污渍和反光被 SAM 候选当成大量小孔，轮廓因此呈现“凹凹凸凸”的观感。问题重点不是重新识别栈板，而是在不填死叉孔和板缝的前提下整理候选几何。

本轮所有输出仍是机器候选；未覆盖人工真值，未自动批准进入训练集。

## 2. 新的两级清理规则

在 Round 242 的小孔清理上继续增加受限规则：

- 原微孔规则保持不变：面积不超过 96 像素、跨度不超过 32 像素；
- 新增短跨度纹理孔规则：面积不超过 256 像素、跨度不超过 24 像素；
- 面积不少于 50,000 像素的近景大栈板使用 7×7 中值边界提案；
- 平滑后只再清一次不超过原 96 像素门槛的残余微孔，不对中等孔洞做二次扩张；
- 连通组件数量必须保持，整体变化不得超过候选面积的 2.5%；
- 长板缝、叉孔和实例边界继续保留；
- 每项修改写入 `candidate_geometry_adjustments`，并标记 `candidate_only`。

## 3. 第 120 帧实测效果

Round 242 后，主栈板仍有 11 个内部孔洞。数值分析显示，其中 7 个是面积 97–167 像素、跨度不超过 23 像素的短小纹理洞；另外 4 个是长度 31–90 像素的连续板缝或叉孔。

最终生产在线任务 `4a4ed3794b444e0b85e451390a25d272`：

- 状态：completed；
- 服务端墙钟时间：25.35 秒；
- 候选：12 个；
- Agent：1 轮，`no_progress` 安全停止；
- 主栈板 shape 9 保持 `pallet`；
- 共清理 59 个孔洞、2,249 个孔洞像素；
- 其中新增短跨度纹理孔 5 个、752 像素；
- 平滑后二次清理微孔 2 个、159 像素；
- 7×7 平滑改动 437 像素；
- 总改动 2,668 像素，占栈板面积 2.1275%；
- 最终只保留 4 个长结构开口：90×11、77×10、32×11、41×9；
- 人工真值未修改，训练批准仍为 false。

本地视觉证据：

- `artifacts/pallet-geometry-round244/frame-120-round242-vs-round244.jpg`；
- `artifacts/pallet-geometry-round247/frame-120-production-final.jpg`；
- `artifacts/pallet-geometry-round247/frame-120-production-final-crop.jpg`。

图中含工厂原始画面，因此只保留在本地验收目录，不推送 GitHub/CNB。

## 4. 在线复测中发现并修复的两项问题

### 4.1 错误把栈板整体改成地面

第一次生产复测任务 `7fa3c5e490bc4802b99fbe1dd267752e` 中，几何清理正确生效，但 VLM 将明显的近景大栈板 shape 9 从 `pallet` 错改为 `floor`，并因此多跑一轮，总耗时 55.45 秒。

新增硬门禁后，实例候选不能由一次模型复查整体改为 `floor` 或 `slope`；系统保留原候选并路由人工复核。任务 `1bd7061463e34d5d82a9601e2279428d` 读回确认 shape 9 仍为 `pallet`，错误操作被记录为 `instance_to_global_surface_requires_human_review`。

### 4.2 结构化解码约束无解

部署平滑后二次清理后，任务 `eb935f682c2d47059ee10e1e709cc9c7` 和 `329baecc01e84304a11674e0102f7de1` 可重复遇到 HTTP 400：`prefix_allowed_tokens_fn returned an empty list`。两次失败均按保护策略保留上一版候选，没有写坏状态。

新增精确恢复逻辑：只有确认是“结构化约束无解”这一种 400 时，自动重试一次普通 JSON；结果仍必须通过 JSON 解析和原 Agent 协议校验，不能绕过安全门。最终任务 `4a4ed3794b444e0b85e451390a25d272` 通过，trace 明确记录 `structured_fallback_reason=decoder_constraint_unsatisfiable`。

## 5. 回归、部署与回滚

- Python：1002 passed，另有 186 subtests passed；
- 网页：145/145 passed；
- 工作台：页面、SAM3、主 VLM、安全通道均正常；
- 生产 `workflow.py` SHA-256：`9FE17F65BCF308A7FF424D8DFE4660721C97B922003381A0D46D14EF80361DB0`；
- 生产 `edit_agent.py` SHA-256：`48C6826A89E277CCC961FA70166B809E2778895964877EB2DFB966EF43A6FDE3`。

回滚备份：

- `D:\Apps\LabelAgent\.local\backups\pallet-compact-texture-round244-20261119-140814`；
- `D:\Apps\LabelAgent\.local\backups\instance-surface-guard-round245-20261119-141430`；
- `D:\Apps\LabelAgent\.local\backups\pallet-post-smoothing-round246-20261119-141806`；
- `D:\Apps\LabelAgent\.local\backups\structured-decoder-recovery-round247-20261119-142407`。

实现提交：

- `4d64209`：短跨度纹理孔与近景栈板平滑；
- `a8d92e4`：实例整体改地面/斜坡的人审门禁；
- `67516d3`：平滑后残余微孔二次清理；
- `46f16a2`：结构化解码无解的受限恢复。

## 6. 当前限制

- 最终结果是同一数据集同一帧的生产在线验收，证明链路生效，但不能替代跨数据集人工真值精度评测；
- 4 个长结构开口是按几何规则保留的候选，需要人工确认其语义是否全部正确；
- 5090 SAM 通道正常，5090 VLM 可选加速通道当前仍未启用，主 VLM 使用 4090；
- C、D 盘均低于 2 GiB，暂不应导入大批量新数据或生成大规模中间产物。

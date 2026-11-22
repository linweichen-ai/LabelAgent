# Ego 组件全覆盖尾部复核策略（Round484）

日期：2026-11-22

状态：最新离线人工复核顺序；默认关闭，不是自动验收或人工真值

## 改进点

Round479/483 按 camera1 排名末 20% 整段延后，共 19 项。两轮互不重叠的盲审合并后，
这 19 项已经全部被视觉复核：18 项为 `likely_fill_ok`，1 项仍为 `uncertain`，0 项为
`likely_reject`。

Round484 因此不再只看排名，而是要求：

1. 至少两轮盲审且组件 ID 互不重叠；
2. camera1 尾部 19 项必须 100% 被盲审覆盖；
3. 只有判断为 `likely_fill_ok` 的项才能进入 `deferred_manual_review`；
4. `uncertain` 或 `likely_reject` 必须重新进入 `priority_review`；
5. 任一证据、排名或绑定异常都失败关闭。

## 最新结果

| 队列 | 数量 | 变化 |
| --- | ---: | --- |
| `priority_review` | 227 | 比 Round483 增加 1 个不确定项 |
| `deferred_manual_review` | 18 | 仅保留已盲审为较安全的尾部项 |
| 自动接受/拒绝 | 0 | 始终禁止 |

被重新提回优先复核的组件为：

`004-camera1-idx5-1793760663542839987-c002`

它的 camera1 风险排名为 77，盲审理由为“暗色横向机械结构模糊”。它不是已确认错误，
但证据不足，因此不能延后。

## 产物与边界

- 最新策略：`artifacts/round484-1104-ego-component-manual-review-policy-v3-20261122.json`
- SHA-256：`52e1e4ac4a96f989ecd08c7d96f60c2c735c975edd76f519a06b4d086f1ad0ad`
- 输入合并盲审 SHA-256：
  `5f29b4580dd2fe3377f31ce60d7b8f84f3f7817ab3e3fe20f5aaf7712d1ba24a`

“延后”仍不等于“接受”。这 18 项以后仍要人工确认；本轮没有写入真值、批准训练、修改
候选 mask、模型权重、参考帧或正式工作台配置。

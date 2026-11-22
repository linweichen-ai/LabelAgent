# Ego 组件人工复核策略固化（Round479）

日期：2026-11-22

状态：离线人工复核顺序；默认关闭，不是自动验收、人工真值或新数据泛化证明

## 为什么做

Round476–478 的盲审显示：camera1 的低风险尾部明显更安全，但仍有不确定项；camera3
即使低风险区也很难。为了把这一结论变成可复现的操作顺序，本轮只生成一份离线策略，
不接入正式工作台，也不修改候选 mask。

## 固化规则

- camera3：154 个组件全部进入 `priority_review`；
- camera1：排名前 72 个进入 `priority_review`，最后 19 个进入
  `deferred_manual_review`；
- `deferred_manual_review` 只表示延后查看，不表示接受；
- 所有 245 个组件的 `automatic_decision` 都为 `null`；
- 输入缺失、相机排名不连续、计数不一致、出现自动决策或盲审门禁不满足时，工具失败关闭。

最终队列为：

| 队列 | 数量 | 含义 |
| --- | ---: | --- |
| `priority_review` | 226 | 优先人工/助手复核 |
| `deferred_manual_review` | 19 | 可延后，但仍必须人工复核 |
| 自动接受/拒绝 | 0 | 明确禁止 |

## 可追溯证据

- 输入风险排序 SHA-256：
  `23a6e15bd884e4fcc955019fa48c1fadcf2012d658f4c1aec33ff2d977cf8782`
- 输入盲审评估 SHA-256：
  `1ecb43e024730b3b7cab491056a26acdfe90cf65a0caa3a10c849304c3cbe7a6`
- Round479 策略：
  `artifacts/round479-1104-ego-component-manual-review-policy-20261122.json`
- 策略 SHA-256：
  `4b22306de7ec1bd5386e3a0279487ed22eed13677d1f750d38a9f42da2a64141`

## 边界

本策略来自同一批已曝光数据上的助手盲审，只能减少“先看什么”的随机性。camera1 尾部
仍有不确定样本，camera3 全部仍难；因此本轮没有自动接受、没有写入真值、没有批准训练，
也没有改变模型、参考帧或正式工作台行为。新数据到来后必须重新盲审校准。

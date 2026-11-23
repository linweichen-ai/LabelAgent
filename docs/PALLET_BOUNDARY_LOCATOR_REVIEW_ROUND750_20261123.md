# Round750：组件定位框复核负结果

## 结论

在双尺度候选上下文中加入醒目的洋红定位框后，两位全新隔离助手只在 4/11 个组件上完全一致，
Cohen's kappa 为 0.0253。由于 Round749 与 Round750 的审阅者不相交，不能把数值变化归因于
定位框；本轮只能报告没有观察到足以支持继续依赖助手盲审的一致性证据。

更关键的是，一位审阅者在理由中把待审区域描述为 dog、car 和 metal structure，与本轮明确的
pallet 目标审阅契约不一致。定位框解决了“组件在哪里”，但没有解决“当前审的是哪一个实例、
原 mask 覆盖了什么”的所有权问题，因此本轮标签不进入任何自动放行、训练或人工真值。

## 逐组件结果

| 帧/候选/组件 | 助手 A | 助手 B | 一致 |
|---|---|---|---|
| 15/3/1 | reject | accept | 否 |
| 95/4/1 | accept | reject | 否 |
| 22/4/1 | accept | accept | accept |
| 72/8/1 | accept | reject | 否 |
| 142/8/1 | reject | reject | reject |
| 149/16/1 | accept | accept | accept |
| 149/16/2 | accept | accept | accept |
| 149/8/1 | uncertain | reject | 否 |
| 149/8/2 | accept | reject | 否 |
| 149/8/3 | accept | uncertain | 否 |
| 149/8/4 | uncertain | reject | 否 |

这 4 个一致项仍然只是助手共识，不是正确性标签；其余 7 项保持 unresolved。

## 工具与复现

- 定位框只画在审阅图的候选上下文 pane，不修改 source、before、after 或生产 mask。
- locator 必须依赖 component context，component context 又必须依赖 component sheets；缺少上游
  开关即失败关闭。定位框绝对坐标写入私有 manifest，方便复核。
- Round743/744 locator 包 manifest SHA 分别为
  `5703173130036a07cfc8112af1eee5a3aee210a3a08e55eb615614f0a4ebf5f7` 和
  `654b7cf6d75df40285cd9fd871a5353b2e0c98bedce1180e6c31725be25676d8`；独立重跑分别
  8/8、25/25 文件 SHA 完全一致。
- 盲审协议与 22 个原始标签收据 SHA 为
  `0b7bda806b57e2c878930edce7dc18e23c569f2ff17bd544c10d3ed794bf4a65`；收据还逐项绑定
  三个语义契约误读摘录与对应审阅者、帧、候选和组件。
- 固定 UTF-8 环境定向 10 项通过；正式 `tests/` 全量 1784 项通过、1 项跳过、286 子测试通过。

## 下一步

不再继续单独微调定位框。下一版审阅图同时展示：原始候选上下文、带定位框的原 mask 覆盖、
目标类别 `pallet`、局部 before/after/addition。随后优先让本轮同一审阅者做固定顺序配对复核，
并明确记录记忆与顺序偏差；若仍不能稳定，就停止助手门禁探索，转入真正人工复核。

# Round747：同审阅者低清→高清配对复核

## 结论

让最初只看过低清联系表的两位助手，在不读取梯度、策略和另一位结果的情况下，再看同一批
7 个高分辨率细节图。两人在高清阶段 5/7 完全一致，Cohen's kappa 为 0.4815；相对各自的
低清判断，审阅者 A 改变 5/7，审阅者 B 改变 4/7。

这是固定顺序“低清→高清”的配对复核，存在记忆、学习和顺序效应，不能证明变化由分辨率
单独造成。所有结果仍是助手视觉意见，不是人工真值。

## 配对结果

| 帧 / 候选 | A 低清 | A 高清 | B 低清 | B 高清 | 高清两人一致 |
|---|---|---|---|---|---|
| 15 / 3 | reject | accept | accept | accept | accept |
| 95 / 4 | uncertain | reject | reject | reject | reject |
| 22 / 4 | uncertain | reject | accept | accept | 否 |
| 72 / 8 | accept | accept | accept | uncertain | 否 |
| 142 / 8 | accept | reject | accept | reject | reject |
| 149 / 16 | accept | accept | uncertain | accept | accept |
| 149 / 8 | uncertain | accept | uncertain | accept | accept |

## 四位高清助手汇总

合并 Round746 的两位全新高清盲审者与本轮两位配对审阅者：

- 15/3：4 accept；
- 95/4：3 reject、1 uncertain；
- 22/4：2 accept、1 reject、1 uncertain；
- 72/8：3 accept、1 uncertain；
- 142/8：4 reject；
- 149/16：4 accept；
- 149/8：2 accept、1 reject、1 uncertain。

按“至少 3/4 同标签且没有相反标签”的保守助手共识规则，得到 3 accept、2 reject、2 unresolved。
在这 5 个共识项上，absolute64 与 4/5 一致，其余三种梯度策略为 3/5。这仍不是 accuracy：
同一序列、样本极少，且四位都是助手，其中两位已先看过低清图。

## 决策

梯度单一规则无法同时满足现有助手共识：absolute64 会放行 95/4，而相对梯度规则会拒绝
15/3。下一步不继续微调单一梯度阈值，而是对 22/4 与 149/8 生成逐补点细节图，并把局部
几何连续性、颜色/纹理归属与梯度联合成审计特征；在真正人工真值到位前仍不部署。

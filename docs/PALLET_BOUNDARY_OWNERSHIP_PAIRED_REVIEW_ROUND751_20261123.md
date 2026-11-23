# Round751：候选所有权覆盖的同审阅者配对复核

## 结论

在候选上下文中同时展示原图、`pallet` before-mask 覆盖和组件定位框后，让 Round750 的同两位
助手按固定顺序再次审阅同 11 个组件。两人完全一致从 4/11 增至 8/11，Cohen's kappa 从
0.0253 增至 0.4762；审阅者 A 改变 2 项，审阅者 B 改变 6 项。

这是 locator-only → ownership-overlay 的固定顺序配对，不是随机交叉实验，存在记忆、学习、顺序
和 mask 锚定偏差。因此只能说“当前助手审阅在所有权信息下更一致”，不能说准确率提高，也不能
证明 overlay 本身产生了因果改善。所有标签仍是助手意见，不是人工真值。

## 配对结果

| 帧/候选/组件 | A locator | A ownership | B locator | B ownership | ownership 一致 |
|---|---|---|---|---|---|
| 15/3/1 | reject | reject | accept | reject | reject |
| 95/4/1 | accept | accept | reject | uncertain | 否 |
| 22/4/1 | accept | accept | accept | accept | accept |
| 72/8/1 | accept | accept | reject | accept | accept |
| 142/8/1 | reject | reject | reject | reject | reject |
| 149/16/1 | accept | accept | accept | accept | accept |
| 149/16/2 | accept | accept | accept | accept | accept |
| 149/8/1 | uncertain | accept | reject | accept | accept |
| 149/8/2 | accept | accept | reject | accept | accept |
| 149/8/3 | accept | accept | uncertain | uncertain | 否 |
| 149/8/4 | uncertain | accept | reject | uncertain | 否 |

8 个一致项包含 6 个 accept 和 2 个 reject，但这不是 6 个正确补点和 2 个错误补点；没有人工
真值时只能作为人工复核排序线索。审阅者 B 仍在一条理由中使用“狗体”描述，说明即使明确展示
所有权信息，底层语义误读仍未消除。

## 工具与复现

- 新增可选 ownership pane：候选上下文上的绿色 before-mask 覆盖、洋红 locator 和明确的
  `pallet before mask + locator` 标题；与原始上下文、局部 source/before/after/addition 同屏，
  输出为 3840×480 无损 PNG。
- ownership 必须依赖 locator，locator 必须依赖 context，context 必须依赖 components；任一
  上游开关缺失即失败关闭。默认所有新开关仍为 false，旧输出不变。
- Round743/744 ownership 包 manifest SHA 分别为
  `a8be64f8fd9c43fd208cb72561c667efac95ff93c46bdf2af8ff570c9c4d1d02` 和
  `6473e8b35299ee546b344c65d3ffaf6f6678549c6ab7b55264fbbedd540bd4fb`；独立重跑分别
  8/8、25/25 文件 SHA 完全一致。
- 配对协议、22 个 ownership 标签和语义误读摘录收据 SHA 为
  `3bcf891abe72f5897a096374d7469a5d62769618f460fc4198e95258060cedb3`。
- 固定 UTF-8 环境定向 11 项通过；正式 `tests/` 全量 1785 项通过、1 项跳过、286 子测试通过。

## 决策

ownership sheet 是当前最适合作为人工/助手复核界面的候选格式，但不作为自动门禁。下一步应把
这套只读图接入候选复核工作台，保留 accept/reject/uncertain 与审阅身份收据；只有真正人工确认
的标签才能回流训练或决定生产策略。

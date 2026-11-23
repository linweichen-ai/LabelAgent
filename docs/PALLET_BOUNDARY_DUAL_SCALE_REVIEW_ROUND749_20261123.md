# Round749：栈板组件双尺度上下文复核

## 结论

把真实数据上的 11 个候选补点组件统一放到两种审阅图中：第一组只看局部 96 像素级裁剪，第二组同时看
整个候选上下文与局部细节。两组各由两位互相隔离的助手审阅，组内都只有 5/11 完全一致，
Cohen's kappa 都是 0.1951。在这两个不相交助手组的本次观测中，双尺度组未观察到更高的
组内一致率；由于审阅者不同，不能作视图因果比较，也不能宣称审阅质量提升。

但两种视图得到的明确共识内容差异很大：共同明确的只有 142/8/1 的 reject；95/4/1 从局部组
一致 accept 变为双尺度组一致 reject，149/8/3 则从一致 reject 变为一致 accept。两组审阅者
不是同一批人，所以这些反转不能归因于上下文的单一因果效应。局部组在当前小样本中一致性
较低，而且两组共识内容不同；这只提示局部图可能不足，仍需同一审阅者配对或随机顺序实验验证。

## 逐组件标签

| 帧/候选/组件 | 局部 A | 局部 B | 双尺度 A | 双尺度 B |
|---|---|---|---|---|
| 15/3/1 | accept | uncertain | accept | accept |
| 95/4/1 | accept | accept | reject | reject |
| 22/4/1 | uncertain | accept | accept | uncertain |
| 72/8/1 | accept | accept | uncertain | reject |
| 142/8/1 | reject | reject | reject | reject |
| 149/16/1 | accept | uncertain | accept | accept |
| 149/16/2 | reject | reject | accept | uncertain |
| 149/8/1 | accept | uncertain | reject | accept |
| 149/8/2 | uncertain | reject | reject | uncertain |
| 149/8/3 | reject | reject | accept | accept |
| 149/8/4 | uncertain | reject | reject | uncertain |

这里的 A/B 只是每组内的匿名助手，不表示两组是同一审阅者配对实验。全部标签仍是助手意见，
不是人工真值，也不能计算真实准确率。

## 工具与复现

- 组件审阅生成器新增可选双尺度图：3200×480 无损 PNG，依次展示候选上下文、局部原图、
  局部 before、局部 after 和新增像素；原有 2560×480 局部图保持不变。
- `include_component_context` 必须与组件图同时启用，否则失败关闭；路径、SHA 二读、add-only、
  像素核算和组件连通门保持不变。
- Round743 双尺度包为 2 个组件、8 个文件，manifest SHA
  `4cd0471dad160a5a98bcc6a129328e072f0227306d5810731b4246d5a6ab89bb`。
- Round744 双尺度包为 9 个组件、25 个文件，manifest SHA
  `93fb11d4c6013a33a3693911ac9caf6506edc68081a8b2eece95e8d39c63535f`。
- 两批独立重跑均逐文件 SHA 完全一致；盲审协议与 44 个原始标签收据 SHA 为
  `75d179a6f8f764d34207fce108b2fe735a66e39a334d3834fc98461bde5153bd`。
- 固定 UTF-8 环境定向 9 项通过；正式 `tests/` 全量 1783 项通过、1 项跳过、286 个子测试通过。

## 决策

当前双尺度图仍缺一个关键线索：候选上下文 pane 没有明确标出微小组件在全局图中的位置。
下一轮只优化审阅证据，在上下文图增加不遮挡目标的定位框，再由全新助手复核；生产 mask、
人工真值、训练数据和模型均保持不变。

# Round746：栈板边界高分辨率盲复核包

## 结论

旧联系表把每个候选缩到固定 360×300 面板，几到几十个新增像素难以辨认。两位未接触梯度
和策略结果的助手在旧联系表上只有 3/7 完全一致，Cohen's kappa 为 0.0667。新增无损 PNG
局部四联图后，另外两位全新盲审助手达到 4/7 完全一致，kappa 为 0.3226，并形成
3 个一致接受、1 个一致拒绝、3 个未解决候选。

这个结果与“更清晰的局部图可能改善判断”一致，但两轮审阅者不同，审阅者差异与图像分辨率
完全混杂，不能证明一致率变化由细节图造成。仍需同一审阅者配对或随机顺序实验；现有结果也
不是人工真值，不足以选择生产阈值。

## 高分辨率盲审结果

| 帧 / 候选 | 盲审 A | 盲审 B | 助手共识 |
|---|---|---|---|
| 15 / 3 | accept | accept | accept |
| 95 / 4 | reject | uncertain | unresolved |
| 22 / 4 | uncertain | accept | unresolved |
| 72 / 8 | accept | accept | accept |
| 142 / 8 | reject | reject | reject |
| 149 / 16 | accept | accept | accept |
| 149 / 8 | uncertain | reject | unresolved |

在 4 个双方明确一致的助手标签上，absolute64 与 4/4 一致，其余三种策略为 3/4。但这个
对比样本极少、来自同一序列，而且标签仍是助手视觉判断，不能称为 accuracy、不能据此部署。

## 工具改进

- 保留原始联系表，新增每候选一张 2560×480 无损 PNG 四联图；
- 细节框围绕实际新增像素，最小 128×128 上下文，再放大显示 source、before、after 和 added；
- 强制 before 是 after 的子集，实际新增像素必须与 manifest 一致；
- 重复的 frame/candidate 身份失败关闭，不能覆盖同名细节图或重复计数；
- source/before/after 均做 SHA 二读，basename 还要解析真实路径并限制在输入根目录内；
- 符号链接/重解析点逃逸、删除像素和哈希漂移均失败关闭；
- 原图和细节图只保留在私有 `.repro`，不提交 GitHub/CNB。

## 可复现证据

- Round743 细节 manifest 双跑 SHA：`63d01cd1e4d942f695d681fb39092d45250a85d60ba9f878e09d492dfca6c197`；
- Round744 细节 manifest 双跑 SHA：`79de8588aac4a6e002a7709997d2195f5a86c6c5bc2dc772f10f325c38e8e52e`；
- 原联系表 SHA 保持不变，说明新增细节图没有改变旧输出；
- 定向组合测试 14 项通过；固定 UTF-8 全量回归
  `1779 passed, 1 skipped, 286 subtests passed`；
- 正式候选 mask、人工真值、训练集、模型权重和部署均未修改。

## 下一步

先用同一审阅者做配对、随机顺序的低清/高清复核，隔离审阅者差异；再把 3 个 unresolved
候选做逐补点而非整候选复核，并优先获取真正的人工像素判断。只有人工
复核覆盖至少 accept/reject 两类后，才计算策略 precision/recall；当前 absolute64 仍只是
诊断候选，不进入正式 v8 门。

# Round745：栈板边界梯度策略只读对比

## 结论

新增一个失败关闭的离线对比器，对 Round743 与 Round744 的 7 个真实触发候选重新校验输入
manifest、原始/结果 mask SHA、add-only 像素差、v8 审计字段和逐补点梯度，然后比较四种候选门。
工具不改 mask、不访问工作台、不写真值，也不选择或部署策略。

| 诊断策略 | 定义（一个候选的所有补点都必须通过） | 放行 | 拒绝 |
|---|---|---:|---:|
| relative0 | 缺口梯度 ≤ 邻域梯度 | 3 | 4 |
| relative+8 | 缺口梯度 ≤ 邻域梯度 + 8 | 4 | 3 |
| absolute64 | 缺口梯度 ≤ 64 | 6 | 1 |
| combined | 同时满足 relative+8 和 absolute64 | 4 | 3 |

这些计数只反映规则松紧，不反映准确率。7 个候选没有独立人工像素真值，且都来自同一正式序列，
所以不能从“放行更多”或“拒绝更多”推断哪种策略更好。

## 安全与可复现性

- 输入不是裸路径，而是带 SHA 的冻结 input-map；manifest 和 mask 在处理前后均重新读回验签；
- 每个 changed candidate 必须满足 before 是 after 的子集、实际新增像素与记录一致、零删除；
- 多补点候选采用 all-notches-pass，不能把一个补点通过夸成整个候选通过；
- 缺少梯度证据的候选单列 insufficient，不混入放行或拒绝；本轮 insufficient 为 0；
- 输出不包含绝对路径、原图、mask 或像素坐标；truth/training/deployment/workbench/mask write 均为 false；
- 双跑 report SHA 均为 `5e8f25db56c8bceb233c0018810b8ed5115793c71f8d5039cde31c9f9bc5fd71`；
- 定向测试 9 项通过，覆盖策略边界、多补点 all-pass、缺证据、安全标志、非有限梯度、
  mask 删除/漂移、实际组件桥接、严格布尔、符号链接逃逸、跨 smoke 重复候选身份和已有输出
  失败关闭。
- 固定 UTF-8 全量回归 `1776 passed, 1 skipped, 286 subtests passed`。

## 下一步

把 7 个候选交给独立人工像素复核后，再计算四种策略的 precision/recall 和边界收益。人工结果
返回前，这四种策略都保持诊断状态，正式 v6、v7、v8 mask 行为不变。

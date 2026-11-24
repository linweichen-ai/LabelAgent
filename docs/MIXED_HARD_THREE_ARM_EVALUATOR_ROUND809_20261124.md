# Round809：困难集三路对照评测器

## 结论

本轮把“仅本地 Qwen8B、全部外部专家、混合路由”三路方案的对比逻辑固化为可复现命令行工具：

- `tools/evaluate_mixed_hard_three_arm.py`
- `tests/test_evaluate_mixed_hard_three_arm.py`

评测器自身 6 项测试通过；包含主 Agent、外部路由、缓存、熔断、回退和复核工具的扩展回归为 361 项测试及 87 项子测试全部通过。当前没有产生真实质量指标，因为 34 帧声明式人工复核尚未完成，且 34 帧外部专家结果尚未在新的明确授权下执行。工具会在这两类输入缺失、不完整或串集时拒绝输出，不能用模板、机器候选或旧结果伪装成人工基准。

## 评测范围

三路方案使用完全相同的 34 帧开发困难集，`test24` 不参与：

1. `local_only`：全部使用本地 Qwen8B 结果；
2. `all_external`：全部使用外部专家结果；
3. `hybrid`：先执行本地 Qwen8B，仅对 Round806 冻结的 17 帧困难队列使用外部专家结果。

输出包含类别级 precision、recall、F1、macro F1、斜坡召回与误检数、托盘实例数 MAE/完全匹配率、托盘质量准确率、平均/P95 延迟、外部调用数、调用比例以及可选成本估算。

这里的人工参考被明确标记为 `declared_human_review_not_pixel_truth`，只支持类别存在性、托盘数量和托盘质量的语义级比较；它不能证明像素级掩膜质量，也不能代替后续像素真值评测。

## 数据绑定与安全门

评测前强制检查：

- 人工复核为完整、无 `uncertain` 的 v2 声明式复核；
- 本地与外部结果均 34/34 成功且失败数为 0；
- 三方 sample ID 集合完全一致；
- manifest 哈希一致；
- 每一帧的原图 SHA-256 一致；
- 混合队列与本地建议文件 SHA-256 绑定；
- 所有输入继续保持 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`、`test24_used=false`；
- 输出固定 `deployment_gate_passed=false`，不得自动部署。

任何一项不满足都不会写出结果文件。使用当前未完成人工复核模板进行命令行实测时，程序以非零状态退出，错误为 `unsupported adjudication schema`，且目标输出在执行前后均不存在。

## 已冻结输入

- 困难集 manifest SHA-256：`4cd7eadb244cadad83208b4f2d9f304da457830b839070dd796edf77f53dad50`
- 本地 34 帧建议 SHA-256：`b0bff89db15a664d0e6bae1ce1a06b3a357b89c7efd3825b517e4a90051c8b1b`
- 混合 17 帧队列 SHA-256：`39e12f5f7fa5823ec925a364c0b630333a59e72dc9c8d3e0d1347320e012df9b`
- 全外部 34 帧队列 SHA-256：`63638ee8754b0b274562cf279e564f5fcbf6ba72b4af16a8c58c87fb589b782c`

## 验证结果

```text
361 passed, 87 subtests passed in 10.27s
```

本轮外部 API 调用 0 次，真实图片外发 0 张，新增 API 成本为 0。没有使用 `test24`，没有写入人工真值，没有批准训练，没有部署正式工作台。

## 后续执行条件

1. 人工在 Round803 离线复核页完成 34 帧声明式复核；
2. 获得本轮 34 张开发图片用于三路对照的明确外发授权；
3. 使用 Round807 的授权绑定执行器分别得到完整 34 帧外部结果；
4. 运行本评测器并进行独立读回；
5. 只有质量、误检、延迟、失败回退和成本门槛全部通过，才讨论正式部署。


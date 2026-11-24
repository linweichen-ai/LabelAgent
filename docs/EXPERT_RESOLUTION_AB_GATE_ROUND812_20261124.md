# Round812：真实困难帧分辨率 A/B 授权门与评测闭环

## 已完成

本轮将 Round811 的合成效率候选固化为受控真实开发集 A/B 流程，但没有执行任何真实图片外发。

实验范围被冻结为 Round806 的17张开发困难帧：

- `source_640x512`：使用哈希核验后的原始640×512字节；
- `resize_512x410`：使用固定 Lanczos 缩放与 JPEG quality=90；
- 每张图片最多上传两个变体；
- 授权上限为34次外部调用；
- 全程 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`；
- 不使用 `test24`，不自动写真值，不部署正式工作台。

## 授权门

已生成非授权模板：

- 路径：`artifacts/round812-expert-resolution-ab-authorization-template-20261124/authorization-template.json`
- SHA-256：`27c00a3c04b168d009f610a0902569ee0534bab2242024248b08cf40efff8032`
- 队列 SHA-256：`39e12f5f7fa5823ec925a364c0b630333a59e72dc9c8d3e0d1347320e012df9b`
- 当前状态：`authorized=false`
- 当前授权人：空

执行器会同时核对队列哈希、样本数、两个图像变体、上传上限、授权人和所有安全字段。旧版单分辨率授权不能被复用于缩放变体。原始分辨率不符640×512、转换参数漂移、图片哈希漂移或结果续跑绑定不同，均会拒绝执行。

使用当前未授权模板进行了真实命令行拦截检查：程序退出码为1，错误为 `matching explicit external-upload authorization is required`，结果文件执行前后均不存在，未发生网络请求。

## 结果可追溯性

每个成功结果会同时保存：

- 原图 SHA-256；
- 实际上传变体名称与转换方式；
- 实际上传图片 SHA-256；
- 源图与上传图尺寸；
- 模型结果、延迟、token用量、缓存命中和重试；
- 队列、授权、manifest 与 receipt 的哈希绑定。

新增 `tools/evaluate_expert_resolution_ab.py`，在人工声明式复核完整后比较两种尺寸的：

- 类别 precision、recall、F1与macro F1；
- 斜坡召回和普通地面误检；
- 托盘数量与质量；
- 平均/P95延迟；
- token用量与成本；
- 512×410相对640×512的指标差值。

该评测仍只覆盖语义存在性、托盘数量和质量，不声称像素掩膜质量；输出固定 `deployment_gate_passed=false`，仍需独立部署门评估。

## 验证

```text
374 passed, 87 subtests passed in 10.21s
```

本轮授权门开发阶段外部API调用0次、真实图片外发0张、未写真值、未使用test24、未部署。

## 所需明确授权

如需执行，请由用户原样或等价确认：

> 授权将Round806筛选的17张开发困难帧，以原始640×512和固定缩放512×410两个变体发送至已配置的Astra API，最多34次调用，仅用于candidate-only分辨率质量与效率A/B；不使用test24、不写人工真值、不批准进入训练集、不自动部署。

在获得这项明确授权前，模板保持 `authorized=false`，执行器不会外发真实图片。


# Round502：Task3 人工栈板实例候选包

## 本轮完成

在 Round501 证明 Task3 没有完整全人工帧之后，本轮没有强行做整帧训练，而是把其中 81 个 `source=manual` 的栈板轮廓整理成受控的“正实例候选包”。

- 81 个候选、81 帧、81 张图像全部重新核对文件 SHA-256；
- 候选 ID 全部唯一；
- 只纳入 `manual` 形状，自动形状纳入数为 0；
- test24 纳入数为 0；
- 每个候选都规定：只监督已标栈板实例，实例外像素全部为 `ignore`；
- 禁止整帧语义损失，禁止从未标区域采负样本；
- 当前状态为 `prepared_not_training_authorized`，仍需逐实例独立复核签字。

这样既保留了现有 81 个人工轮廓的价值，又不会因为同帧还有机器标注和未标区域而制造错误监督。

## 产物边界

包含轮廓坐标和图像关联信息的私有清单只保存在本机：

`.repro/cvat-task3-manual-instance-candidates-round502.json`

- 大小：4,469,262 字节；
- SHA-256：`1033e8ad16fc0e4c8b03ef246557ae321273b4188b92bc3171c9604558f32f99`；
- 没有上传到 GitHub/CNB，也没有复制原始图片。

公开仓库只同步不含轮廓坐标的统计回执：

`artifacts/round502-cvat-task3-manual-instance-candidates-20261122/receipt.json`

- 回执 SHA-256：`04d353e2b0620dad82b5e92c0d2e39a1f752e03511181973cabb7f9e4bad2f37`；
- `private_geometry_published=false`；
- `raw_images_published=false`；
- 私有清单哈希已写入回执，可在后续复核前检测任何变化。

## 可复现工具和验证

- 构建器：`tools/build_cvat_manual_instance_candidates.py`
- 测试：`tests/test_build_cvat_manual_instance_candidates.py`

3 项定向测试通过：

1. 混合帧只提取人工实例，自动形状不进入候选；
2. 图像哈希不一致时失败关闭；
3. 帧在导入清单中缺失时失败关闭。

实际包二次读取确认：81 个候选 ID 唯一、0 个非人工候选，私有文件实算哈希与公开回执一致。

## 下一步门槛

先为 81 个栈板实例生成只读聚焦复核视图，逐个确认“轮廓是否完整、是否为栈板、是否存在明显漏切/粘连”。复核结果必须绑定私有清单哈希；只有明确通过的实例才能进入偏标注微调。训练时仍必须把实例外区域设为 ignore，并在冻结评测集上与原始 SAM3 基线比较；没有通过质量和误检守门就不部署。

本轮未修改 CVAT、原始数据、候选真值、权重或生产工作台，未启动 GPU 训练。

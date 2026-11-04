# 软包含惩罚的固定对照

上一轮只读预测轮廓的诊断发现，置信度排序新增 54 个候选中有 29 个至少 95% 的像素位于原预测内；旧四分块为 14/47。包含碎片可能仍有助于区分融合实例，因此本组只改变软排序权重，不直接拒绝所有包含片段，也不改变原模型 score 或轮廓。

在原新区域优先项 `confidence*(1-max_mask_IoU)` 上乘以 `0.5+0.5*fraction_outside_pallet_union`。最小因子为 0.5；孔洞和多部分使用实际像素，不能把包围框内的所有区域当作已覆盖。静态版使用原始托盘预测并集，动态版在每次接受候选后更新并集与原新区域优先项。各比较是否拒绝与原人员预测相交的新增候选，共四个变体。

固定同样 20 张调参图、最低置信度 0.5、原 50% 追加预算和硬 NMS 0.8。保留旧四分块和旧多尺度排序作为两个原样重放对照。相对旧四分块，仍要求实例支持严格超过 75/118、点覆盖不低于 5179/6291、人员交叉覆盖不高于 55/11068，并通过相对原始基线的 +2 个百分点、候选数 1.5 倍、并集面积 1.25 倍、人员交叉覆盖增幅 0.5 个百分点等原门槛。

预测只读取图像及绑定源图的历史真实 SAM3 响应，完整导出后才读取三维参考计分。同一套 260 份真实响应复用，不计为新的模型调用。240 项完整回归通过，包括低 IoU 的包含碎片、碎片仍可被选中、孔洞实际像素、动态更新并集和预算保持。实际结果尚未评分，不能报告为改进；见 [固定范围及阶段](containment_progress.json)。

本组是同一采集的探索性调参，未新增独立人工二维真值，不将稀疏参考支持称为像素准确率。工作台默认流程、生产 schema 及原始标注保留。

```sh
python tools/replay_multiscale_candidates.py --dataset <local-dataset> --candidates-json <original-candidates.json> --prediction-receipt <bound-receipt.json> --study <four-crop-study> --study <nine-crop-study> --output <new-private-output> --containment-study
```

计分仍使用原 `tools/evaluate_projected_support.py`、同一批帧/相机和 0/3/6px 参数。未通过全部门槛不扩大到其余图，也不写入工作台候选。

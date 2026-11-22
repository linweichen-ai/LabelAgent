# Round 555：3D 深度路由的全样本触发范围审计

## 结论

本轮没有继续调高单个样本的分数，而是检查 Round 554 的深度拆分规则会不会误伤原本正常的图像。审计覆盖冻结的 010、011、012 三个序列、四路相机，共 12 张图。

- 12/12 张均完成读取，图像与冻结清单哈希一致；
- 10/12 张不存在达到阈值的 3D 托盘投影重叠，被明确跳过；
- 2/12 张触发深度路由，正好是 Round 550 已发现实例隔离不足的两张；
- 没有新增正常样本被送进拆分流程；
- 工作台写入、人工真值改写、训练回流、正式部署均为 0。

这说明当前触发条件是收敛的：它不会对全部托盘掩码一刀切，只会在目标与另一个投影框的重叠比例达到 25% 时进入深度/遮挡处理。

## 结果说明

在两张真正触发的图中，原始 SAM 独立掩码仍无法直接分开：严格二维分离为 0/2。加入 3D 深度后：

- 1 张得到可用的深度拆分候选；
- 1 张识别为远端托盘被近端托盘大面积遮挡，保留 3D 身份但不生成伪精确二维掩码；
- 两张均得到明确处理路径，但仍属于候选，未获得人工像素真值确认。

## 复现

新增 `--selection all_localized` 审计模式；默认的 `ambiguous` 行为保持不变。

```powershell
.\.venv\Scripts\python.exe tools\evaluate_pallet_box3d_overlap_pairs.py `
  --source-report artifacts\round551-cross-sequence-box3d-mask-archive-20261122\results.json `
  --manifest docs\evidence\workbench-stability-20261117\heldout-test12-manifest-round49.json `
  --labels-root .repro\round545-cross-sequence-box-roi\labels `
  --calibration <calib_640x512目录> `
  --output <新的空输出目录> `
  --selection all_localized
```

## 证据与边界

- 完整结果：`artifacts/round555-box3d-depth-full-conflict-audit-20261122/results.json`
- 收据：`artifacts/round555-box3d-depth-full-conflict-audit-20261122/receipt.json`
- 实现提交：`f45641f`
- 16 项相关单元测试通过。
- 该结果只能证明冻结 12 张图上的触发范围没有扩大，不能代替更大规模跨批次验证或人工边界验收。

## 下一步

保持当前实现为离线候选。获得对应 PCD 后，用逐像素 LiDAR 深度替代“3D 框中心深度”，再扩到更多序列做独立验证；在此之前不写回训练集和正式工作台。

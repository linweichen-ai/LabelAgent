# Round 556：托盘 3D 投影冲突阈值稳定性

## 通俗结论

当前规则不是“看起来有一点挨着就拆”，而是只有两个托盘的 3D 框投到图像后发生明显重叠，才进入深度拆分。

在冻结的 12 张跨序列图像中：

- 原本隔离正常的 10 张，最高重叠比例只有 **3.78%**；
- 原本隔离失败的 2 张，最低重叠比例也有 **65.00%**；
- 两组之间有 **61.23 个百分点**的安全间隔；
- 触发阈值从 10% 调到 50%，命中的始终只有同两张问题图。

因此当前 25% 阈值不是卡在临界点上，在这组冻结数据中具有较大余量。

## 方法

工具只读取冻结图像清单、三份标签 JSON 和标定文件，计算目标托盘与其他托盘 3D 框投影的重叠比例。它不请求 SAM/VLM，不占用 GPU，不修改工作台、人工真值或训练数据。

```powershell
.\.venv\Scripts\python.exe tools\audit_pallet_box3d_conflict_margin.py `
  --source-report artifacts\round551-cross-sequence-box3d-mask-archive-20261122\results.json `
  --manifest docs\evidence\workbench-stability-20261117\heldout-test12-manifest-round49.json `
  --labels-root .repro\round545-cross-sequence-box-roi\labels `
  --calibration <calib_640x512目录> `
  --output <新的results.json>
```

## 证据

- 实现：`tools/audit_pallet_box3d_conflict_margin.py`
- 测试：`tests/test_audit_pallet_box3d_conflict_margin.py`
- 结果：`artifacts/round556-box3d-conflict-margin-audit-20261122/results.json`
- 收据：`artifacts/round556-box3d-conflict-margin-audit-20261122/receipt.json`
- 实现提交：`df48236`
- 相关测试：18 项通过。

## 边界

这只能证明当前冻结的 3 个序列、12 张图上存在明显间隔，不能自动外推到所有厂区、光照和托盘堆叠方式。规则继续保持离线候选，不部署、不写回、不进入训练集。

下一步仍是取得对应 PCD 后，用逐像素 LiDAR 可见性验证深度拆分结果；如果没有 PCD，则先扩大只读标签/标定审计的序列覆盖。

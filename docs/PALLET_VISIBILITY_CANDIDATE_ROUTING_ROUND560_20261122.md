# Round 560：托盘 LiDAR/时序证据自动路由

## 本轮解决了什么

前几轮已经能分别回答“当前帧有无 LiDAR 支持”和“相邻帧是否支持遮挡”，但仍需要人工把两份报告拼在一起。本轮新增机器可读路由器，将两层证据自动合并为候选处理动作。

## 路由规则

| 输入状态 | 自动路由 | 当前二维候选 | 3D 身份 | 后续动作 |
| --- | --- | --- | --- | --- |
| 单帧稳定可见 | `depth_split_candidate` | 允许保留为候选 | 保留 | 人工抽检后才可升级 |
| 单帧 LiDAR 支持遮挡 | `single_frame_occlusion_defer_2d` | 抑制 | 保留 | 复核遮挡关系 |
| 单帧无回波、时序门控通过 | `temporal_occlusion_defer_2d` | 抑制 | 保留 | 只提供有效传播源帧 |
| 其余情况 | `insufficient_evidence_review` | 抑制 | 保留 | 强制人工复核 |

所有路线统一设置：`write_human_truth=false`、`write_training_data=false`、`production_deploy=false`、`requires_review=true`。

## 当前结果

| 序列/相机 | obj_id | 单帧证据 | 路由 |
| --- | ---: | --- | --- |
| 010 / camera0 | 7 | 稳定可见 | 深度拆分候选 |
| 010 / camera0 | 6 | 稳定可见 | 深度拆分候选 |
| 010 / camera0 | 11 | 稳定可见 | 深度拆分候选 |
| 011 / camera1 | 15 | 稳定可见 | 深度拆分候选 |
| 011 / camera1 | 20 | 单帧无回波、时序支持遮挡 | 抑制当前二维候选 |

obj20 只允许以下 3 个传播源帧进入后续候选生成：

- `1789900388515701364`
- `1789900390014812364`
- `1789900390315797364`

四个 `ignore=true` 帧和可见点不足 3 的帧均未进入传播源。

## 复现

```powershell
.\.venv\Scripts\python.exe tools\route_pallet_visibility_candidates.py `
  --single-frame-report artifacts\round558-pallet-lidar-visibility-final-20261122\results.json `
  --temporal-report artifacts\round559-obj20-temporal-visibility-final-20261122\results.json `
  --output <新的results.json>
```

## 产物与边界

- 实现：`tools/route_pallet_visibility_candidates.py`
- 测试：`tests/test_route_pallet_visibility_candidates.py`
- 结果：`artifacts/round560-pallet-visibility-candidate-routes-20261122/results.json`
- 收据：`artifacts/round560-pallet-visibility-candidate-routes-20261122/receipt.json`
- 实现提交：`3f2003e`
- 26 项相关测试通过。
- 5/5 实例需要复核；4 个允许保留当前二维候选，1 个明确抑制。
- 人工真值、训练集、工作台、GPU、OSS 写入均为 0。
- 当前仍是离线路由合同，不代表已部署到工作台或已获得人工边界准确率。

## 下一步

在隔离环境把路由 JSON 接到候选生成器：`depth_split_candidate` 才允许输出掩码；`temporal_occlusion_defer_2d` 只输出遮挡状态和传播源，不输出中心帧掩码。通过浏览器和文件回读验收后，再评估是否进入正式工作台。

## 后续结果（Round 561）

隔离候选生成器已完成：4 个 `depth_split_candidate` 各输出一个经哈希回读的 640×512 二值 PNG；obj20 没有生成 PNG，只在 bundle 和预览图中保留抑制状态与 3 个传播源帧。详见 `PALLET_VISIBILITY_CANDIDATE_BUNDLE_ROUND561_20261122.md`。

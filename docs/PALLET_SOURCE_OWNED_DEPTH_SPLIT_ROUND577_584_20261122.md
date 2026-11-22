# Round 577–584：密集栈板实例拆分的像素归属修复与保守去碎屑

日期：2026-11-22

## 通俗结论

之前的深度拆分有一个明确错误：近处栈板可以从“所有候选掩码的并集”里取像素，因此会把只属于另一块栈板的区域也拿走。最坏样本因此膨胀到 2,879 像素，并出现 29 个碎片。

本轮做了两层修复：

1. **像素归属修复**：每块栈板只能在自己的 SAM 候选掩码内参与深度裁决；双方都预测到的重叠像素仍按深度分配，但另一实例独有的像素绝不转移。
2. **保守去碎屑**：仅移除小于候选总面积 1%（且小于最小 8 像素阈值）的孤立小岛；不扩大边界、不填孔洞、不合并大块、不修改原始候选。

这两层处理均是离线候选优化，不写入人工真值、训练集、正式工作台或 OSS。

## 量化结果

| 指标 | 修复前 Round 576 | 修复后 Round 584 | 变化 |
| --- | ---: | ---: | ---: |
| 冲突图深度拆分通过 | 2 / 4 | 3 / 4 | +1 |
| 有二维掩码候选 | 7 / 9 | 8 / 9 | +1 |
| 安全门抑制候选 | 2 / 9 | 1 / 9 | -1 |
| 高优先级人工复核 | 4 | 2 | -2 |
| 自动结构告警 | 1 | 0 | -1 |

最坏样本 `frame=1789897549982491905, camera=1, obj=48`：

| 指标 | 修复前 | 像素归属修复后 | 去碎屑后 |
| --- | ---: | ---: | ---: |
| 面积 | 2,879 | 1,650 | 1,592 |
| 连通域 | 29 | 26 | 2 |
| 孔洞 | 5 | 3 | 3 |

解释：先去掉错误借来的 1,229 个像素，再移除 58 个小碎屑；剩余两个较大可见片段被保留，没有为追求“平滑”而强行连接。该样本仍属于高优先级人工复核，不因为结构告警归零就自动升级为真值。

## 仍然没有解决的部分

- `frame=1789897549982491905, camera=0, obj=48` 只有 152 个有效像素，拆分门仍失败，因此继续抑制，不输出掩码。
- `camera=1` 的 obj47 和 obj48 虽已通过自动结构门，但可见区域薄、遮挡重，仍标记为高优先级人工复核。
- 本轮只证明冻结的 5 帧、20 图密集栈板序列上，像素归属错误被修复；不等于跨厂区、跨材质泛化已经成立。
- 所有 8 个输出仍是候选，不是人工真值，也未批准进入训练回流。

## 代码与测试

修改：

- `tools/evaluate_pallet_box3d_overlap_pairs.py`
  - 深度裁决由全体掩码并集改为实例自己的源掩码；
  - 新增仅减不增的 `remove_tiny_islands`；
  - 报告新增移除小岛数量和像素数。
- `tests/test_evaluate_pallet_box3d_overlap_pairs.py`
  - 验证近处实例不能获得另一实例独有像素；
  - 验证去碎屑不会新增像素，且保留主体和显著次组件。

验证结果：

- 相关链路：`17 passed`；
- 正式测试目录（UTF-8 模式）：`1611 passed, 1 skipped, 262 subtests passed`；
- 不加 UTF-8 模式时，Windows 本地 GBK 默认编码导致 15 个已有中文持久化测试读取 UTF-8 JSON 失败；用 `python -X utf8` 复跑后全部通过，与本轮算法无关；
- 全仓库无筛选运行还会误收集 SAM3 自带 `qualitative_test.py`，其可选绘图库 `matplotlib` 未安装；本轮没有为此污染项目依赖。

## 可复核证据

- 深度拆分报告：`artifacts/round581-dense-pallet-overlap-source-owned-clean-20261122/results.json`
  - SHA-256：`f5a8890e4b3c527dd544142b4ec178d9babe1a558281b774b3781dc51f9daaa7`
- 候选路由：`artifacts/round582-dense-pallet-candidate-routes-clean-20261122.json`
  - SHA-256：`7191bb0d3258968705e9ce71b9263a32b8cd5f1591a326a39cc3e3b84c87e9ae`
- 复核包：`artifacts/round583-dense-pallet-candidate-bundle-clean-20261122/bundle.json`
  - SHA-256：`b4b27277412642ae954257319f710d7fc3ae9c992eb7230cd0e8e17e62c801b5`
- 像素审计：`artifacts/round584-dense-pallet-pixel-audit-clean-20261122/results.json`
  - SHA-256：`b95a6c3d051aa4ff91f0661135c3226fcb36bd058b0c5c4f207248c929d48c5b`
- 复核总图：`artifacts/round584-dense-pallet-pixel-audit-clean-20261122/pallet-candidate-pixel-review.jpg`
  - SHA-256：`9117e214c43319d81b1fe6909c85fb172d3ed417bd51af1248ce5d4c1e1bd50f`
- 机器可读回执：`docs/evidence/pallet-source-owned-depth-split-round577-584.json`

## 正式工作台边界

- 正式主页：HTTP 200；
- `/api/run/status`：`idle`；
- `D:\Apps\LabelAgent\work\state.json` SHA-256 仍为 `9ef08118dd2ba0760cc93cf57bbeaae4b3bed062d383df4e6a454c389d4ab115`；
- 正式工作台未部署本轮离线算法，未写候选、未写真值、未启动训练、未启动长期 GPU 任务。

## 下一步

1. 对剩余一个被抑制实例使用相邻帧和 LiDAR 可见性补证据，但不自动扩边；
2. 对两个高优先级候选做人工像素确认；
3. 在独立、从未参与调参的序列上重复同一冻结门限，只有精度与误检门同时通过后才考虑接入正式候选生成链路；
4. 人工门未通过前，继续保持“候选 ≠ 真值 ≠ 训练数据”。

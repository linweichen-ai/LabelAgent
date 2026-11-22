# Round 586–593：真实 LiDAR 回波引导的密集栈板候选恢复

日期：2026-11-22

## 通俗结论

上一轮用投影框猜正负点失败。本轮改成只使用真实 LiDAR 可见回波：

- 目标 obj48 的 3D 框内找到 29 个只属于它的可见回波；
- 两个冲突栈板中找到 65 个只属于它们的可见回波；
- 从目标回波选 3 个分散正点，从冲突回波选 4 个负点；
- SAM 生成候选后，不只检查单个目标，还重新检查整组三个实例的定位、深度拆分和两两交叠。

结果：唯一被抑制的 `frame=1789897549982491905, camera=0, obj=48` 被恢复为 871 像素的可见面候选，1 个连通域、0 孔洞、0 结构告警。冻结候选包从 8/9 有掩码提升到 9/9，抑制项从 1 降到 0。

这仍是高优先级人工复核候选，不是人工真值，也没有回流训练。

## 为什么这次成立、上次不成立

| 方法 | 点从哪里来 | 结果 |
| --- | --- | --- |
| Round 585 | 3D 投影框独占区域 | 可能落在背景或极薄边缘，冲突分离从 3/4 退化到 2/4，拒绝 |
| Round 586–593 | 3D 框内真实可见 LiDAR 回波 | 目标与冲突实例都有真实表面证据，唯一失败实例恢复，整组门通过 |

恢复候选没有把整块投影框涂满，只保留当前视角实际可见的细长栈板表面；这比形态膨胀或强制补全更保守。

## 多重门控

候选只有同时满足以下条件才标记为恢复：

1. 原输入是冻结的评测清单、PCD 清单和离线拆分报告；
2. 至少 80% 目标 LiDAR 正点落入候选；
3. 至少 80% 冲突实例负点保持在候选之外；
4. 通过原有 3D 投影定位门；
5. 与其他实例一起重新深度拆分后，所有实例仍通过候选门；
6. 目标与所有冲突实例的交叠占较小实例比例不超过 15%；
7. 选择通过门的方案中边界更保守的较小候选。

实际选中的 `three_positive_all_negative`：

- 正点覆盖率：1.00；
- 负点排除率：1.00；
- SAM 原始候选：891 像素；
- 深度拆分后目标候选：871 像素；
- 三实例拆分门：3/3 通过；
- 两个冲突对分离门：2/2 通过。

## 稳定性

- 同一冻结输入独立重复两次，三个策略的掩码 SHA-256 均逐一一致；
- 两个通过门的三正点策略，其最终掩码 IoU 为 0.941；
- 选择的最终掩码 SHA-256：`42724a6adc06027fe03ed12bea07622525b9281ccbcd8107ed9ee3c98d76054c`；
- 两次复核总图 SHA-256 均为 `45dc83e68a09007d72a2e49f3ccf2027e9a2145cd5b3da723fa67f9e4c0a496c`。

这里的稳定性只说明同机、同服务、同输入确定性，不代表跨序列泛化。

## 最终候选包

| 指标 | Round 584 | Round 593 |
| --- | ---: | ---: |
| 候选总数 | 9 | 9 |
| 有掩码 | 8 | 9 |
| 被抑制 | 1 | 0 |
| 高优先级人工复核 | 2 | 3 |
| 自动结构告警 | 0 | 0 |

高优先级由 2 增至 3 是预期结果：新恢复实例虽然结构干净，但面积小、遮挡重，必须优先人工确认；这不是质量回退，也不能被自动批准。

## 实现与复现

- 新增 `tools/evaluate_pallet_lidar_conflict_points.py`；
- `tools/build_pallet_visibility_candidate_bundle.py` 支持可选 `--recovery-report`；
- `tools/audit_pallet_candidate_pixels.py` 同时保留旧失败指标与新恢复指标；
- 新增/扩展相应单元测试；
- Windows 产物文件名使用稳定短编号，避免超过旧式路径长度限制，完整策略名保留在 JSON 回执中。

核心命令：

```powershell
.\.venv\Scripts\python.exe tools\evaluate_pallet_lidar_conflict_points.py `
  --split-report artifacts\round581-dense-pallet-overlap-source-owned-clean-20261122\results.json `
  --manifest .repro\round568-dense-pallet-heldout-20261122\manifest.json `
  --pcd-manifest .repro\round568-dense-pallet-heldout-20261122\pcd-manifest.json `
  --labels-root .repro\round568-dense-pallet-heldout-20261122\labels `
  --calibration <只读source\calib> --pcd-root <只读source\lidar> `
  --output <新目录> --sam-url http://127.0.0.1:28766
```

验证结果：

- 新增链路测试：`8 passed`；
- 项目测试目录（UTF-8）：`1615 passed, 1 skipped, 262 subtests passed`。

## 证据

- 选定恢复报告：`artifacts/round589-lidar-points-selected-20261122/results.json`
  - SHA-256：`6ba04a03ae209c84fc6fb49b023c90165a7c8e41ffd6c4775973c392c2bc6cd8`
- 恢复候选包：`artifacts/round592-dense-pallet-bundle-lidar-recovered-20261122/bundle.json`
  - SHA-256：`3225afcc29c70cf8ca846f0b425fb0ceaed63afe2c94cb61c0c8fa7972754273`
- 最终像素审计：`artifacts/round593-dense-pallet-pixel-audit-lidar-recovered-20261122/results.json`
  - SHA-256：`bdc4000bc37c9f26d502e99f1c5927532b5f9743be3b3d226c819317b94ccab6`
- 最终复核总图：`artifacts/round593-dense-pallet-pixel-audit-lidar-recovered-20261122/pallet-candidate-pixel-review.jpg`
  - SHA-256：`6bf1e5ef52e60eea58debdaaded2a50b1727c5923f68b67dafd94918796a59e2`
- 机器可读回执：`docs/evidence/pallet-lidar-conflict-recovery-round586-593.json`

## 安全边界与下一步

- `human_truth=false`、`approved_for_training=false`、`production_deployed=false`；
- 工作台写入、训练数据写入、OSS 写入均为 0；
- 正式工作台保持 `idle`，正式状态文件 SHA-256 不变；
- 下一步必须在新的、未用于门限设计的密集栈板序列上复测召回和误检；
- 独立序列未通过前，不把这条恢复链部署进正式自动标注，也不进入训练回流。

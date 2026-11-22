# Round 550：跨序列 3D 框定位托盘 ROI 级联

## 结论

在冻结的 0920 批次评测清单上，选择 010、011、012 三个序列的同一标注帧，每帧覆盖四路相机，共 12 张图。人工标注的 3D 托盘框只用于定位 ROI，不作为二维像素真值。

最终级联为：

1. 投影 3D 托盘框并裁剪单目标 ROI；
2. 同一次图像编码依次运行 `pallet`、`wooden pallet`、`plastic pallet`、`pallet base`；
3. 用投影框关联候选，避免只选最高置信度实例；
4. 仅对文字分割失败的 ROI，一次编码批量运行中心点、裁剪角负点和 3D 框外环负点三种策略。

最终结果：

- 3D 定位通过：12/12；
- 同时可证明与其他投影托盘框隔离：10/12；
- 多提示后仍失败、触发点兜底：2/12；
- 点兜底成功：2/2；
- 工作台写入：0；正式模型替换：0；训练数据回流：0。

这证明跨 3 个序列的“3D 定位 + 局部 SAM + 按需点兜底”定位闭环可以跑通，但不证明 12/12 像素边界正确。当前 3D 标签不是独立二维人工轮廓，仍需人工像素真值验证边界质量。

## 消融对比

| 方法 | 3D 定位通过 | 实例隔离通过 | 说明 |
| --- | ---: | ---: | --- |
| 单提示 `pallet` | 7/12 | 6/12 | 木/塑料材质与遮挡场景仍漏定位 |
| 四提示、一次编码 | 10/12 | 8/12 | 新增恢复两张蓝色塑料托盘 |
| 四提示 + 单点兜底 | 12/12 | 10/12 | 仅对两张失败图额外调用 |
| 四提示 + 三策略批量点兜底 | 12/12 | 10/12 | 维持定位率，减少 camera2 外溢 |

“实例隔离”比“定位”严格。010-camera0 和 011-camera1 的二维分割落在目标托盘上，但多个 3D 标注框在投影中重叠，因此只能证明定位成功，不能证明实例完全分离。

## 点兜底的边界变化

| 序列 / 相机 | 最终策略 | 中心距离 | 投影框覆盖率 | 掩码位于投影框内比例 | 掩码像素 |
| --- | --- | ---: | ---: | ---: | ---: |
| 011 / camera2 | 3D 框外环负点 | 0.15 px | 26.03% | 100% | 336 |
| 011 / camera3 | 中心点 | 0.42 px | 13.30% | 100% | 1,701 |

camera2 的中心点单策略原结果为 954 像素，只有 45.18% 位于投影框内；批量策略选中框外环负点后，外溢降为 0。覆盖率从 33.38% 降到 26.03%，可能略偏保守，因此只计几何定位改善，不宣称人工边界更准。

## 性能

首次四提示运行的 12 张图总请求时间约 4.27 秒，平均约 0.356 秒/图；这是同图一次编码、四个文字提示。后续重复运行命中远端响应缓存，不能用作冷启动速度结论。两张失败图的三种点策略各只编码一次图像。

## 数据与安全边界

- 冻结图像清单：`heldout-test12-manifest-round49.json`；`evaluation_only=true`。
- 标签只读取 010、011、012 各一个 JSON；未写回数据集或 OSS。
- 本轮尝试读取三个单文件 PCD，但当前 4090 会话的 OSS 列表没有返回对象，因此没有猜测 AccessKey，也没有下载 PCD。
- `human_pixel_truth=false`
- `candidate_only=true`
- `approved_for_training=false`
- `production_deployed=false`
- 正式工作台状态 SHA-256：`9ef08118dd2ba0760cc93cf57bbeaae4b3bed062d383df4e6a454c389d4ab115`

## 复现

```powershell
.\.venv\Scripts\python.exe tools/evaluate_pallet_box3d_roi_sam.py `
  --manifest docs/evidence/workbench-stability-20261117/heldout-test12-manifest-round49.json `
  --labels-root <仅含010、011、012三个冻结标签JSON的目录> `
  --calibration <calib_640x512目录> `
  --output <新的空输出目录> `
  --sam-url http://127.0.0.1:28766 `
  --target-policy largest_empty `
  --prompt-mode multi `
  --point-fallback `
  --point-fallback-mode batch
```

## 产物

- 代码：`tools/evaluate_pallet_box3d_roi_sam.py`
- 测试：`tests/test_evaluate_pallet_box3d_roi_sam.py`
- 最终结果：`artifacts/round550-cross-sequence-box3d-batched-fallback-20261122/results.json`
- 四联对比图：`artifacts/round550-cross-sequence-box3d-batched-fallback-20261122/cross-sequence-box3d-roi-sam.jpg`
- 消融收据：`artifacts/round550-cross-sequence-box3d-batched-fallback-20261122/ablation_receipt.json`
- 实现提交：`d5c0d1d`、`84eacec`

## 下一步

1. 将 12 张结果整理成独立人工二维边界复核包，分别标记正确、漏边、粘连和错误目标。
2. 用人工复核结果计算边界 F1 和实例粘连率，避免只依赖投影框。
3. 重点确认 camera2 框外环策略是否过度收缩；在人工边界门槛通过前，不接入正式自动写回。

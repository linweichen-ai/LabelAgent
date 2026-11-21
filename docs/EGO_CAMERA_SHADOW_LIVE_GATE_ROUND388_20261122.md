# Round388：ego 相机影子评测器隔离实测门禁（2026-11-22）

## 结论

Round387 的默认关闭评测器已在四路本地固定样本上完成一次隔离 point-SAM 调用。
camera0、camera2、camera3 返回可供人工复核的几何摘要；camera1 返回有效轮廓，但 SAM 明确警告
没有覆盖全部正点，因此被标为 `blocked_sam_warning`。四路结果都保持
`eligible_for_candidate_merge=false`、`state_written=false`，没有进入正式候选、人工真值或训练集。

本轮没有部署评测器、没有开启正式配置，也没有上传真实图像或叠加图。

## 为什么调整警告处理

Round387 最初把任何 SAM 警告都归为普通失败，导致 camera1 的面积、外接框和分数也一起丢失。
这虽然保守，但不利于影子诊断。本轮改为“可观察、不可用”：

- 只要响应的分数和几何仍合法，就保留有界摘要；
- 有警告时 `candidate_available=false`，门禁为 `blocked_sam_warning`；
- 只记录警告数量和警告列表 SHA-256，不输出警告原文；
- malformed warnings 仍然失败关闭；
- 任何结果仍不返回原始轮廓，且永远不可自动合并。

## 四路隔离结果

| 相机 | 分数 | 面积占比 | 门禁 | 可用候选 |
| --- | ---: | ---: | --- | --- |
| camera0 | 0.695312 | 11.2744% | human_review_required | 是，但仅人工复核 |
| camera1 | 0.515625 | 9.2353% | blocked_sam_warning | 否 |
| camera2 | 0.632812 | 10.5589% | human_review_required | 是，但仅人工复核 |
| camera3 | 0.812500 | 12.2656% | human_review_required | 是，但仅人工复核 |

这里的“是”只表示评测器观察到一个通过基础响应门禁的影子结果，不代表它是人工真值、可合并候选
或可用于训练的数据。camera1 明确被阻断，其他三路仍需逐像素人工真值验证。

## 验证

- 完整 Python：`1224 passed, 217 subtests passed`；
- 网页 Node：`167 passed`；
- 正式工作台保持 `idle`，SAM、主 VLM、快速 VLM 均 ready；
- 四份正式状态/历史文件 SHA-256 在隔离调用前后完全一致；
- 真实图像与叠加结果只保存在本地未跟踪目录，不进入 GitHub/CNB。

## 门禁结论

Round388 只证明评测器可以在不写正式状态的前提下收集四路有界诊断，并能把 camera1 的 SAM
警告单独阻断。它不证明像素质量、跨工厂泛化或生产可用。下一步仍是建立小规模逐像素人工真值，
分别计算 ego IoU、person/floor 误纳率和警告样本拒绝率；这些指标通过前继续保持默认关闭。

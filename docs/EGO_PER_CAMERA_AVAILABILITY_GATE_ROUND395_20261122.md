# Round395：ego 分相机质量与候选可用性门禁（2026-11-22）

## 结论

Round394 固定了整体最低标准，但整体平均 IoU 仍可能掩盖某一路相机长期偏低；同时，已经被提示点
约束判为不可用的候选仍可计算出较高像素指标，容易被误读为“可以上线”。

本轮将清单升级为 v3，新增分相机平均质量和候选可用性硬门禁。被阻断的候选仍保留像素指标供
诊断，但不能帮助正式门禁通过。

## 新增硬门禁

- 每一路相机平均 ego IoU 都必须不低于 0.85；
- 每一路相机必须至少有 5 个 `candidate_available=true` 的候选；
- 任何样本只要仍为不可用候选，报告就增加 `blocked_candidate_present` 并拒绝通过；
- 报告明确输出整体可用/阻断数量，以及每路相机的可用/阻断数量；
- 清单中 `minimum_camera_mean_iou` 可以调高，不能低于 0.85。

## 为什么整体平均值不够

定向负控构造了 camera0 五帧 IoU 均为 0.75、其他三路均为 1.0 的情况：整体平均仍达到
0.9375，最低单帧也正好达到 0.75，但 camera0 明显没有达到正式质量标准。v3 会以
`camera0:mean_iou_below_gate` 拒绝，不允许其他相机的高分抵消 camera0。

另一项负控中，全部像素 IoU 都是 1.0，但一个候选被标为 `blocked_prompt_constraint`。v3 同时以
`blocked_candidate_present` 和 `camera0:insufficient_available_samples` 拒绝。

## 对 Round392 的真实含义

Round392 本地复核包现有候选可用性为：

| 相机 | 总样本 | 可用 | 阻断 |
| --- | ---: | ---: | ---: |
| camera0 | 5 | 4 | 1 |
| camera1 | 5 | 0 | 5 |
| camera2 | 5 | 5 | 0 |
| camera3 | 5 | 5 | 0 |

因此，即使未来补齐人工像素真值，当前候选包也会因 6 个阻断候选、camera0/camera1 可用样本不足
而失败。这是预期行为：需要先得到更多通过提示点约束的独立候选，而不是用高 IoU 绕过已有门禁。

## 版本

- manifest：`labelagent.ego_pixel_truth_manifest.v3`；
- report：`labelagent.ego_pixel_truth_report.v3`；
- human receipt：继续使用 Round394 的
  `labelagent.ego_human_pixel_truth_receipt.v3`；
- 旧 manifest v2/v1 失败关闭。

## 验证

- 定向 Python：`12 passed, 2 subtests passed`；
- 完整 Python（`PYTHONUTF8=1`）：`1241 passed, 219 subtests passed`；
- 网页 Node：`167 passed`；
- 正式工作台 HTTP 200，SAM、主 VLM、快速 VLM 均 ready；
- 四份正式状态/历史 SHA-256 未变化；
- 没有人工真值、没有正式质量报告、没有部署或训练批准写入。

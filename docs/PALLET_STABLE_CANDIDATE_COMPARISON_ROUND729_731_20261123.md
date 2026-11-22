# 栈板稳定候选并排确认包（Round 729–731）

日期：2026-11-23

## 结论

已把 obj12、obj13、obj78 的“Round 721 当前候选、Round 728 稳定中心候选、50% 共识、75% 共识”放到相同 ROI、颜色和透明度下直接比较。

- obj12：稳定中心从 62 像素补到 136 像素，并与 50%/75% 共识完全一致，可提升为优先人工确认候选；
- obj78：47 像素补到 58 像素，与旧候选 IoU 0.780、与保守共识 IoU 0.914，可提升为优先人工确认候选；
- obj13：36 像素扩到 109 像素，虽与共识高度一致，但新增横向区域可能跨相邻堆叠层，继续保持待定。

这些判断仍是助手建议，不是人工签核、训练批准或正式候选替换。

## 对比结果

| 目标 | 当前策略 | 稳定中心策略 | 像素变化 | 当前/稳定中心 IoU | 稳定中心/50%共识 | 稳定中心/75%共识 | 建议 |
|---|---|---|---:|---:|---:|---:|---|
| 006 / obj12 | all negative | one negative 0 | 62 → 136 | 0.445 | 1.000 | 1.000 | 优先人工确认稳定中心 |
| 006 / obj13 | all negative | one negative 0 | 36 → 109 | 0.330 | 0.991 | 0.945 | 保持待定，重点检查跨层 |
| 008 / obj78 | two negative | one negative 0 | 47 → 58 | 0.780 | 0.892 | 0.914 | 优先人工确认稳定中心 |

三项新旧 mask 都不相同。这里的“稳定”只表示同源提示策略之间一致，不表示像素语义已经被人工证明。

## 更新后的助手分流

综合 Round 722、局部放大和本轮稳定性对比：

- 5 个可进入人工确认：obj16、obj23、obj69、obj12-medoid、obj78-medoid；
- 7 个拒绝并重画：obj10、obj15、obj67、obj75、obj76、obj77、obj79；
- 3 个继续待定：obj11、obj13、obj14。

这个分流只改变后续人工复核顺序，不会自动写回工作台。

## 可重复性

Round 729/730 双跑一致；增加重复 Round 674 图像身份拒绝门禁后，Round 731 使用当前工具复跑，三轮各 3 张对比图和 1 份 manifest，逐文件完全一致。

| 检查 | 结果 |
|---|---|
| manifest SHA-256 | `b5424198b265f6851a6817ae7090f9be51445694646f96b9f2faa514a90d867a` |
| 规范文件树 SHA-256 | `a5ff83a15dc7756841857ac8d67d0216beeff2d8a30342b160eca4dc5509dc9c` |
| 不同文件数 | 0 |
| 定向测试 | 6/6 通过 |
| 项目测试集 | 1720 通过、1 按设计跳过、262 个子测试通过 |

本机输出：

- `artifacts/round729-pallet-stable-candidate-comparison-20261123/`
- `artifacts/round730-pallet-stable-candidate-comparison-repeat-20261123/`
- `artifacts/round731-pallet-stable-candidate-comparison-current-tool-20261123/`

## 安全边界

本轮未调用 GPU/SAM，未修改人工真值、训练集、正式工作台或 OSS。所有 decision 字段仍为空，稳定中心候选必须经过人工确认才能进入下一步。

结构化证据见 `docs/evidence/pallet-stable-candidate-comparison-round729-731-20261123.json`。

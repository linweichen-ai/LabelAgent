# 栈板多提示 mask 稳定性审计（Round 726–728）

日期：2026-11-23

## 结论

对 006/obj12、006/obj13、008/obj78 的 15 个有效提示策略 mask 做了同图一致性比较。三项目标的平均两两 IoU 分别为 0.775、0.633、0.710，说明候选位置并非随机漂移；但最低 IoU 仍只有 0.442、0.316、0.485，因此这只能作为“提示稳定性”证据，不能证明实例身份或像素边界正确。

三项目标的 medoid 都是 `three_positive_one_negative_0`。它是与其他策略平均最接近的助手候选，后续可优先展示给人工，但不会自动替换正式候选、写入真值或进入训练。

## 指标怎么理解

- 两两 IoU：两种提示策略所得 mask 的重叠程度，越高说明提示扰动下越稳定；
- medoid：与其余策略平均 IoU 最高的现有 mask，不是生成的新真值；
- 50% 共识：至少一半纳入策略都选择的像素；
- 75% 共识：至少四分之三策略都选择的更保守像素；
- 本轮所有 mask 来自同一图、同一模型和同一诊断链路，不是独立测试集验证。

## 结果

| 目标 | 有效策略数 | 平均/最低两两 IoU | medoid 平均 IoU | 50%/75% 共识像素 | 结论 |
|---|---:|---:|---:|---:|---|
| 006 / obj12 | 5 | 0.775 / 0.442 | 0.858 | 136 / 136 | 稳定性最高，但实例身份仍待人工确认 |
| 006 / obj13 | 4 | 0.633 / 0.316 | 0.741 | 110 / 103 | 策略差异较大，继续保持待定 |
| 008 / obj78 | 6 | 0.710 / 0.485 | 0.774 | 65 / 53 | 有稳定核心，但覆盖很小，继续保持待定 |

三项 medoid 均为 `three_positive_one_negative_0`。Round 721 当前展示的 obj12、obj13 使用 `three_positive_all_negative`，obj78 使用 `three_positive_two_negative`；因此本轮提出的是新的人工复核优先候选，不是静默替换。

## 可重复性

Round 726/727 双跑一致；增加重复 Round 674 图像身份拒绝门禁后，Round 728 使用当前工具复跑，三轮 13 个文件逐字节一致。

| 检查 | 结果 |
|---|---|
| 纳入策略 mask | 15 |
| manifest SHA-256 | `dde20194fb20fbe18a98d7d8ccbf00d5304f9aec25abba6506d79fd08bbf9c3a` |
| 规范文件树 SHA-256 | `05d68aee958c56bb097ec0ff9fe044ef7511fcf6b7d925b74e3558c8c825ace2` |
| 不同文件数 | 0 |
| 定向测试 | 7/7 通过 |
| 项目测试集 | 1714 通过、1 按设计跳过、262 个子测试通过 |

本机输出：

- `artifacts/round726-pallet-prompt-mask-stability-20261123/`
- `artifacts/round727-pallet-prompt-mask-stability-repeat-20261123/`
- `artifacts/round728-pallet-prompt-mask-stability-current-tool-20261123/`

## 边界与下一步

本轮未调用 GPU/SAM，未写正式工作台、人工真值、训练集或 OSS。下一步把 medoid、原候选、50%/75% 共识并排交给人工确认；在人工确认前，三个目标仍保持候选或 `ignore`，不得作为训练监督。

结构化证据见 `docs/evidence/pallet-prompt-mask-stability-round726-728-20261123.json`。

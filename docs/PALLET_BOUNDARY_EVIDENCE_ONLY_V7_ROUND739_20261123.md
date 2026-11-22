# Round739：栈板边界 v7 图像证据隔离候选

## 结论

`pallet-boundary-cleanup-v7-evidence-only` 可以作为 **shadow candidate（影子候选）**
保留和继续研究，但没有部署，也不能替代当前默认的 v6 路径。

本轮解决的是 Round738 暴露的“边界越修越差”风险：v7 在没有明确局部图像证据时完全不改
候选；在 3 个真实代表帧的 34 个栈板候选上，两次独立回放均为 34/34 mask 不变。与同一原图
绑定的 11 个人工复核栈板也全部保持原指标。因此本轮通过的是**非退化回放**，不是效果提升门；
真实数据上尚无一次 v7 修复被触发，不能宣称栈板分割质量已经提升。

## 为什么需要 v7

Round738 证明旧 v6 清理在唯一可严格绑定的人工作用域上发生了真实回退：10 个变化候选全部
IoU 下降，5 个触发单实例回退门。复核发现这批候选没有需要修补的内部孔洞或小岛，退化主要
来自无图像证据的中值平滑和整条轮廓简化。

v7 因此不再把“轮廓看起来平滑”当成充分依据，而采用以下失败关闭策略：

1. 默认关闭中值平滑、整轮廓简化和几何小岛删除；
2. 只允许有局部图像证据的孔洞或边界缺口进入修复；
3. 修改只能增加被授权区域内的像素，禁止删除任何原前景；
4. 最终重建 mask 必须与内部批准 mask 完全一致，序列化不得带来额外像素；
5. 所有 8 连通前景组件（包括小于 9 像素的微小组件）前后数量必须一致，禁止实例粘连；
6. 单次总预算不超过候选面积的 1%，并且绝对不超过 256 像素；
7. 已识别的结构孔保留率必须为 100%。

默认 `geometry_policy=full` 仍保持原 v6 mask 行为；本轮只新增可显式选择的
`appearance_evidence_only` 影子策略及附加审计字段。

## 真实只读回放

| 项目 | 结果 |
|---|---:|
| 代表帧 | 3（frame 0、76、151） |
| 栈板候选 | 34 |
| v7 实际变化 | 0/34 |
| current/refined mask SHA 相同 | 34/34 |
| 输出文件 | 137 |
| 两次 manifest SHA | `ee1f2ff9306d187aedf2bdc75126724b23291aa1c9f8ff45c48c6efa4dc44b59` |
| 两次规范文件树 SHA | `f3c11a999e108d48b9aec90fe4a5f5bf0e70113d4edb5a2387d49174d59be958` |
| 正式载入代次 | `1838476875584112` |

生成器只调用正式工作台 GET 接口；没有写入人工真值、候选、训练数据或模型权重。

## 人工复核门

评测器升级为 schema v2，明确区分“候选真的发生变化并接受评测”和“影子策略没有触发”。
若所有 mask 都不变，不再错误报告 `development_gate_passed`，而报告：

- `status=non_regression_replay_passed`；
- `effect_exercised=false`；
- `development_gate_passed=false`；
- `production_deployment_authorized=false`；
- `training_authorized=false`。

本轮 11/11 人工复核栈板固定匹配成功，IoU、Dice、BF1@1px、BF1@2px 的前后差值均为 0，
`changed_rows=0`。两次独立评测的四项输出逐文件一致：

| 输出 | SHA-256 |
|---|---|
| `report.json` | `76e5f8c4c8b335c517f8a91db73af91e6b70ec2c1b80bad2f057e5d0d7b3a158` |
| `per_instance.csv` | `1aa481566f546cc76113bb458376ea00fd47709a8a89a58e22c684cb1618b63b` |
| `contact_sheet.png` | `75fe227d18955effb20d79ba451e1e5ec6ab6c8d3aa1dc577da06450a5848402` |
| `public_receipt.json` | `e0d5460914099772db11a848472f484548d484daaa4225accafbc09c7096c999` |

## 测试与现场状态

- v7、候选包和评测器定向回归：285 项通过、83 个子测试通过；
- 项目正式 `tests/` 回归：1745 项通过、1 项跳过、272 个子测试通过；
- 直接对仓库根目录裸跑会额外收集 SAM3 自带的 `qualitative_test.py`，当前项目环境未安装其
  可选 `matplotlib` 依赖而在收集期停止；正式测试范围不包含该上游定性脚本；
- WorkBuddy 独立只读复审结论：可提交为 shadow candidate，无 P0/P1 阻断，但不得部署；
- 正式 `state.json`、`files.json`、`file_sessions.json` SHA 仍分别为
  `9ef08118...115`、`7edb40a9...5c1`、`94e8bee3...e02e`；
- 安全通道、4090/5090 SAM、Qwen/5090 加速和标注页面全部正常，正式首页、bootstrap、
  SAM health 和 VLM models 均 HTTP 200。

## 决策与下一步

1. 本轮代码只进入版本库作为影子实验能力，不部署正式工作台，不改变默认 v6；
2. 下一步从新数据或未复核帧中寻找会触发局部图像证据的真实正例，先离线评测再谈采用；
3. 一旦出现真实 changed row，人工门增加逐实例 BF1@1px 非退化和更严格的 IoU 门，避免
   BF1@2px 饱和掩盖细边界回退；
4. 将当前“候选面积至少占全图 30%”改造成分辨率/ROI 稳健的尺度条件前，不能声称跨数据泛化；
5. 继续保持机器候选、人工真值、训练批准和正式部署四层隔离。

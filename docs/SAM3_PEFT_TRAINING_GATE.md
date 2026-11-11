# SAM3 参数高效监督微调门禁

当前训练路线锁定为 supervised parameter-efficient SAM3 fine-tuning，不进行大范围或全模型 post-training。大图像骨干和语言骨干冻结，只训练 segmentation heads/adapters；正式训练获批后使用双 5090 DDP（world size 2）。

当前状态仍是 `formal_training_authorized=false`、`formal_training_started=false`。原因不是服务或算力不足，而是真实人工确认的像素 mask 尚未完成导出、不可变版本化和 split 隔离核验。

进入本决策前的三个前置里程碑已经完成：人工门禁原子性/候选哈希绑定等安全修复已通过回归；slope 已统一为项目定义的 passable，并在下游执行 `traversable_area = floor UNION slope`；固定 640 holdout 已完成真实 `VLM → Agent no_op → 独立 VLM` 复核并停在 `human_review_required`。这些里程碑只证明候选闭环和门禁可用，不替代训练所需的人工像素真值。

## 数据边界

- 72/24 机器预标注只能用于 prompt、ROI 和 candidate，不得作为 human truth。
- BEV 3D boxes 只能用于 prompt、ROI 和 candidate，不得转换后冒充二维人工像素真值。
- `test24` 与固定 640×512 holdout 必须始终排除在训练之外。
- 只有经当前人工门禁确认、绑定候选哈希并导出的真实像素 mask，才能进入监督训练清单。
- synthetic/test-only 人工事件只用于门禁测试，永远不能进入训练清单。

## 开训前全部条件

1. 真实人工像素 mask 已导出且数量大于零；
2. 真值已生成不可变 `version_id` 和 manifest SHA-256；
3. train/val/test 按序列或实体隔离并完成无泄漏核验；
4. `test24` 与固定 640 holdout 的排除关系已经对最终训练 manifest 回读确认；
5. 配置明确冻结 image/language backbone，仅开放 segmentation heads/adapters；
6. 双 5090 DDP 配置为 `world_size=2`；
7. 机器可读门禁返回零错误且 `formal_training_authorized=true`。

机器可读当前状态见 [`test-results/sam3-peft-training-readiness-20261111.json`](test-results/sam3-peft-training-readiness-20261111.json)，Schema 见 [`../schemas/sam3-peft-training-gate.schema.json`](../schemas/sam3-peft-training-gate.schema.json)。当前文件只固化训练决策和启动门禁，不表示已经微调、启动训练或取得训练效果。

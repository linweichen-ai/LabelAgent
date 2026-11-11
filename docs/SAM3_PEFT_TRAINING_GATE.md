# SAM3 参数高效监督微调门禁

当前训练路线锁定为 supervised parameter-efficient SAM3 fine-tuning（监督式参数高效微调，PEFT）。PEFT 属于广义 post-training 的一个具体子类；本项目选择这个明确子类，不采用含义模糊的“全模型后训练”。首轮冻结 image backbone 和 language backbone，只允许 mask decoder、segmentation head，以及经过代码与参数清单验证的 adapter 进入可训练集合。`segmentation_head.mask_predictor.mask_embed.*` 只定义为 stage-0 最小可逆基线，不是最终上限。尚未找到经过验证的通用 adapter/LoRA 接入，因此 adapter 当前记为 `not_verified_in_current_source`，不能写成已支持。正式训练获批后使用双 5090 DDP（world size 2）。

当前状态仍是 `formal_training_authorized=false`、`formal_training_started=false`。原因不是服务或算力不足，而是真实人工确认的像素 mask 尚未完成导出、不可变版本化和 split 隔离核验。

进入本决策前的三个前置里程碑已经完成：人工门禁原子性/候选哈希绑定等安全修复已通过回归；slope 已统一为项目定义的 passable，并在下游执行 `traversable_area = floor UNION slope`；固定 640 holdout 已完成真实 `VLM → Agent no_op → 独立 VLM` 复核并停在 `human_review_required`。这些里程碑只证明候选闭环和门禁可用，不替代训练所需的人工像素真值。

## 数据边界

- 72/24 机器预标注只能用于 prompt、ROI 和 candidate，不得作为 human truth。
- BEV 3D boxes 只能用于 prompt、ROI 和 candidate，不得转换后冒充二维人工像素真值。
- `test24` 与固定 640×512 holdout 必须始终排除在训练之外。
- 只有经当前人工门禁确认、绑定候选哈希并导出的真实像素 mask，才能进入监督训练清单。
- synthetic/test-only 人工事件只用于门禁测试，永远不能进入训练清单。
- 当前 96 包主要覆盖 `pallet/person`。即使该包以后完成真实人工复核并用于 PEFT，也不能据此宣称 slope 提升。
- slope 需另建真实人工像素真值小集，并保持序列/实体级隔离；训练和专项评测继续保留 `floor`、`slope` 两类，下游输出才执行 `traversable_area = floor UNION slope`。

## 无人工真值时允许做什么

在真实 `_annotations.coco.json`、不可变 `truth_version` 和 split 隔离未齐全时，只允许完成配置校验、manifest 校验、可训练模块清单和一次性的双 5090 DDP 初始化/通信启动验证。该 smoke 不得执行 optimizer step，不得产生或宣称训练 loss、性能指标、checkpoint 或“已微调”。若服务器实时验证尚未完成，应明确记录为 pending，不能用本地模拟替代双 5090 NCCL 证据。

## 开训前全部条件

1. 真实人工像素 mask 已导出且数量大于零；
2. 真值已生成不可变 `version_id` 和 manifest SHA-256；
3. train/val/test 按序列或实体隔离并完成无泄漏核验；
4. `test24` 与固定 640 holdout 的排除关系已经对最终训练 manifest 回读确认；
5. 配置明确冻结 image/language backbone，当前仅开放源码实际存在并经过前缀核验的 mask-embedding head；若以后加入 adapter，必须先补实现、参数清单和冻结回归；
6. 双 5090 DDP 配置为 `world_size=2`；
7. 机器可读门禁返回零错误且 `formal_training_authorized=true`。

## 后续扩大训练范围的条件

首轮 PEFT 必须先在真实人工真值上证明有效，并在独立 test/固定 holdout 上通过专项评测。只有当人工样本规模显著扩大、PEFT 已达到可复现瓶颈，并完成过拟合和成本评估后，才评估解冻 image backbone 高层；全模型微调是更晚的备选项，不是默认路线，也不能因算力充足而自动启用。

## Staged unfreeze 与回滚

当前源码静态清点确认了真实调用链：`Sam3Image._run_segmentation_heads → segmentation_head`、`Sam3Image._encode_prompt → geometry_encoder`、`Sam3Image._run_decoder → transformer.decoder`。因此人工真值齐备后按以下顺序实验：

1. stage-0：只开放 `segmentation_head.mask_predictor.mask_embed.*`，失败时回滚原始 SAM3；
2. stage-1：开放完整 `segmentation_head.*`（含 pixel decoder/mask predictor），失败时回滚最佳 stage-0；
3. stage-2：在 stage-1 基础上开放 `geometry_encoder.*`，仅在 prompt/geometry 诊断支持时进入，失败时回滚最佳 stage-1；
4. stage-3：再开放运行时 `named_parameters()` 实际存在的 transformer decoder 上半层和 decoder norm，失败时回滚最佳 stage-2。

每级都必须在同一固定 holdout 上相对原始 SAM3 比较 pixel recall、mask IoU 和 Boundary F1。数值阈值须在人工作真值形成后先版本化，当前不凭空填写。只有当前级未过既定门槛且训练本身通过 sanity check，才允许升级；过门槛即停在更小范围。当前本机没有 torch/checkpoint，服务器认证也不可用，因此目前只有源码模块证据，真实 `named_parameters()`、梯度非零清单和一次前后向依赖仍为 pending。执行工具为 `tools/sam3_trainable_inventory.py`，计划见 `configs/sam3_sfl_cpd15y_staged_unfreeze.json`。

机器可读当前状态见 [`test-results/sam3-peft-training-readiness-20261111.json`](test-results/sam3-peft-training-readiness-20261111.json)，不可启动的配置计划见 [`../configs/sam3_sfl_cpd15y_peft_plan.json`](../configs/sam3_sfl_cpd15y_peft_plan.json)，Schema 见 [`../schemas/sam3-peft-training-gate.schema.json`](../schemas/sam3-peft-training-gate.schema.json)。当前文件只固化训练决策和启动门禁，不表示已经微调、启动训练或取得训练效果。

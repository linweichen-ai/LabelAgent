# 第一条已知类 slope hard-case 事件回执（2026-11-11）

## 已消费证据

确定性 review overlay 已核验为 1,685,438 bytes，SHA-256 `b103e9ba8b42fbb2be01d8943b740caac95e27ca06775712a4eac9207134a9d7`。原图、SAM polygon、VLM 复核、review manifest 和工作台导入回执的当前文件哈希均与其内部引用一致。

来源目录：`C:\Users\seer\Documents\ChatGPT\Seer2clw\outputs\sfl_cpd15y_stage2_20261110\slope_review`。

已落地两个相邻的 1.1 版真实事件：

1. `evt-slope-a000-f0-candidate-b6652e98`：`candidate.generated`，`qc_passed → candidates_generated`；SAM score 0.98046875，仅保存已知类 `slope_visible` 机器候选。
2. `evt-slope-a000-f0-vlm-aee398ed`：`vlm.reviewed`，`candidates_generated → vlm_reviewed`；Qwen3-VL-8B 使用原图和 overlay，结构化判断 `classification=slope`、`visible_slope=true`、`candidate_supported=true`。

两事件使用相同 `run_id`、`correlation_id` 和真实输入 SHA-256；第二事件的 `parent_event_id` 指向第一事件。没有创建 `unknown_candidate`，没有修改 ontology。

## 强制边界

两事件都明确保存：

- `human_pixel_truth_available=false`；
- `approved_for_training=false`；
- `business_mapping=traversable_area`；
- `traversability_status=passable`；
- `policy_default=passable, policy_basis=user_defined_project_semantics`；
- `review.required=true, review.status=pending`。

SAM 点 `(1100,620)` 是 Codex/脚本选择的操作种子，事件固定记录 `prompt_provenance=assistant_selected_prompt`、`prompt_role=operator_seed`，不得解释为人工提示或人工真值。

事件协议新增向后兼容的可选 `approved_for_training` 字段，并增加运行时门禁：没有人工像素真值时不得为 `true`；`candidate.generated`、`vlm.reviewed`、`agent.action` 或 `unknown.proposed` 等机器事件不得批准训练。历史 1.1 事件没有该字段时不会被自动解释为批准。

VLM 本次没有输出可校准的数值置信度，所以事件中的 `decision.confidence=null`，没有用 SAM score 冒充 VLM 置信度。VLM 服务耗时 5.099 秒、总 token 1,257，只证明该候选获得已知类视觉支持。

## 当前状态

该 1920 frame0 历史链停在 `vlm_reviewed`。工作台回读仍为 1 个 candidate、0 truth、0 reference、0 agent run；底层账本仍为 `received → human_review_required`，`minimum_closed_loop_complete=false`。未产生人工批准、`truth.committed` 或 `dataset.versioned`。它不是固定 640×512 holdout 的替代品。

业务可通行性按项目策略直接为 passable，下游投影执行 `traversable_area = floor UNION slope`。LiDAR/车辆信息只能作为候选改善或辅助观测，不能阻塞该映射。人工像素真值缺失仅阻止 IoU/专项质量结论、训练批准和真值提交。

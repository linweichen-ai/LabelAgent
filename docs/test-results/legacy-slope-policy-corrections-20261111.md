# Legacy slope 策略纠正与不可变证据关系（2026-11-11）

## 边界

旧回执和旧 VLM sidecar 保留其形成时的字节、SHA-256 和当时语义。它们只允许经 `source/agent/legacy_slope_reader.py` 做哈希校验后的归档回读，禁止作为当前事件重新写入。当前 schema/runtime 不接受旧的 `geometry_evidence_sufficient`、`geometry_evidence_artifact_ids` 或 `traversability_status=unknown/review_required`。

新策略不回写旧文件，而是由相邻、不可变的 correction artifact 覆盖其业务解释：`label_type=slope`、`traversability_status=passable`、`policy_basis=user_defined_project_semantics`，下游执行 `traversable_area = union(floor, slope)`。人工像素真值仍不存在，因此 `approved_for_training=false`、`truth_committed=false`、`training_ready=false`。

## 三份旧回执与 correction

| 旧文件 | 旧 SHA-256 | old_event_id | correction 文件 | correction SHA-256 |
|---|---|---|---|---|
| `closed-loop-slope-known-label-candidate-generated-20261111.json` | `fe801bed6cd617cd3e545c159af14d975d71f8d1f3d3eee1d1076112c5909b00` | `evt-slope-a000-f0-candidate-b6652e98` | `closed-loop-slope-known-label-candidate-generated-20261111.policy-correction.json` | `9074a8444f9739a1ea633868bf4331fd2a72cad33dcfda106accb426dd75591b` |
| `closed-loop-slope-known-label-vlm-reviewed-20261111.json` | `7cc56fbe1c9493c5dadde5d0d8c0518cf585250e945ccbcad2ae6fc278b2c93e` | `evt-slope-a000-f0-vlm-aee398ed` | `closed-loop-slope-known-label-vlm-reviewed-20261111.policy-correction.json` | `850fd3227d410033dc27064186bbd018d0e7f63ad57ab1c33039a344d0aca811` |
| `slope-frame0-overlay-review-20261111.json` | `79a1be7b9e0a7141a1233e6cc59572e9b42abe63b19312345c9e37db478c7a19` | `null`；归档标识写入 correction | `slope-frame0-overlay-review-20261111.policy-correction.json` | `2daaa69e64577b0135353073d818d4c0176f649c6beb517b2315008e79047edc` |

每份 correction 的 `archival_binding.old_artifact_sha256` 和 `old_event_id` 精确绑定旧文件，`superseded_by` 指向该 correction 自身的 `correction_id`；这表示业务策略被纠正，不表示历史文件被替换或删除。

## 旧 VLM sidecar

旧 sidecar `outputs/sfl_cpd15y_stage2_20261110/slope_review/vlm_slope_candidate_review.json` 保持 SHA-256 `aee398edace4c7b46ed04a1379a05cd6f0502e3809b7ae1a88972dba005a7dea`。既有不可变纠正事件与回执为：

- `vlm_slope_candidate_review.policy-correction.json`：SHA-256 `f633dff41ee52d7804edbb09c6c408e21067a63007d69a0cef8c1a5310344262`；
- `vlm_slope_candidate_review.policy-correction.receipt.json`：SHA-256 `1382b33aba719a91bb6ebd3d210f822eb90d7e7b9547128b739a3e382d845bc4`。

为满足当前统一的 archival binding 字段，另新增（不改写前述既有 correction）`docs/test-results/vlm-slope-sidecar-legacy-policy-correction-20261111.json`，其 SHA-256 为 `27825b3edc5f893fe5f35a03d1a2e8b0246c9b54cc002bf26d4d50faf73aa08f`。它显式记录 `old_event_id=null`、旧 sidecar SHA、`superseded_by=corr-legacy-slope-sidecar-aee398ed` 和当前 passable 策略。

## 校验口径

- 三份旧 JSON 与策略迁移前的 Git 基线 `cc20fb0` 内容一致；
- 四份旧 artifact 均按预期 SHA-256 回读；
- 三份新 correction 对原文件逐字节进行完整 JSON Schema 校验；
- correction 中不存在旧 geometry 字段，且不允许 `unknown/review_required` 进入新策略；
- legacy reader 只能返回归档快照，调用 `as_current_event()` 会显式失败。

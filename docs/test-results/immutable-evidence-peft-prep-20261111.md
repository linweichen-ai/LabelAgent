# 不可变证据修复与 SAM3 PEFT 准备回执（2026-11-11）

`963a46c` 没有被重写。该提交当时在 `docs/test-results/` 中就地修改的 9 个历史 JSON/MD，现已全部恢复为 `cc20fb0` blob 内容，并逐文件重算 SHA-256。旧 VLM sidecar 仍为 `aee398edace4c7b46ed04a1379a05cd6f0502e3809b7ae1a88972dba005a7dea`，未改写。

每个历史文件均有独立 `*.superseding-correction.json`，统一绑定 `prior_artifact_path`、`prior_sha256`、`prior_event_id`、`superseded_by` 和当前 slope 策略。10 份 correction（含 sidecar）均通过 `historical-evidence-superseding-correction.schema.json` 整文件校验。fixed holdout 仅在新 correction 中追加 `candidate_final → human_review_required`，旧回执继续保留当时的 `candidate_final` 事实。

当前 runtime/schema 继续拒绝 `geometry_evidence_sufficient`、`geometry_evidence_artifact_ids` 和默认 `traversability_status=unknown`。归档读取仅通过 `legacy_evidence_reader.py` 做 SHA 绑定的 metadata 回读，无法转换为当前事件。

96 pack 本地没有 `_annotations.coco.json`，历史 CVAT 回执仍是 train72/test24 机器预标注、人工复核 pending。当前对 5090 与 4090 的只读 SSH 均因认证失败而无法回读远端，因此没有声称远端已经产生人工真值，也没有启动训练。

已完成受阻模式下的配置计划：冻结 image/language backbone，只计划训练 `segmentation_head.mask_predictor.mask_embed.*`；NCCL、world size 2、每卡 batch 1；train72 为序列 000–008，test24 为 010–012，固定 640 holdout 和 test24 均禁止进入训练。launch-gate dry-run 验证了双卡启动合同，但因人工真值和服务器实机配置验证缺失，未生成或冒充可执行 Hydra 配置，返回 `executable_hydra_config_ready=false`、`process_started=false`、`formal_training_authorized=false`。相关回归 50/50 通过。

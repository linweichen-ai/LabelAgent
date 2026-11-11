# Slope policy、固定 holdout 与 SAM3 PEFT 顺序证据

## 1. Superseding policy correction

旧 `vlm_slope_candidate_review.json` 未改写，回读 SHA-256 始终为 `aee398edace4c7b46ed04a1379a05cd6f0502e3809b7ae1a88972dba005a7dea`。新增：

- `vlm_slope_candidate_review.policy-correction.json`：`f633dff41ee52d7804edbb09c6c408e21067a63007d69a0cef8c1a5310344262`；
- `vlm_slope_candidate_review.policy-correction.receipt.json`：`1382b33aba719a91bb6ebd3d210f822eb90d7e7b9547128b739a3e382d845bc4`。

event 明确记录 `business_mapping=traversable_area`、`union(floor,slope)`、`traversability_status=passable`、`policy_basis=user_defined_project_semantics`、`human_pixel_truth_available=false`、`approved_for_training=false`。event 与 receipt 分别使用 Draft 2020-12 Schema 对完整文件原样校验，错误数均为 0；receipt 同时绑定旧 sidecar、event 和 event Schema 的 SHA。

## 2. 固定 640 holdout

既有真实模型链按不可变哈希回读，未重复生成或覆盖：Batch A/000、`frame_id=1789897506782450197`、`camera_0`、图像 SHA-256 `34e5094d24db487b3fcd5c458ac810cde5d58293f89fe0b92a2933c72a146812`。

SAM 候选后，首轮 VLM 同时读取原图与 overlay；Agent 明确记录 `no_op`；独立二次 VLM 仍判定 slope 候选受支持并建议 `no_op`。模型候选冻结为 `candidate_final`，工作流因没有人工像素真值停在 `human_review_required`。扫描未发现 `human_accepted`、`truth_committed`、`version_committed` 或 `training_ready` 产物。逐文件哈希见 `fixed-holdout-chain-readback-20261111.json`（SHA-256 `32f1d0d8a349abf2c559cf1a7079df1bf85a0dcc0f7bd0354382650c27e29274`）。

## 3. SAM3 PEFT 训练准备

当前源码实际可确认的首选可训练前缀是 `segmentation_head.mask_predictor.mask_embed.*`；image/language backbone 均冻结。当前源码未验证通用 adapter/LoRA 接入，因此 adapter 不作为已支持模块。SAM3 Trainer 的 DDP 包装、单机按 `gpus_per_node` 启动和 `WORLD_SIZE=2` 路径已做源码哈希核验；官方示例配置虽写两卡，但默认 `enable_segmentation=False`、mask loss 仍是注释，不能直接当作可运行的 mask 微调配置。

本地工作区没有找到 `_preannotations.coco.json` 或人工 `_annotations.coco.json`；此前 96 帧快照是 72 train/24 test 的机器预标注，不是当前远端回读。对 factory-4090 与双 5090 的只读 SSH 当前均因认证失败，无法声称远端现状或真实双卡 DDP smoke 已通过。预飞回执 `sam3-peft-preflight-20261111.json` SHA-256 为 `fc235b0f1ac9f0ad2935da5927a16887be8307f2241dd8932a3094089c239ac4`。

资源计划仅为：每卡 batch size 1、BF16 待服务器实测确认、gradient accumulation 1、启用 activation checkpointing；首批前后向必须实测峰值显存。不可启动的配置计划保存在 `configs/sam3_sfl_cpd15y_peft_plan.json`，其 Schema 强制 `execution_enabled=false`、禁止 `_preannotations.coco.json` 作为真值，并固定双卡 NCCL DDP。当前 `formal_training_authorized=false`、`formal_training_started=false`，无 PID/job、loss 或 checkpoint。

## 4. 进程边界

本轮未启动正式训练、GPU 占位、长期 hold 或新的推理服务。由于没有真实人工 mask 且双 5090 SSH 未认证，未产生训练进程，也无可报告的首批 loss/checkpoint。获得人工 `_annotations.coco.json` 并完成 split/manifest 门禁后，才执行一次性双卡 DDP smoke；smoke 通过后再启动正式训练并记录 PID/job、GPU 利用率、首批 loss、checkpoint 和训练结束后的显存释放状态。

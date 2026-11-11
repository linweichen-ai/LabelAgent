# Slope frame0 候选 overlay 与审查回执（2026-11-11）

## 当前结果

批次 A/000、camera0、frame0 已固定为第一条 `slope_hard_case`。工作台中现有 SAM 候选已只读导出为 overlay；候选、overlay 和原始帧均有独立 SHA-256，工作台候选与真值没有被修改。

- 原始帧：1920×1536 JPEG，SHA-256 `3775ee5f41374c9a302c458cfffa2799222e22ff17541045d6acfb8434bd682f`。
- slope 候选：1 个 shape、44 个外轮廓点、57,614 个像素，占全图 1.9536%，bbox 为 `(977,550)–(1410,780)`。
- SAM score：0.98046875，仅作为候选来源证据。
- 候选 SHA-256：`efbcb54360b5e9d9df622aa05522d456479cd227cc616d9b0833695b957e29d2`。
- overlay SHA-256：`ce0676eab1d73ad62bb1cc062fe1f0d99ee4a3e78600609c74aa12098fb0acfe`。
- overlay 文件：`C:\Users\seer\Documents\ChatGPT\Seer2clw\work\slope-frame0-hardcase\candidate-overlay.jpg`。

## 目视检查

助手目视检查认为候选基本覆盖可见黄色坡面主体；坡面与普通地面的前缘、右侧边界整体贴合，未看到大面积泄漏到普通 floor 或白色门板。货柜入口处的深色门槛/坡根是一条窄而困难的边界，是否存在少量欠分或过分无法脱离人工像素真值判断。

这不是人工批准。当前 `human_review.status=pending`、人工 truth 数为 0、该帧不是 reference frame，因此 IoU、Dice、boundary F1 和跨类泄漏像素全部保持 `null`。

## 闭环路由

该帧固定携带：

- `hard_case_tags=["slope"]`；
- `semantic_label=slope_visible`；
- `traversability_status=review_required`；
- `geometry_evidence_sufficient=false`。

后续顺序为：VLM 对 `slope_visible` 和混淆类复核 → LiDAR/车辆几何门禁 → Agent 生成 ROI、正负点或明确 no-op → 必要时 SAM 重分割 → VLM/质量复核 → 人工像素审核 → 独立 `truth_version`。没有 LiDAR 平面、法向、高差、坡度、宽度、边缘落差、障碍和车辆约束证据前，不允许输出 `traversable`。

当前工作台的 `/api/run` 会强制要求已保存的人工作真值参考帧和相应描述，而 `/api/state` 回读的 truth/reference 均为空。因此本次没有启动自动标注任务，也没有用现有 SAM 候选伪造参考真值。要继续真实全流程，必须先由人确认或修正至少一帧 slope 像素真值；如果 frame0 作为参考帧，则另选同一实体/序列外的 slope 帧作为闭环目标，避免把参考真值当作目标预测效果。

机器可读回执见 `slope-frame0-overlay-review-20261111.json`；专项指标与数据隔离规则见 `docs/SLOPE_SEGMENTATION_ACCEPTANCE.md`。

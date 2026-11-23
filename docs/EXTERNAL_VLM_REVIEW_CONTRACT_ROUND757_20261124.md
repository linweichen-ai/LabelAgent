# 外部专家候选复核协议（Round757，2026-11-24）

## 目的

为后续外部视觉模型影子 A/B 建立固定输出协议。专家模型只评价 LabelAgent 已有候选，不直接生成编辑操作、不修改掩膜、不添加正式类别，也不写人工真值。

## 输入

- 图像 1：原始帧；
- 图像 2：带候选 ID 的可视化覆盖图；
- 已知标签表；
- 候选 ID 与原类别绑定；
- 触发专家复核的原因代码。

提示词不会附带候选坐标或完整候选几何。每个候选必须且只能复核一次。

## 输出

每个候选只能给出以下结论之一：

- `supported`
- `wrong_class`
- `boundary_issue`
- `instance_merge`
- `instance_split`
- `missing_instance`
- `uncertain`

同时允许报告已知类别漏检和最多 5 个新类别提议。新类别只进入人工 ontology 复核，不自动修改标签表。

## 失败关闭

以下情况会拒绝整个专家结果：候选缺失、重复或未知 ID；候选 ID 与原类别不一致；非法置信度；非错类结论却提供新类别；新类别与已知标签同名；`candidate_acceptable` 与实际问题互相矛盾。

通过验证的结果仍固定为：

- `route=assistant_review_only`
- `candidate_only=true`
- `human_truth=false`
- `approved_for_training=false`
- `candidate_mutated=false`
- `ontology_mutated=false`

## 验证

- 6 项协议测试通过，覆盖提示词最小披露、动态 ID/标签约束、完整候选覆盖、未知/重复 ID、非法类别建议、新类别别名和输入候选不变。
- 所有模型调用均由测试替身完成，没有外部请求、图片上传、API Key、训练、推理、部署或真值修改。

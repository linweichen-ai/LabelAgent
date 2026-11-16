# 当前数据与训练范围

当前主测是常熟0920采集包：152 个时刻、四相机 608 图、152 帧点云及标定。用户选择 labels 的全部帧为参考；ali_label 和 wilddet3d-labels 是同一采集的其他三维框来源，不是新数据集。20 张调参与 588 张验证也属于同一约 84 秒采集，不代表独立场景。

本机历史 CVAT Task3 仍有 150 图、7 类、1,550 个标注。导入清单中 source 字段为 auto 1,469、manual 81；source 字段不能单独证明是否完成后续人工复核。旧评测比较的是与既有人工和自动混合标注的一致性，未检查远端 CVAT 当前版本。使用它训练前应核查类别、对象完整性和复核记录，保留来源及场景划分。

现有 SAM3 与 Qwen 权重没有在本项目中微调。NF4、图像预算、提示、约束解码与候选选择是推理或工程改动。较大 VLM 仍需用固定输入验证；本次历史 2B/8B 短测没有显示稳定容量收益，不能直接据此预测其他模型或配置。

后续 Qwen 任务微调需要复核过的类别判断和正确工具操作，SAM 适配需要与目标一致的二维框或掩码。三维框的投影只能提供弱监督，不能直接当作精确二维真值。应将误检、漏检、边界、重复实例与无效操作分开评测，并按采集场景或序列留出测试，避免相邻帧泄漏。强化学习还需要可靠反馈，点覆盖奖励单独不足以约束实例纯度和语义。

官方提供 [Qwen 微调及 LoRA](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md) 与 [SAM3 自定义数据训练](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md) 入口。目前未开始这些训练，也未创建付费云实例。

[本地清点的脱敏汇总](dataset_training_inventory.json)、[8B 对照](capacity-grounding.md)、[当前局部类别失败对照](vehicle-definition.md)。

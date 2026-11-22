# Round536：71项栈板局部训练候选包

日期：2026-11-22

## 本轮结果

对 Task3 中 81 个 `source=manual` 栈板实例完成两路助手视觉复核。主线与独立复核均检查了
14 张联系表，重点判断目标是否为栈板、轮廓是否存在明显大块漏切、是否粘入地面或别的栈板。
最终保留 71 项，暂缓 10 项，明显拒绝 0 项。暂缓项均因目标过小、贴裁剪边或实心叠加层遮挡，
不是判定标注错误。

这些结论严格记为 `assistant_reviewed`，不冒充人工真值，也不打开生产部署门。用户授权助手代做
试验判断后，本轮只把 71 项用于隔离候选训练；test24 没有进入数据包，正式 SAM3 权重保持不变。

## 为什么改为局部裁剪

原始 Task3 同一帧中仍可能存在其他未签核实例。如果直接用整帧训练，未标区域可能被错误当成负样本。
新构建器因此围绕每个通过实例生成带 12% 上下文的局部裁剪，每张裁剪只监督一个可见栈板实例：

- 71 张裁剪、71 个实例；
- 每张源图重新核对 SHA-256；
- 掩码转换为 COCO 非压缩 RLE，并保留面积与边界框；
- 只训练 SAM3 mask head 的 6 个张量，共 197,376 个参数；
- 计划双卡单轮 36 个优化步，训练后只生成离线候选检查点；
- 不自动部署，不允许 test24 回流。

公开回执位于
`artifacts/round536-cvat-task3-assistant-pallet-crops-20261122/receipt.json`；私有候选 ID、裁剪图、
COCO 与复核决定只保存在 `.repro/`，不会推送到 GitHub/CNB。

## 新增可复现代码

- `tools/build_sam3_assistant_instance_crop_dataset.py`：失败关闭的数据构建器；
- `source/sam3/sam3/train/stage0_scope.py`：冻结全模型，只开放 6 个 mask-head 张量；
- `source/sam3/sam3/train/stage0_trainer.py`：优化器范围守门；
- `source/sam3/sam3/train/assistant_instance_crop_trainer.py`：限定 71 图、36 步、单轮训练；
- `source/sam3/sam3/train/configs/sfl_cpd15y/stage0_assistant_instance_crop_epoch.yaml`：隔离训练配置；
- `tools/run_sam3_assistant_instance_crop_train.sh`：服务健康、test24 路径和显存门禁；
- 6 项定向测试全部通过，Python 编译和 Git 差异检查通过。

首次从 GitHub 干净克隆启动时，Hydra 在加载模型前立即发现 Stage-0 基础配置引用少了
`configs/` 层级；该尝试没有加载模型或占用训练显存。随后已把引用改为仓库内实际存在的
`/configs/roboflow_v100/...` 并加入干净克隆依赖测试，不隐藏这次失败。

## 当前边界与下一步

这批数据来自 CVAT 中 `source=manual` 的实例轮廓，但本轮没有伪造 81 项人工签字，所以训练结果仍是
“助手代审候选”。训练完成后必须先与原始 SAM3 在未参与训练的栈板困难集上做同提示 A/B，至少同时
检查 IoU、边界 F1、孔洞保留和地面误吞；任何一项明显退化就保留原始模型。

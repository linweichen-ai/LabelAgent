# 参考文献与上游依赖

本页列出 LabelAgent 方法、实现和复现说明所依赖的主要一手资料。项目的本地实验结果以 `docs/` 中带轮次编号的记录为准；上游论文指标不等于本项目在工厂数据上的实测效果。

## 1. 分割模型

1. N. Carion et al., “SAM 3: Segment Anything with Concepts,” *arXiv preprint arXiv:2511.16719*, 2025. <https://arxiv.org/abs/2511.16719>
2. Meta AI, “SAM 3 official implementation,” 2025. <https://github.com/facebookresearch/sam3>
3. Meta AI, “SAM 3 training guide.” <https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md>

本项目使用 SAM3 生成候选掩膜，并在人工标注中使用点、框和负提示。上游模型支持微调不代表本项目已经完成可部署的 SAM3 微调；正式模型替换仍需独立测试集验收。

## 2. 视觉语言模型

4. S. Bai et al., “Qwen3-VL Technical Report,” *arXiv preprint arXiv:2511.21631*, 2025. <https://arxiv.org/abs/2511.21631>
5. Qwen Team, “Qwen3-VL official implementation.” <https://github.com/QwenLM/Qwen3-VL>
6. Qwen Team, “Qwen3-VL fine-tuning guide.” <https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune>
7. Qwen Team, “Qwen3-VL-8B-Instruct model card.” <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>

本项目以本地 Qwen3-VL-8B 处理普通复查，外部专家模型仅用于受控困难帧。Round375 的 2B LoRA 是离线候选，不是已部署模型，也不能代替人工真值训练。

## 3. 数据与文件格式

8. Foxglove, “MCAP specification and SDK documentation.” <https://mcap.dev/>
9. PyTorch, “PyTorch documentation.” <https://pytorch.org/docs/stable/index.html>
10. Hugging Face, “Transformers Qwen3-VL documentation.” <https://huggingface.co/docs/transformers/model_doc/qwen3_vl>

## 4. 引用与许可证边界

- SAM3 源码与模型使用受其上游许可证和模型访问条款约束；
- Qwen 模型与代码分别遵循其模型卡和仓库许可证；
- 模型权重、工厂图片、MCAP、人工真值和账号凭证不随代码分支公开；
- 本项目文档对上游能力的描述仅用于说明技术来源，不把上游公开指标当作 LabelAgent 的实测指标。

复现者应在使用或再分发模型资产前重新阅读相应上游许可证与模型卡；仓库中的哈希清单用于确认文件身份，不替代许可证授权。

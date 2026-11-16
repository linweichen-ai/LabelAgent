# 用户实际GPU：双RTX PRO 5000 72GB Blackwell

2026-11-04用户终端回传：两张NVIDIA RTX PRO 5000 72GB Blackwell，每张73415MiB；driver580.126.09，Python3.12.12，nvcc12.8.93。原上传ZIP已由用户执行verify_bundle与60例校验成功。

原包按L20N候选方案准备，安装和训练固定第三方FlashAttention-2。其官方CUDA支持列表列Ampere/Ada/Hopper，不能据此宣称该Blackwell卡已兼容。本补充首先建立torch2.8.0/cu128、transformers4.57.6、PEFT0.17.1环境，使用PyTorch SDPA验证BF16矩阵乘、注意力前向与非零有限梯度。无需安装第三方flash-attn/DeepSpeed，不修改原包。

用户允许把数据存数据盘。当前df只证实/mnt/workspace与98GB根文件系统相同，尚未提供数据盘挂载点。必须先读df -hT、lsblk输出，再将LABELAGENT_BLACKWELL_ROOT设成该数据盘上的全新项目目录。不要猜测/mnt/data已挂载，也不要把tmpfs当作项目磁盘。

```bash
# 用核实后的数据盘挂载路径替换下面占位符
export LABELAGENT_BLACKWELL_ROOT=/实际数据盘挂载点/labelagent-blackwell-20261104
bash setup_blackwell_sdpa.sh 2>&1 | tee blackwell-setup.log
```

安装器要求至少50GiB空闲，用于环境和后续小模型试验；完整2B/8B/32B对照建议项目盘预留500GiB，开始大模型下载前核验实际可用量。虚拟环境、权重、HuggingFace/Torch缓存及临时文件都放在该项目目录；后续终端运行先source该目录env-blackwell.sh。安装器不改宿主驱动、不停止其他进程、不修改原始标注，不自动下载模型。

预检通过只证明基础GPU算子可执行和模型类可导入。**尚未完成云端安装、真实Qwen推理、LoRA训练和原训练入口的SDPA适配。不要运行原train_qwen.sh / run_qwen.py / evaluate_qwen.py来假定使用了SDPA**，它们仍显式使用flash_attention_2。后续需提供独立、可核验的SDPA入口，然后再做真实模型forward/backward与适配器保存/重载。

官方依据：
- https://pytorch.org/blog/pytorch-2-7/ （Blackwell及CUDA12.8预编译支持）
- https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support （FA2架构支持列表）

本地Python AST/bash语法检查可做；用户PAI上的实际执行仍待结果回传。原始ZIP及其摘要保持不变。

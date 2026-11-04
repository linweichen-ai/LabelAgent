# LabelAgent：PAI首轮探索与训练包

本包状态：本地数据准备和工具检查；尚未在PAI安装、推理或训练。它用于先检验“大模型是否看得更准”，再检验“小模型能否通过任务微调改善”。所有模型输出保持候选身份。原始标注、现有工作台、SAM3服务均不修改。

## 1. 首轮申请什么资源

| 项目 | 首选配置 | 可用替代 |
|---|---|---|
| PAI产品 | DSW先调通，随后可转DLC复现实验 | 不需要先建训练集群 |
| GPU | 同一实例2张完整GPU，每卡80GB或以上 | 现有L20N：2张完整48GB卡；降低微批量、顺序运行 |
| CPU / RAM | 32核 / 256GiB | 32核 / 128GiB |
| 共享内存 | 32GiB | 16GiB起，实际核验 |
| 持久化项目存储 | 500GiB；放权重、数据、检查点、日志 | 不把模型都装进小系统盘 |
| 镜像 | Linux x86_64，Ubuntu22.04，Python3.11，兼容CUDA12.8的GPU环境 | 实际先跑CUDA/BF16预检 |

这些是本项目首轮工程预算，不是官方最低显存要求。2B/8B LoRA默认单卡；`NPROC_PER_NODE=2`是DDP，每卡各放一份模型，**不会把显存合并**。32B BF16评测通过device_map自动分片，显存余量取决于图片网格与上下文。先申请这一台，不必先占8卡。DSW调通再保存环境、镜像摘要和数据版本，DLC使用相同入口；本包未自动创建DSW或提交DLC作业。

## 2. 包内有什么

- `review-data/`：60个未确认候选，20张既有调参图、5个时间组；含完整场景与局部图。模型轮廓不等于人工掩码。
- `review-data/review.html`：双击打开，填写审核人，按1—5审核，下载`reviewed.jsonl`；可导入之前的JSONL接着审。
- `data_tools.py`：校验来源、图片摘要、审核状态，按时间组导出训练/开发集。
- `setup_qwen.sh`：独立环境安装脚本，官方训练源码锁到提交，模型版本另行锁定。
- `download_model.py`、`train_qwen.sh`：下载模型、10步训练测试、2轮LoRA小试验。
- `evaluate_qwen.py`：相同输入比较基座和适配器，记录严格JSON、正确率、误改、拒答、图像网格与耗时。
- `inference_reference/`：已有32B API部署材料，作为LabelAgent后续接入参考；与训练环境分开。
- `SAM3_DIAGNOSTIC.md`：SAM3训练之前必须完成的分割诊断和数据条件。
- `experiment_plan.json`：首轮对照、预算、验收与未完成事项。
- `manifest.sha256.json`：包内文件摘要；`verify_bundle.py`负责校验。

数据包是私人本地交付，不上传到公共GitHub。`review-data`只含所需图片及候选元信息，不含账号、API配置、原始对话或服务器密码。

## 3. 第一步：先审核，不能直接拿候选训练

打开`review-data/review.html`，审核60例。黄色轮廓用于指定当前候选，只审核类别及是否可保留；有边界问题在备注注明。按钮：保留、拒绝、改为人员、改为托盘、不确定。下载`reviewed.jsonl`，放在`review-data/`里。

```bash
python data_tools.py validate --records review-data/reviewed.jsonl --data-root review-data --require-reviewed
python data_tools.py export --records review-data/reviewed.jsonl --data-root review-data --out prepared/pilot-v1 --dev-group Capture_20261103:block_004
```

现成60例只够数据/训练流程小试，不足以证明通用质量。默认至少24条训练、6条开发样例，开发组固定为第135帧所有相机；其他时间组用于训练。后续每轮用新输出目录，禁止根据开发结果反复换组。这套图已被调参分析过，**没有独立测试集**。正式训练目标为300—500条人工复核决策样本，覆盖保留、错类、错误候选、不确定。另采一段序列做独立测试；同一时刻所有相机及相邻帧不得分到不同集合。Task3的auto/manual来源字段不能替代人工审核。

## 4. DSW安装与下载

上传并解压ZIP到持久化目录，如`/mnt/workspace/LabelAgent-PAI-Exploration-20261104`。在终端进入该目录。安装需要联网，第一次编译flash-attn可能较慢；安装失败保留日志，不修改其他项目环境。

```bash
cd /mnt/workspace/LabelAgent-PAI-Exploration-20261104
python3 verify_bundle.py
export LABELAGENT_PAI_ROOT=/mnt/workspace/labelagent-pai-exploration-20261104
bash setup_qwen.sh 2>&1 | tee setup-qwen.log
export PY="$LABELAGENT_PAI_ROOT/venv-qwen/bin/python"
"$PY" download_model.py --size 2B --models-root "$LABELAGENT_PAI_ROOT/models"
"$PY" download_model.py --size 8B --models-root "$LABELAGENT_PAI_ROOT/models"
```

训练栈候选：torch2.8.0/cu128、transformers4.57.6、PEFT0.17.1、flash-attn2.8.3；版本存在性和脚本参数在本地核对，GPU兼容性需PAI实测。脚本会核对CUDA张量计算、BF16、包导入及实际显存，并记录环境。不要把预检通过当作训练通过。

## 5. Qwen LoRA：先10步，再小试

```bash
export LABELAGENT_DATA_ROOT="$PWD/review-data"
export LABELAGENT_EXPORT="$PWD/prepared/pilot-v1"
MODEL_SIZE=2B NPROC_PER_NODE=1 bash train_qwen.sh smoke
MODEL_SIZE=2B NPROC_PER_NODE=1 bash train_qwen.sh pilot
# 2B流程跑通后，在同一人工数据上做8B对照
MODEL_SIZE=8B NPROC_PER_NODE=1 bash train_qwen.sh smoke
MODEL_SIZE=8B NPROC_PER_NODE=1 bash train_qwen.sh pilot
```

LoRA r16/alpha32/dropout0.05；学习率1e-5，微批量1，梯度累积8，种子3407，试验最多2轮。冻结基座，审计仅语言侧LoRA可训练。10步测试时限30分钟，小试时限6小时；超时退出不算训练完成。每次新目录，保存数据摘要、版本、训练参数、日志与适配器。`TRAIN_COMPLETED.json`只表示训练程序完成，不表示质量提升。首轮参数是待验证起点；之后根据开发集结果决定至多再试一个学习率，禁止无上限扫参。

## 6. 相同输入评测：基座、LoRA、32B

```bash
"$PY" evaluate_qwen.py --model "$LABELAGENT_PAI_ROOT/models/Qwen3-VL-2B-Instruct" --records prepared/pilot-v1/dev_reviewed.jsonl --data-root review-data --out results/2b-base-v1
# 替换下面路径为本次pilot实际输出的adapter目录
"$PY" evaluate_qwen.py --model "$LABELAGENT_PAI_ROOT/models/Qwen3-VL-2B-Instruct" --adapter /mnt/workspace/labelagent-pai-exploration-20261104/runs/实际运行目录/adapter --records prepared/pilot-v1/dev_reviewed.jsonl --data-root review-data --out results/2b-lora-v1
"$PY" download_model.py --size 32B --models-root "$LABELAGENT_PAI_ROOT/models"
"$PY" evaluate_qwen.py --model "$LABELAGENT_PAI_ROOT/models/Qwen3-VL-32B-Instruct" --records prepared/pilot-v1/dev_reviewed.jsonl --data-root review-data --out results/32b-base-v1
```

基座和LoRA用相同BF16、同一提示/图片字节、相同max_pixels451584和输出192上限；实际image_grid_thw会逐例记录。后续单独把32B的`--max-pixels`提高至1048576再比较，不把规模、精度和输入密度的变化混为一谈。旧NF4结果保留为历史实际部署基线。

这些脚本评测的是候选复查任务，输出`decision/label`，**不是现有Agent的operations协议**。因此不能直接替换在线模型；须另做协议转换和离线操作回放，检查删除/重分类是否正确，再在20帧上按每帧最多3轮验证完整闭环。未实现的桥接不会被本包宣称为完成。

首轮继续门槛：开发集按存在的决策类别统计的平衡正确率较同精度基座提高至少10个百分点、固定正确人员/托盘样例中不新增破坏性操作、严格JSON合法率100%。这是进入独立测试的筛选条件，小样本通过不能证明泛化。`uncertain`对可判定样例按错误/未解决统计，不通过多拒答抬高分数。正式验收另需独立序列、人工二维指标、原有回归门槛与人工修改时间；人员误标单列，原有门槛保持。

```bash
python compare_results.py --baseline results/2b-base-v1 --candidate results/2b-lora-v1 --out results/2b-lora-comparison-v1.json
python compare_results.py --baseline results/2b-base-v1 --candidate results/32b-base-v1 --out results/32b-comparison-v1.json
```

比较器核对同一审核文件、案例顺序和真实图像网格；输入密度试验才显式加`--allow-density-change`。样本少时要同时查看逐例得失，不能只看平均数。

32B作为强模型基线和可审核的教师候选；其答案需要人工确认后才进入训练集，不直接拿自己生成的答案自评。SAM3先做诊断再决定训练，详见下一份文档。强化学习暂缓，不把稀疏投影覆盖直接用作奖励。

## 7. 交回哪些结果

把`logs/`、对应`runs/`的日志/适配器/数据摘要、`results/`、人工审核JSONL打包返回。模型基座权重无需回传，保留模型版本号即可。报告失败、显存峰值、耗时、实际有效修改及人员误改，不只报告loss下降或done。开始前保留独立测试集；结果不满意也保留，不覆盖历史实验。

## 官方来源

- PAI DSW：https://help.aliyun.com/zh/pai/create-and-manage-dsw-instances
- PAI DLC：https://help.aliyun.com/zh/pai/create-a-training-task
- Qwen训练：https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- SAM3训练：https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md

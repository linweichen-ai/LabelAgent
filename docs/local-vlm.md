# 本地视觉模型接入与真实 Agent 验收

## 当前状态（2026-11-04）

远端部署目标是用户已有的双 RTX 4090 服务器。本轮 SSH 登录、主机名和双卡资源读取已成功；先前会话超时的根因尚未确认。两卡均驻留已有服务，保留原有任务。在新的独立用户目录中复制 Python 运行时、创建虚拟环境并准备官方 8B 权重，后台准备进程已核验真实运行。环境与权重的具体完成阶段见 [远端准备状态](test-results/remote_vlm_progress.json)。尚未启动远端 GPU 模型、进行真实推理或切换工作台。

本机路线继续使用 Qwen3-VL-2B NF4 与 SAM3，见 [本机运行](local-gpu.md) 和 [实际结果](test-results/local_gpu_results.md)。远端采用固定版本 Qwen3-VL-8B-Instruct，保存官方原始权重，计划在推理加载时量化；CUDA 路径与质量仍需实测。现有空闲卡检查保持启用，不停止驻留服务以腾出显存。

## 模型和运行范围

准备使用官方 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 稠密模型，通过 Transformers 提供兼容图片聊天请求的本地接口。实现依据 [Qwen 官方仓库](https://github.com/QwenLM/Qwen3-VL) 和 [Transformers 4.57.1 文档](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl)。是否能在目标机器稳定运行、耗时和显存峰值须实测，尚未验证。

- 选择一张空闲 4090，不默认占用双卡；另一张保留现有工作。
- 使用独立用户目录和虚拟环境，不替换共享环境，不停止已有计算进程。
- 服务仅绑定 `127.0.0.1:18868`，Windows 通过 SSH 隧道访问。
- 推理只从本地目录加载权重，并设置 Hugging Face/Transformers 离线模式；模型首次下载单独进行。
- 请求仅接受内嵌 JPEG/PNG，不抓取外部图片 URL；限制为八张图片、8192 总 token 和最多 4096 输出 token。默认输出预算 2048。
- 每张图片的处理预算为 64～512 个 32×32 像素块面积，不改变调用方要求的原图坐标约定。细节是否足够仍需真实标注验证。
- 同时仅执行一个生成请求；忙碌时明确返回错误，不静默排队或转到云端。

## 服务器恢复后的步骤

以下是剩余步骤及复现说明。清单读取和独立环境准备已开始，实际阶段以准备状态为准。`<...>` 表示需要根据实际清单填写的值，不能原样执行。

1. 先确认普通 SSH 能返回主机名，再读取服务器用户目录内适用的 `AGENTS.md` 和现有任务信息。
2. 查看 Python 版本、`nvidia-smi`、磁盘可用空间和已有用户环境。为本项目选择新的独立目录，例如 `~/labelagent-local-vlm`，不得覆盖别人的目录。
3. 将本分支的工具放入该目录；创建独立虚拟环境。根据实测驱动安装匹配的 CUDA 版 PyTorch 和 torchvision，再安装：

```sh
python -m pip install -r tools/requirements-local-vlm.txt
python tools/vlm_resources.py
```

资源清单包含 GPU UUID、空闲显存、利用率及计算进程存在性。选择明确空闲的 UUID，启动工具会再次检查：必须是 4090，没有现存计算进程，利用率不超过 10%，默认空闲显存至少 22000 MiB。条件不足则退出，不结束任何进程。该检查不构成 GPU 预订；共享服务器仍需核对实际占用。

4. 从官方模型仓库读取并记录完整提交 SHA，再下载固定版本。示例：

```sh
hf download Qwen/Qwen3-VL-8B-Instruct --revision <verified-model-commit> --local-dir <private-model-directory>
python -m pip freeze > <private-deployment-directory>/installed-packages.txt
```

记录下载版本、文件摘要、GPU/驱动、Python/PyTorch/CUDA 版本；不把模型权重或本机私有清单提交到 Git。

5. 在服务器的独立环境前台验证启动：

```sh
python tools/local_vlm_server.py --model-dir <private-model-directory> --gpu-uuid <idle-GPU-UUID>
```

默认服务模型名为 `labelagent-vlm`。加载失败、CUDA 不可用或请求超出上下文预算都会明确报错。启动成功后仍需真实图片探测和完整 Agent 验收；`/health` 成功本身不证明标注流程完成。持久化后台启动应在前台验收后设置。

## Windows 连接与工作台配置

建立隧道，使用已经配置的服务器别名：

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18868:127.0.0.1:18868 <configured-server-alias>
```

在本机 LabelAgent 环境先验证真实图像请求，再更新工作台：

```sh
python tools/connect_local_vlm.py --image <local-image.png> --output work/vlm-probe.json
python tools/connect_local_vlm.py --image <local-image.png> --apply --output work/vlm-configured.json
```

`--image` 可重复提供多张图，Agent 复核通常同时携带原图、候选、空洞图及两张参考图，因此还需五图探测。工具要求服务声明真实 CUDA 后端并确认收到全部图片，返回可解析 JSON；它拒绝将接口测试替身配置到工作台。仅在 `--apply` 且工作台空闲时写入配置，以无秘密的 `local` 值替换旧模型密钥。

当前工作台已配置本机 2B 模型，占用本机 18868 入口。未来验证远端隧道时需选另一空闲端口，探测后再切换配置。`real_vlm_probe_passed` 只表示模型连通和图片请求成功，完整流程另看实际验收汇总。

## 数据集上的完整流程验收

采集包只有三维框，二维参考轮廓须单独提供。使用 JSON 形状数组，结构与工作台一致：每项包括唯一整数 `id`、`label`、原图像素 `points`，可附 `holes`/`parts`。类别须覆盖请求的全部标签。标注必须来自所选话题和参考帧。

参考来源必须显式区分：`human_reviewed` 表示调用者提供经过人工复核的二维参考；`unconfirmed_functional_fixture` 表示仅用于功能验收的未确认示例。不得把现有 SAM3 候选或三维 `labels` 自动当成人工二维真值。当前尚无新增的人工二维参考。

```sh
python tools/run_local_agent_acceptance.py --dataset work/datasets/example --reference-json <reference-2d.json> --reference-kind human_reviewed --frames 76,151 --output work/agent-acceptance-new
```

默认使用 `camera_1` 第 0 帧为参考，第 76、151 帧为目标，最多五轮；`--topic`、`--reference-frame`、`--labels`、`--frames` 和 `--max-rounds` 可调整。每个话题分别提供对应参考，不能将某一相机轮廓移用于其他相机。输出目录必须是新目录。

工具直接调用与工作台相同的 `workflow.describe` 和 `workflow.run`：真实 VLM 生成规范 → SAM3 候选 → VLM 复核 → 按决定执行局部修改 → 再观察。若模型认为无需修改，允许直接结束；不会为了展示多轮而强制篡改候选。

输出包含规范、每次模型调用、逐轮事件、候选轮廓、候选预览和 `public_summary.json`。原图在前后校验摘要。该独立验收不写生产工作台状态、不保存真值；后续通过工作台界面还需分别验收候选展示和刷新恢复。

完成条件：

- 真实模型图片请求通过，真实规范生成成功。
- 每个目标都有成功的 SAM3 和 VLM 调用、至少一轮复核，并以 `done` 收敛。
- `max_rounds`、调用异常、修改循环或无进展均不报告为完整成功。
- `multi_round_correction_observed` 只有确实应用修改并完成下一轮复核时才为真。
- 未确认示例只能证明该示例上的功能路径；即使收敛，也不证明像素准确率或达到人工质量。
- 原始三维参考、图像、既有会话和真值保持完整。逐对象记录和图片留在本机，只同步脱敏汇总。

扩大到全部 608 张图之前，先检查首、中、末帧的真实调用质量、显存峰值、耗时、失败原因和参考适用范围，再确定逐相机的批量执行预算。

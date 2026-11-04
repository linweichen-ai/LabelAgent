# LabelAgent：PAI DSW 的 L20N 双卡部署与接入

日期：2026-11-04。状态：部署说明及脚本已准备；尚未在用户 PAI 实机安装或验证。脚本的本地语法检查不等于云端部署成功。本包不含账号、密钥、数据集、模型权重或原始标注，也不会自动创建计费实例。

采用本机 LabelAgent 工作台与 SAM3，云端部署 Qwen3-VL-32B-Instruct，使用 SSH 本地转发连接视觉模型。先验证推理与协议，再切换工作台配置。已确认的参考轮廓、原标注和 2B 基线需要继续保留。

## 1. 按截图创建实例

截图中选择了现有 L20N 资源配额。L20N 官方标称每卡 48GB；本次采用同一实例内两张完整卡及显式张量并行。资源池表中的节点/卡数是概览，下方 GPU（卡数）才是实例申请值。共享显存切片不能当作两张完整 48GB 卡。

| 页面字段 | 本次建议值 | 原因 |
| --- | --- | --- |
| 地域 | 沿用现有资源配额所在地域 | 不为此强行迁移资源池 |
| 实例名称 | labelagent-vlm32b-l20n2 | 便于识别任务和资源 |
| 标签 | project=LabelAgent，purpose=vlm-evaluation | 按团队允许的标签填写 |
| 资源类型 | 资源配额 | 使用当前已有配额；不自动切到公共付费资源 |
| 资源配额 | 当前可用的 L20N 配额 | 以配额权限和当前空闲量为准 |
| GPU（卡数） | 2 | 两张完整 L20N，同一实例可见 |
| CPU（核数） | 16；有余量可选 32 | 本次推理起步预算，并非官方最低要求 |
| 内存（GiB） | 128 | 权重载入、图像处理和服务进程预留 |
| 共享内存（GiB） | 16 | /dev/shm，用于进程间通信；不是 GPU 显存 |
| 优先级 | 保持团队默认；无要求时 1 | 不擅自抢占其他任务 |
| 指定节点调度 | 先关闭 | 两卡通常在实例所在同一节点；调度问题再查 |
| 闲时资源 | 关闭 | 先建立稳定复查服务 |
| 官方镜像 | Ubuntu 22.04/24.04、Python 3.11/3.12、较新的 NVIDIA GPU 镜像 | 另建独立环境，匹配 vLLM 依赖 |
| 系统盘 | 无额外数据盘时建议 200GiB | 截图 50GiB 不够放 BF16 32B 权重与环境 |
| 其他存储 | 如只能用 50GiB 系统盘，另挂至少 200GiB 的项目存储 | 模型目录必须实际位于大容量挂载点 |
| 可见范围 | 仅实例所有者或团队约定范围 | 按实际协作方式选择 |
| 实例 RAM 角色 | 沿用工作空间默认的合适角色 | 公共模型推理不需要新增管理员权限 |
| 自动保存镜像 | 首次先保持默认；跑通后再保存环境 | 不把模型权重和运行缓存都塞进环境镜像 |

截图当前为 PyTorch 2.3.1 / Python 3.11 / CUDA 12.1 的旧环境，不建议直接在其全局环境安装。可在官方镜像列表寻找较新的 PyTorch GPU 镜像，例如官方创建文档列出的 PyTorch 2.8.0 / Python 3.11 / CUDA 12.4 系列。下面仍会独立安装与 vLLM 匹配的 PyTorch；镜像里的 CUDA 工具包版本、PyTorch 运行时版本及宿主机驱动能力不是同一个概念。

如果该配额只能选择截图中的镜像，可以保留其 Python 3.11 底座，严格使用下面的独立环境安装；是否可运行仍由实际驱动及安装后的 CUDA 检查决定，不能仅凭旧镜像标签判断宿主机驱动过旧。

若只有一张 96GB 卡，仍可用 BF16 32B，但设置 LABELAGENT_TP=1。若只有单张 48GB 卡，本说明的 BF16 32B 配置不成立；需要另选量化配方或较小模型。两张卡必须使用 tensor-parallel-size=2，显存不会自动拼接。

## 2. 访问配置

本方案让模型仅监听云实例的 127.0.0.1:18878，通过 SSH 转发，不直接把模型端口开放公网。

优先复用团队已配置的 VPC、交换机与 SSH 网络。在访问配置中启用 SSH，将 Windows 本机的公钥粘贴进去，监听端口按控制台默认 22。SSH 外部访问端口和实例内部 22 可能不同，后续使用控制台生成的实际连接命令。

Windows PowerShell 生成独立密钥，不覆盖已有密钥：

```powershell
$key = Join-Path $env:USERPROFILE '.ssh\labelagent_pai_ed25519'
New-Item -ItemType Directory -Force (Split-Path $key) | Out-Null
if (-not (Test-Path -LiteralPath $key)) {
    ssh-keygen -t ed25519 -f $key -C 'labelagent-pai'
}
Get-Content -LiteralPath ($key + '.pub')
```

只把 .pub 内容放到 PAI 公钥框。私钥和口令保留本机。访问推理服务的 API Key 是稍后自己设置的值，不是阿里云 AccessKey。

如果没有现成公网 SSH 网络：

- SSH 直连需要合适的 VPC 网络以及公网入口，可使用团队已有 NAT/EIP/NLB；这些资源有独立费用。
- ProxyClient 是官方的另一种 SSH 访问方式，可不用另建 NAT/EIP，但要按官方文档安装客户端并配置认证。
- 当前可先创建并使用 DSW WebIDE 完成云端验证，随后再建立 SSH 通道；不能将 WebIDE 的登录页面地址直接填进 LabelAgent API 地址。

## 3. 创建后，先在 DSW Terminal 检查

以下命令均在云端 Linux 执行：

```bash
nvidia-smi
python3 --version
df -hT / /mnt/workspace /dev/shm
nvidia-smi topo -m
```

应看到两张 L20N、每卡约 48GB，以及足够磁盘。以实例实际可见 GPU 和空闲显存为准。保留 PAI 设置的 CUDA_VISIBLE_DEVICES；不要为了凑两卡覆盖该变量，也不要停止其他项目进程。若只看到一张卡或显存被切分，先处理资源分配。

重点检查模型存储目录究竟在哪个挂载点、是否有足够空间，以及 /dev/shm 是否约 16GiB。系统内存 128GiB 和共享内存 16GiB 都不能替代 GPU 显存。

## 4. 上传本包并安装独立环境

将 tools/pai_dsw 中四个脚本上传到 DSW 的同一目录，或使用会话提供的部署 ZIP。假设上传到 /mnt/workspace，执行：

```bash
cd /mnt/workspace
python3 -m zipfile -e LabelAgent-PAI-Setup-20261104.zip labelagent-pai-setup
cd /mnt/workspace/labelagent-pai-setup
bash install_cloud.sh
```

脚本使用独立项目目录 /mnt/workspace/labelagent-pai-vlm32b，创建 .venv，不改系统 Python。如果该环境目录已存在，脚本停止，不覆盖；可以复用已验证环境或设置一个新的 LABELAGENT_PAI_ROOT。

安装使用 uv 根据驱动选择 PyTorch 后端，安装当前稳定 vLLM 与 ModelScope，不使用 nightly。CUDA 二进制与驱动仍必须匹配；若安装或导入报驱动错误，需要结合 nvidia-smi 选择合适的官方构建，不要在共享 PAI 宿主机上自行升级驱动。

安装后确认：

```bash
source /mnt/workspace/labelagent-pai-vlm32b/.venv/bin/activate
python -c 'import torch,vllm,transformers; print("torch",torch.__version__,"cuda",torch.version.cuda,"vllm",vllm.__version__,"transformers",transformers.__version__,"gpus",torch.cuda.device_count())'
python preflight_pai.py --expected-gpus 2
```

安装版本记录位于项目 logs/installed-packages.txt。首次跑通后固定这些版本，后续对照不要自动升级。

## 5. 下载正确模型并核验

```bash
export LABELAGENT_PAI_ROOT=/mnt/workspace/labelagent-pai-vlm32b
source "$LABELAGENT_PAI_ROOT/.venv/bin/activate"
mkdir -p "$LABELAGENT_PAI_ROOT/models" "$LABELAGENT_PAI_ROOT/logs"
modelscope download --model Qwen/Qwen3-VL-32B-Instruct \
  --local_dir "$LABELAGENT_PAI_ROOT/models/Qwen3-VL-32B-Instruct"
python preflight_pai.py --expected-gpus 2 \
  --model-dir "$LABELAGENT_PAI_ROOT/models/Qwen3-VL-32B-Instruct" \
  --hash-weights --output "$LABELAGENT_PAI_ROOT/logs/model-preflight.json"
```

选择 Qwen3-VL-32B-Instruct，名称里要有 VL，并选择 Instruct。本次不混入纯文本 Qwen3-32B、Thinking、MoE 或另一种量化模型。模型约 33.4B 参数，BF16 权重约 66.7GB（十进制粗估），还要给视觉编码、激活、KV 缓存和运行时留空间。

预检会检查 CUDA、GPU 数、可用显存、共享内存、模型类型和分片是否齐全，并可记录实际权重摘要。摘要可冻结本地文件，不能单凭自算摘要声称已独立认证模型来源。下载日志和模型页面版本也应保留。

## 6. 启动模型服务

首次使用前台运行，方便直接看报错：

```bash
cd /mnt/workspace/labelagent-pai-setup
export LABELAGENT_TP=2
export LABELAGENT_MAX_PIXELS=262144
bash start_vlm32b.sh
```

脚本会隐藏输入提示，要求设置自己的 VLM API Key。这个值稍后在本机探测和 LabelAgent 设置中使用，不要将其贴进聊天或 GitHub。脚本不打印 API Key。

核心配置为：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| served-model-name | labelagent-vlm32b | 工作台填写的模型名，必须一致 |
| host / port | 127.0.0.1 / 18878 | 仅实例回环，通过 SSH 转发 |
| tensor-parallel-size | 2 | 两张卡显式分担同一模型 |
| dtype | bfloat16 | 首轮使用较高精度 |
| gpu-memory-utilization | 0.85 | vLLM 的每卡内存预算比例 |
| max-model-len | 16384 | 输入和输出合计上下文上限 |
| max-num-seqs | 1 | 起步串行，不追求并发 |
| limit-mm-per-prompt | image=8、video=0 | 容纳多图 Agent 复查 |
| min_pixels / max_pixels | 65536 / 262144 | 初始图像处理上限，对应约 256 编码预算档 |
| generation-config | vllm | 避免模型仓库默认生成配置额外介入 |
| seed | 0 | 记录随机种子，不能保证跨实现逐字一致 |
| enforce-eager | 开启 | 首轮减少图编译和 CUDA Graph 的调试变量 |
| disable-custom-all-reduce | 开启 | 先使用常规通信路径验证 PCIe 双卡 |

首次启动需要加载权重与初始化，进程存在不等于可推理。等 API 服务已启动，并完成后面的真实图片请求才算这一层通过。

若要后台运行，在同一个云端 Terminal 中隐藏输入 Key 并仅导出到当前环境：

```bash
read -r -s -p 'VLM API key: ' VLLM_API_KEY
printf '\n'
export VLLM_API_KEY
nohup bash start_vlm32b.sh > /mnt/workspace/labelagent-pai-vlm32b/logs/vlm32b.log 2>&1 &
printf '%s\n' "$!" > /mnt/workspace/labelagent-pai-vlm32b/logs/vlm32b.pid
```

查看日志：

```bash
tail -n 80 /mnt/workspace/labelagent-pai-vlm32b/logs/vlm32b.log
```

前台与后台方式二选一，不能同时启动两份模型。前台可用 Ctrl+C 结束自己刚启动的服务。后台停止前先核对 PID 和项目命令，不使用 pkill python 或 killall。

## 7. 云端先验证接口

保留服务运行，另开 DSW Terminal：

```bash
curl -i http://127.0.0.1:18878/health
```

原生 vLLM 的 health 可能是 HTTP 200 空响应，不会返回本项目自写的 backend/model_loaded 元数据。它只证明服务就绪，不能代替真实图片推理。

上传一张用于测试的 PNG/JPG 到云端，然后执行：

```bash
source /mnt/workspace/labelagent-pai-vlm32b/.venv/bin/activate
cd /mnt/workspace/labelagent-pai-setup
python probe_vlm_api.py --image /mnt/workspace/test_frame.png \
  --output /mnt/workspace/probe-cloud-one.json
```

图片路径替换为真实文件。Key 在隐藏提示中输入。脚本先检查 /v1/models，再发送真实图片与 JSON Schema，要求正常结束、严格 JSON、模型别名一致和图片数量正确；它不改 LabelAgent。

API 通过不代表识别正确。读一下 answer 的类别和理由是否符合图片。随后使用多张不同的真实输入图片重复 --image，验证多图：

```bash
python probe_vlm_api.py \
  --image /mnt/workspace/target.png \
  --image /mnt/workspace/target_overlay.jpg \
  --image /mnt/workspace/reference.png \
  --image /mnt/workspace/reference_outline.jpg \
  --output /mnt/workspace/probe-cloud-four.json
```

## 8. Windows 建立 SSH 转发

先用 PAI 控制台生成的实际命令登录成功。公网连接端口不一定是 22；使用 ProxyClient 的连接配置时也要保留它的 ProxyCommand。将已验证连接存为 SSH 别名 labelagent-pai。示意配置中 HostName、Port 和 User 用实际值：

```sshconfig
Host labelagent-pai
    HostName <PAI生成的主机名>
    Port <PAI生成的外部SSH端口>
    User root
    IdentityFile <本机labelagent_pai_ed25519私钥路径>
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

若云实例实际登录用户不是 root，替换 User。不要把示意尖括号原样复制执行。

Windows PowerShell 先检查本地转发端口：

```powershell
Get-NetTCPConnection -LocalPort 18878 -ErrorAction SilentlyContinue
```

若已被其他程序占用，改用另一个本地端口，例如 18888，不停止未知程序。端口可用时执行：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18878:127.0.0.1:18878 labelagent-pai
```

保持该终端运行。只做转发时没有普通服务器提示符是正常现象。关闭隧道后本机将无法调用云端模型。本机原 VLM 使用 18868，此处用 18878，原服务不需要提前停止。

## 9. Windows 再发真实图片请求

在本地解压本包的目录，使用 LabelAgent 已有 Python：

```powershell
& '<LabelAgent安装目录>\.venv\Scripts\python.exe' .\probe_vlm_api.py --base 'http://127.0.0.1:18878/v1' --image '<本机第76帧PNG绝对路径>' --output '.\probe-windows-one.json'
```

这一步将这张测试图片通过隧道发给自己的云端模型。随后用目标原图、候选图、参考原图、参考轮廓做多图检查。不要把 HTTP 200 或能聊天当作完整 Agent 通过。

现有 tools/connect_local_vlm.py 要求自写服务的 health JSON 与 local_vlm 元数据，会拒绝原生 vLLM；这里应使用本包的探测脚本。旧 local_vlm_server.py 的 4090 硬件检查以及单 GPU device_map 也不能直接当作 L20N 双卡启动器。

## 10. 接入 LabelAgent

先备份当前工作台状态及旧 2B 输出，并使用独立 PAI 对比会话。确认工作台没有运行任务，然后打开设置：

| 工作台设置 | 值 |
| --- | --- |
| 大模型 API 基础地址 | http://127.0.0.1:18878/v1 |
| 模型名 | labelagent-vlm32b |
| API Key | 云端 VLLM_API_KEY 的同一值 |

保存并关闭。若本地转发使用 18888，这里也改成 18888。基础地址只到 /v1，不要再追加 /chat/completions。

保持同一相机和经用户确认的第 0 帧参考，主动点击“生成真值描述”。当前项目的参考签名不含模型名/地址，因此换模型后旧描述不一定自动失效，必须主动重生成，避免沿用 2B 描述。

然后：第 76 帧 → 处理帧数 1 → 开始自动标注 → 查看分割、复查输入、决策和最终结果。不要直接批量跑 608 张图，也不要自动点击“保存结果”把新候选变成真值。

LabelAgent 自己使用严格 JSON 的 operations 字段执行工具，并不是直接使用 OpenAI tools/function calling；先让普通 chat/completions + JSON Schema 路径工作，不额外启用工具调用解析器。

## 11. 对照怎么做

- 本机 2B 基线已完整跑过目标 76：9 托盘、2 人员，1 轮 done、实际编辑 0，候选未确认。
- 第一组固定图片字节、参考轮廓、提示与 schema、输出预算和图像预算上限，对比 2B 与 32B。量化精度、框架和处理器也不同，结果需分别披露，不能称为只改变参数量。
- 第二组在同一 32B 和同一输入下，将 LABELAGENT_MAX_PIXELS 从 262144 改为 1048576，受控重启自己的服务后比较。该参数对应上限，不自动证明每个视图实际用了 1024 个视觉编码；须记录真实处理器网格再做严格密度结论。
- 使用空白、不确定对象、正常候选及已知错类对照，检查误报、误删、实际恢复和下一轮复查。不能只统计 done 比例。
- 真正的像素质量提高仍需目标帧独立二维人工标注，三维投影支持不能代替二维准确率。

## 12. 常见现象与处理

| 现象 | 优先检查 |
| --- | --- |
| 等待调度 / 资源缓存 | 资源配额、空闲卡、节点可用量；不要反复创建重复实例 |
| 模型路径不存在 | 下载目录、挂载点、分片清单，不用空目录启动 |
| 磁盘满 | 50GiB 系统盘不足；检查实际模型目录的 df，而不是只看申请页 |
| CUDA / Driver 错误 | nvidia-smi、torch.version.cuda、vLLM 构建匹配；不要升级共享宿主机驱动 |
| 只用一张卡 | 实际 GPU 数及 tensor-parallel-size=2，不是 data-parallel-size=2 |
| NCCL / 共享内存报错 | /dev/shm、同实例双卡拓扑、通信日志；不要直接认定网卡或显卡坏了 |
| 显存不足 | 是否两张完整卡、其他占用、图像预算、上下文和并发；保持 max-num-seqs=1 |
| Connection refused | 服务是否启动、SSH 转发是否仍在、端口两端是否正确 |
| HTTP 401 | 两端 API Key 是否一致，不是阿里云 AccessKey |
| Model not found | served-model-name 与工作台模型名是否完全一致 |
| 不支持 response_format | vLLM 版本与结构化输出能力；不要删除项目输出校验来掩盖问题 |
| 只有 health 通过 | 还需真实图片、严格 JSON、多图及完整 Agent 验证 |
| 输出 done 却不改错 | 质量问题，保留失败样例，不把结束等同正确 |

需要回退本机时，恢复工作台基础地址 http://127.0.0.1:18868/v1、模型名 labelagent-vlm，API Key 恢复原本机配置；主动重生成本机描述。原 SAM3 入口 8766 保留。不要覆盖原状态文件来回退模型地址。

开发机停止、删除或重建前保存代码、冻结依赖、权重版本、日志和结果。关闭浏览器不会停止算力计费；磁盘、NAT/EIP 等可能独立计费。共享配额也要释放不用的计算资源。持久性取决于系统盘/挂载类型，不能仅凭目录名 /mnt/workspace 判断。

## 官方参考

- 创建与参数：https://help.aliyun.com/zh/pai/create-and-manage-dsw-instances
- L20N 48GB：https://help.aliyun.com/zh/cs/user-guide/gpu/
- SSH 网络：https://help.aliyun.com/zh/pai/dsw-network-configuration/
- SSH 直连：https://help.aliyun.com/zh/pai/dsw-direct-connection
- ProxyClient：https://help.aliyun.com/zh/pai/connect-to-a-dsw-instance-over-ssh
- Qwen 官方模型：https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
- vLLM GPU 安装：https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- vLLM 参数：https://docs.vllm.ai/en/stable/cli/serve/
- vLLM 结构化输出：https://docs.vllm.ai/en/stable/features/structured_outputs/
- vLLM 安全边界：https://docs.vllm.ai/en/stable/usage/security/
- ModelScope 下载：https://modelscope.cn/docs/models/download

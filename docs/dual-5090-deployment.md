# 双 RTX 5090 部署说明

这套部署将 GPU0 专用于 SAM3，GPU1 专用于 LabelAgent 的 Qwen3-VL 服务。两项都是真实模型服务，不启动空占显存的 hold 进程。服务仅监听服务器回环地址，通过 SSH 隧道供本机工作台访问。

## 目录与环境

- 项目：`/home/clw/projects/labelagent-5090/app`
- SAM 环境：`/home/clw/projects/labelagent-5090/.venv-sam`
- VLM 环境：`/home/clw/projects/labelagent-5090/.venv-vlm`
- 模型：`/home/clw/models/labelagent/`
- 日志：`/home/clw/logs/labelagent-5090/`
- PID：`/home/clw/run/labelagent-5090/`

安装使用 Python 3.12、PyTorch 2.10.0/torchvision 0.25.0 的 CUDA 12.8 轮子。SAM 与 VLM 分环境，避免依赖互相覆盖。模型权重、真实配置、日志和数据不进入 Git。

## 配置与启动

发布或复制到 Linux 前必须先运行原始字节换行门禁：

```bash
python -m unittest \
  tests.test_dual_5090_deployment.Dual5090DeploymentTests.test_all_shell_scripts_use_lf_line_endings
```

该门禁递归检查 `tools/**/*.sh`，任何 `CR`/CRLF 字节都会使发布失败。`.gitattributes` 的 `*.sh text eol=lf` 不能替代这个读回检查：旧 Windows 工作树或绕过 Git 属性的复制仍可能把 CRLF 带入部署包。命中时只允许先保存逐文件 SHA-256 和原件，再做 CRLF 到 LF 的纯换行转换；不得借修复换行修改脚本逻辑。

复制 `tools/dual_5090.env.example` 为服务器上的 `~/projects/labelagent-5090/dual-5090.env`，填写两张卡的 UUID。SAM 权重应位于 `source/sam3/weights/*.pt`，VLM 目录必须包含 `config.json`。

```bash
LABELAGENT_5090_CONFIG=~/projects/labelagent-5090/dual-5090.env \
  bash tools/dual_5090_services.sh start all

bash tools/dual_5090_services.sh status all
```

启动前脚本读取 `nvidia-smi`，只接受明确 UUID 的 RTX 5090；卡上已有计算进程、利用率超过 10%，或真实空闲显存不足 28000 MiB 时拒绝启动。脚本不会结束其他用户进程。停止命令只接受本部署记录且命令行可核验的 PID，SIGTERM 后未退出会保留进程供人工检查，不会强制杀死。

本机访问使用 SSH 隧道，例如把 5090 的 `18766`、`18878` 映射到未占用的本地端口。健康检查分别为 `/health` 和 `/health`；VLM 还提供 OpenAI 兼容的 `/v1/chat/completions`。

若需保留原工作台作为基线，可为第二个工作台设置独立目录和端口：

```bash
LABELAGENT_WORK=/path/to/isolated-work \
SAM3_URL=http://127.0.0.1:<sam-tunnel-port> \
AUTOLABEL_PORT=<free-local-port> \
python -m source.agent.server
```

然后通过 `/api/config` 将该工作台的 `llm_url` 指向 VLM 隧道。独立工作目录用于隔离真值、候选、运行记录和模型配置；不得让两个工作台共写同一个 `work` 目录。

## 验收边界

健康检查只证明服务进程和模型装载状态。正式验收还必须各跑一次真实请求，并记录模型、GPU UUID、耗时、显存和输出摘要。SAM 微调必须等人工确认的 mask 真值准备好后再启动；没有真值时只能做基线推理和候选生成，不能称为微调完成或效果提升。

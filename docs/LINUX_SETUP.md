# LabelAgent Linux 兼容版安装与使用教程

## 1. 推荐架构与两种运行方式

推荐把网页后端、SAM3 和 VLM 全部放在同一台 Linux GPU 服务器。本机只运行一条 SSH 隧道并打开浏览器，不再运行 Python 后端：

```text
本机浏览器 -> SSH安全隧道 -> Linux网页后端 -> 同机SAM3 / 同机VLM
```

Linux 兼容版支持：

1. **网页层 + 远端模型服务**：Linux 机器只运行工作台，通过 HTTP 使用已有 SAM3/VLM；适合先验收页面和数据流程。
2. **单机 NVIDIA GPU 完整版**：同一台机器运行 SAM3 与工作台；需要 Python 3.12、NVIDIA 驱动、CUDA GPU，以及单独获得授权的 `sam3.pt`。

仓库不包含模型权重、MCAP、工厂图片、人工真值或 API Key。不要把 `.env`、权重或数据提交到 Git。

## 2. 下载

Linux 发布入口统一为 `linux` 分支。

CNB：

```bash
git clone --branch linux --single-branch https://cnb.cool/seer-robotics/EIC/BEV/LabelAgent
cd LabelAgent
```

GitHub：

```bash
git clone --branch linux --single-branch https://github.com/linweichen-ai/LabelAgent.git
cd LabelAgent
```

## 3. 系统准备

要求：x86_64 Linux、Python 3.12、`curl`、`flock`。完整 GPU 模式还需要 NVIDIA 驱动、可用 CUDA GPU和至少8 GiB可用空间。

Ubuntu 可由管理员安装基础包：

```bash
sudo apt update
sudo apt install -y curl util-linux python3.12 python3.12-venv
```

如果系统没有 Python 3.12，可使用单位批准的 Python、uv 或 conda 环境，通过 `--python` 指向 Python 3.12。安装脚本会拒绝其他主次版本。

## 4. 安装

只运行网页并使用远端模型：

```bash
chmod +x tools/*_linux.sh
tools/setup_linux.sh --web-only
```

本机 NVIDIA GPU 完整安装：

```bash
chmod +x tools/*_linux.sh
tools/setup_linux.sh
```

然后将获得授权的权重放到 `source/sam3/weights/sam3.pt`。安装脚本只写仓库内 `.venv` 和 `.repro`，不会修改系统 Python。

## 5. 启动

服务器已经运行 SAM3 与 VLM 时，使用独立工作台端口启动完整网页后端：

```bash
tools/start_linux.sh \
  --sam-url http://127.0.0.1:18766 \
  --vlm-url http://127.0.0.1:18878/v1 \
  --vlm-model labelagent-vlm8b \
  --host 127.0.0.1 \
  --port 18765 \
  --work-dir /data/labelagent-work \
  --no-browser
```

工作台默认端口仍为8765；共享服务器必须先检查端口归属，使用空闲端口，不得停止或覆盖其他用户的服务。启动记录会保存实际端口，`status_linux.sh` 会自动读取，不再假设固定8765。

使用远端 SAM3：

```bash
tools/start_linux.sh --sam-url http://SAM服务器地址:8766 --no-browser
```

本机运行 SAM3 和网页：

```bash
tools/start_linux.sh --no-browser
```

检查状态：

```bash
tools/status_linux.sh
```

默认启动时打开 `http://127.0.0.1:8765/`；若使用上面的服务器模式，则服务器内部地址为 `http://127.0.0.1:18765/`。

**不要直接打开 `source/web/index.html`。** `file://` 页面没有后端来源，看起来像网页但无法连接 API。

## 6. 从另一台电脑访问

默认只监听回环地址。Windows 本机可直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File tools/connect_remote_workbench.ps1 `
  -Target factory-5090 -RemotePort 18765 -LocalPort 18765
```

脚本会先检查服务器页面，再建立隐藏的受管 SSH 隧道并打开 `http://127.0.0.1:18765`。停止隧道：

```powershell
powershell -ExecutionPolicy Bypass -File tools/disconnect_remote_workbench.ps1
```

也可手工建立隧道：

```bash
ssh -L 18765:127.0.0.1:18765 用户名@Linux服务器
```

然后在自己的电脑打开 `http://127.0.0.1:18765/`。

只有在受控内网并已配置防火墙、反向代理和访问控制时，才显式监听其他地址：

```bash
tools/start_linux.sh --host 0.0.0.0 --no-browser
```

Flask 入口本身不提供公网身份认证，不应直接暴露到互联网。

## 7. 工作目录和数据边界

默认状态写入仓库的 `work/`，可改到大容量磁盘：

```bash
tools/start_linux.sh --work-dir /data/labelagent-work --no-browser
```

MCAP、图片、模型权重和 API Key 必须放在仓库外或被 `.gitignore` 排除的位置。机器候选继续保持 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`。`test24` 只用于冻结评测。

## 8. 停止和日志

```bash
tools/stop_linux.sh
```

停止脚本只处理 `.repro/linux-processes.json` 中属于当前克隆、且 `/proc` 命令身份匹配的进程，不使用 `pkill` 或 `killall`。

日志位于：

```text
.repro/logs/sam3.out.log
.repro/logs/sam3.err.log
.repro/logs/workbench.out.log
.repro/logs/workbench.err.log
```

## 9. 常见问题

网页打不开时依次执行：

```bash
tools/status_linux.sh
curl -f http://127.0.0.1:8765/api/run/status
```

确认浏览器地址不是 `file:///.../source/web/index.html`。完整模式缺少 SAM3 权重会在启动前失败关闭；只有网页或无本地 GPU 时使用 `--web-only` 和外部 `--sam-url`。

## 10. 验收标准

1. `tools/status_linux.sh` 返回进程身份匹配；
2. `/api/run/status` 返回 HTTP 200；
3. 浏览器显示工作台而不是静态 `file://` 页面；
4. 授权测试数据的时间轴、图像和候选叠加正常；
5. 停止脚本只关闭当前克隆记录的进程；
6. 未写人工真值、未触碰 test24、未上传密钥和私有数据。

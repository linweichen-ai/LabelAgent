# LabelAgent 可复现安装与验收

本文面向第一次拿到仓库的人。目标是把“代码能下载”提升为“环境能自检、服务能启动、结果能追溯”。模型权重、原始 MCAP、人工真值和账号凭证不进入 Git；缺少它们时，自检会明确报告，而不会把空服务当作复现成功。

## 1. 获取同一份代码

CNB 与 GitHub 的对外复现入口都固定为 `reproduce`。`labelagent` 是持续开发分支，历史
`codex/*` 与 `backup/*` 分支不作为复现入口：

```powershell
git clone --branch reproduce --single-branch https://cnb.cool/seer-robotics/EIC/BEV/LabelAgent
cd LabelAgent
git rev-parse HEAD
```

如需从 GitHub 获取同一复现版本，可使用：

```powershell
git clone --branch reproduce --single-branch https://github.com/linweichen-ai/LabelAgent.git
```

若团队提供了指定提交号，请使用 `git checkout <commit>` 固定版本，不要只记录“最新版”。

## 2. Windows + NVIDIA GPU 安装

要求：Windows 10/11、Python 3.12、NVIDIA 驱动、可用 CUDA 显卡，以及 PowerShell。安装脚本只在仓库自己的 `.venv` 中写依赖，不修改系统 Python：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
```

脚本安装本项目已验证的 CUDA 12.8 PyTorch、根目录固定依赖和仓库内 SAM3 源码。已有正确 PyTorch 时可加 `-SkipTorch`，但之后仍须通过自检。安装包较大；默认临时目录为仓库内 `.repro/setup-temp`，空间不足时用 `-TempDir E:\LabelAgentTemp` 指向有足够空间的本地磁盘。

SAM3 权重不随仓库发布。取得合法授权后，将权重放到：

```text
source\sam3\weights\sam3.pt
```

不要把权重、MCAP、密钥或 `work/state.json` 提交到 Git。

## 3. 启动与停止

本机 SAM3 + 工作台：

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_local.ps1
```

页面地址为 `http://127.0.0.1:8765`。运行日志位于 `.repro\logs`，本地状态默认位于 `work`，两者都被 Git 忽略。指定独立状态目录时：

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_local.ps1 -WorkDir D:\LabelAgentData\work
```

若 SAM3 已运行在另一台机器或另一个端口：

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_local.ps1 -SamUrl http://127.0.0.1:28766
```

停止由本次启动脚本创建的进程：

```powershell
powershell -ExecutionPolicy Bypass -File tools\stop_local.ps1
```

停止脚本只处理 `.repro/processes.json` 中由当前仓库启动的 PID，不扫描或结束其他用户进程。

## 4. VLM / Agent 配置边界

不配置 VLM 时，人工多边形、SAM3 点提示和候选复核仍可使用；完整自动标注的 GPT 复查和 Agent 修改需要 OpenAI 兼容的视觉模型接口。在页面设置中填写：

- API 基址，例如 `http://127.0.0.1:18868/v1`；
- 服务声明的模型名；
- 仅在服务确实要求时填写密钥。

密钥只保存在本地 `work` 状态中，API 返回和调试记录会隐藏密钥原文。仓库中的 `tools/manage_remote_workbench.py` 是团队 `factory-4090/factory-5090` 环境的站点专用入口；外部复现者不应照搬其中的 SSH 别名和服务器路径。

## 5. 三层验收

### 环境检查

```powershell
.\.venv\Scripts\python.exe tools\repro_doctor.py --require-cuda --require-weight
```

### 代码回归

```powershell
powershell -ExecutionPolicy Bypass -File tools\verify_checkout.ps1 -RequireCuda -RequireWeight
```

该命令执行依赖一致性检查、Python 回归和浏览器逻辑测试。若机器没有 Node.js，可明确使用 `-SkipWebTests`，但报告中必须写明网页测试未执行。
脚本强制使用 UTF-8，并把测试临时文件隔离在 `.repro/test-temp`；可用空间不足 512 MiB 时会在测试前停止，而不是产生误导性的业务失败。

### 服务读回

启动后执行：

```powershell
.\.venv\Scripts\python.exe tools\repro_doctor.py --require-services `
  --sam-url http://127.0.0.1:8766 --web-url http://127.0.0.1:8765 --json
```

HTTP 200 只证明服务连通。完整验收还应在一张已授权图片上完成：加载图片、SAM3 点提示、生成候选、刷新后恢复；若配置了 VLM，再运行一帧自动标注并保存下载记录。机器候选不得自动标记为人工真值。

## 6. 最小复现记录

对外提交结果时至少保留：

1. `git rev-parse HEAD`；
2. `python --version`、`pip check` 和 CUDA/GPU 信息；
3. `repro_doctor.py --json` 输出；
4. 测试通过/跳过/失败数量；
5. 权重来源与摘要（不得上传无授权权重）；
6. 输入数据标识、相机 topic、帧号、标签表和复查轮数；
7. 机器候选、人工确认真值、训练批准三种状态分别记录。

只有“环境、服务、真实请求、结果读回”四层均通过，才能称为端到端复现；仅能打开页面或 `/health` 正常不能替代真实推理验收。

## 7. 受控 LoRA 候选交付与验收

模型权重不进入公开 Git。团队内部获得已授权的受控 ZIP 后，先核对提供方给出的整包
SHA-256，再运行独立验收：

```powershell
.\.venv\Scripts\python.exe tools\verify_qwen3vl_controlled_bundle.py `
  D:\AuthorizedArtifacts\labelagent-qwen3vl2b-round375-lora.zip `
  --expected-sha256 <提供方给出的64位SHA-256> --tensor-scan `
  --output .repro\qwen3vl2b-bundle-verification.json
```

只有输出 `status=passed`、整包摘要一致、逐成员哈希一致且张量扫描通过，才能称为“权重包
完整复现”。这仍不等于模型精度或生产部署通过；还需要相同基座、依赖、输入数据和独立人工
评测。构建端使用 `tools/build_qwen3vl_controlled_bundle.py`，它只接受已经通过发布完整性门禁的
候选，并生成确定性 ZIP 与回执。

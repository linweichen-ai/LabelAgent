# LabelAgent 可复现安装与验收

本文面向第一次拿到仓库的人。目标是把“代码能下载”提升为“环境能自检、服务能启动、结果能追溯”。模型权重、原始 MCAP、人工真值和账号凭证不进入 Git；缺少它们时，自检会明确报告，而不会把空服务当作复现成功。

## 1. 获取同一份代码

面向同事的交付入口固定为单提交的 `release` 分支。`labelagent` 是持续开发分支，
`reproduce` 保留完整研发快照，历史 `codex/*` 与 `backup/*` 分支不作为交付入口：

```powershell
git clone --branch release --single-branch https://github.com/linweichen-ai/LabelAgent.git
cd LabelAgent
git rev-parse HEAD
git rev-list --count HEAD
```

CNB 保持同提交的团队内部镜像，但当前需要组织权限和已配置的 CNB 凭证；未授权的新环境会返回“仓库不存在”。已获内部权限时可使用：

```powershell
git clone --branch release --single-branch https://cnb.cool/seer-robotics/EIC/BEV/LabelAgent
```

不要把 CNB 内部镜像当作匿名公开下载入口。GitHub 与 CNB 的 `release` 提交号必须一致，且 `git rev-list --count HEAD` 应为 `1`。

若团队提供了指定提交号，请使用 `git checkout <commit>` 固定版本，不要只记录“最新版”。

## 2. Windows + NVIDIA GPU 安装

要求：Windows 10/11、Python 3.12、NVIDIA 驱动、可用 CUDA 显卡，以及 PowerShell。安装脚本只在仓库自己的 `.venv` 中写依赖，不修改系统 Python：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
```

脚本安装本项目已验证的 CUDA 12.8 PyTorch、根目录固定依赖和仓库内 SAM3 源码。已有正确 PyTorch 时可加 `-SkipTorch`，但之后仍须通过自检。首次完整安装会提前提示：PyTorch约下载2.9 GB，最终虚拟环境实测约5.1 GiB；网络较慢时下载阶段可能数分钟没有新输出。新环境不足8 GiB可用空间会在下载前停止。默认临时目录为仓库内 `.repro/setup-temp`，空间不足时用 `-TempDir E:\LabelAgentTemp` 指向有足够空间的本地磁盘。

安装入口会先尝试标准 `py -3.12`，再检查 Python Launcher 已注册的其他3.12运行时（包括 Astral/uv 安装的解释器），并核对实际主次版本。仍无法定位时可显式传入 `-Python C:\path\to\python.exe`。

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

停止脚本只处理 `.repro/processes.json` 中由当前仓库启动的进程，并在停止前联合核对仓库根目录、PID、进程创建时间、Python可执行文件和服务角色。旧PID被Windows复用、记录属于另一份克隆或任一身份不匹配时会失败关闭，不扫描或结束其他用户进程。
若脚本明确报告记录已经陈旧，可运行 `powershell -ExecutionPolicy Bypass -File tools\stop_local.ps1 -ArchiveStaleRecord`。它只在没有任何匹配的受管进程、且所有存活PID均被确认属于其他可执行文件或创建时间时，才把记录移动到 `.repro/stale-process-records/`；不会停止这些无关进程，也不会直接删除旧记录。混合了匹配与陈旧身份、身份读取失败或记录属于另一克隆时仍拒绝处理。
启动前会强制检查工作台8765端口以及本地SAM3使用的8766端口；端口已被其他实例占用时会在创建任何新进程前停止并明确报告。预检之后如遇端口竞态、子进程提前退出、健康检查失败或服务读回失败，启动脚本会清理本轮已创建的进程和进程记录，不接管或停止其他实例。
进程记录先写入同目录临时文件，完成逐字节读回和结构校验后再由文件系统原子安装；更新已有记录时使用原子替换。每次启动携带唯一编号，失败回滚只移除仍属于本次启动的记录，其他启动或既有记录不会被误删。
同一克隆的启动、停止和陈旧记录归档共用按绝对路径派生的 Windows 命名互斥锁。并发生命周期动作会在读取记录或创建服务进程前快速失败；脚本崩溃或退出后锁由操作系统释放，不产生需要手工删除的锁文件。启动锁在服务读回完成后、打开浏览器前释放；停止锁在进程退出及记录处理完成后释放。

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

该命令执行依赖一致性检查、Python 回归、浏览器逻辑测试，以及无需私有数据的真实 Edge 浏览器自检。若机器没有 Node.js 或无法运行浏览器验收，可明确使用 `-SkipWebTests`，但报告中必须写明网页逻辑和真实浏览器自检均未执行。
脚本会先检查 Node.js 并运行约数秒的公开合成 Edge 自检，确认浏览器环境可用后再进入完整 Python 回归，避免等待约1500项测试后才暴露浏览器依赖问题。脚本强制使用 UTF-8，并把测试临时文件隔离在 `.repro/test-temp`；可用空间不足 512 MiB 时会在测试前停止，而不是产生误导性的业务失败。Edge 无法启动时会直接提示安装或修复 Microsoft Edge。

### 服务读回

启动后执行：

```powershell
.\.venv\Scripts\python.exe tools\repro_doctor.py --require-services `
  --sam-url http://127.0.0.1:8766 --web-url http://127.0.0.1:8765 --json
```

HTTP 200 只证明服务连通。完整验收还应在一张已授权图片上完成：加载图片、SAM3 点提示、生成候选、刷新后恢复；若配置了 VLM，再运行一帧自动标注并保存下载记录。机器候选不得自动标记为人工真值。

### 无私有数据的离线复核浏览器自检

这项自检不需要模型权重、MCAP、人工真值或账号凭证。它生成81项公开合成身份和1像素占位图，在真实 Edge 浏览器中验证：部分草稿下载、刷新后内存清空、草稿文件恢复、哈希篡改整包拒绝、81项完整前正式导出禁用，以及正式导出中的训练授权始终为否。

只验证这条公开流程时，不必先安装 CUDA/PyTorch、Node.js 或完整项目环境：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_public_browser_check.ps1
```

脚本在 `.repro/public-browser-venv` 建立独立环境，仅安装固定版本的 Pillow 与 Playwright，执行依赖一致性检查和真实 Edge 自检。它不会读取模型权重、MCAP、人工真值或本地工作状态，也不会把合成决定写成训练批准。

`tools/setup_windows.ps1` 已默认安装固定版本的浏览器验收依赖。只有在本轮更新前已经创建过 `.venv` 且不重新执行安装脚本时，才需要手动补装 `tools/requirements-browser-qa.txt`。单独运行命令为：

```powershell
.\.venv\Scripts\python.exe tools\verify_synthetic_signoff_browser.py `
  --output .repro\synthetic-signoff-browser-check
```

输出目录必须预先不存在。成功时 `receipt.json` 中应同时出现 `passed=true`、`synthetic_only=true`、`external_requests=0`、`page_errors=[]`，以及三项授权标志均为 `false`。如需查看合成页面，可加 `--screenshot`；截图只包含公开合成内容。该通过结果只证明离线复核工作流和权限门禁可复现，不证明真实标注质量、模型精度或生产训练通过。

### 可选的只读托盘候选复核页

Round 561 示例包可用于复现 3D/LiDAR/时序路由后的只读候选展示，但它是历史开发证据，**不随精简后的 `release` 分支提供**。其历史资源标识为 `artifacts/round561-pallet-visibility-candidate-bundle-20261122`；不要把旧 `artifacts` 目录复制回 `release` 工作树。

<!-- release-doc-ref: external artifacts/round561-pallet-visibility-candidate-bundle-20261122 -->

如确需复核，请由项目维护者从完整开发快照或项目归档中提供一个已授权、已校验的只读目录，并把该目录的绝对路径写入 `LABELAGENT_AUTHORIZED_ROUND561_DIR`。该示例依赖未随 `release` 发布的历史输入和清单，当前分支没有可独立验证的一键重建命令；未取得授权目录时应跳过本节，不能用占位数据冒充。取得目录后只在隔离端口启动，不要覆盖正式工作目录：

```powershell
$candidateReviewDir = $env:LABELAGENT_AUTHORIZED_ROUND561_DIR
if ([string]::IsNullOrWhiteSpace($candidateReviewDir)) {
  throw "请先由项目维护者提供已授权的 Round 561 只读目录"
}
$env:AUTOLABEL_PORT = "18767"
$env:LABELAGENT_WORK_DIR = (Join-Path (Get-Location) ".repro\pallet-review-work")
$env:LABELAGENT_PALLET_CANDIDATE_REVIEW_DIR = `
  (Resolve-Path -LiteralPath $candidateReviewDir)
New-Item -ItemType Directory -Force $env:LABELAGENT_WORK_DIR | Out-Null
.\.venv\Scripts\python.exe -m source.agent.server
```

打开 `http://127.0.0.1:18767/` 后，顶部应出现“托盘候选 5”；点击后会在新窗口打开 `/pallet-candidate-review/`。该页面没有保存、训练或部署入口。删除 `LABELAGENT_PALLET_CANDIDATE_REVIEW_DIR` 后重新启动同一份代码，入口必须隐藏，`/api/pallet-candidate-review` 必须返回 404。

后端会校验候选清单的只读安全标志、相对路径、PNG 签名和逐文件 SHA-256；任一资源缺失或被篡改都会失败关闭。此示例只证明候选路由与复核入口可复现，不代表候选已经成为人工真值、训练数据或正式部署结果。

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

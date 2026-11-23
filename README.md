# LabelAgent 自动标注工作台

面向工厂图像与 MCAP 数据的可复现自动标注工作台。系统以 SAM3 生成分割候选，使用视觉语言模型复查语义，再由 Agent 做受约束的局部修改；所有机器结果先作为候选，只有人工确认后才可成为真值或训练数据。

## 交付入口

团队同事只使用 `release` 分支。该分支是单提交、精简后的正式交付版本；完整研发历史仍保留在 `labelagent` 与 `reproduce` 分支，不影响追溯。

```powershell
git clone --branch release --single-branch https://cnb.cool/seer-robotics/EIC/BEV/LabelAgent
cd LabelAgent
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
# 将获授权的 sam3.pt 放到 source\sam3\weights\
powershell -ExecutionPolicy Bypass -File tools\start_local.ps1
```

打开 <http://127.0.0.1:8765>。

如果 CNB 尚未授权，可临时使用 GitHub：

```powershell
git clone --branch release --single-branch https://github.com/linweichen-ai/LabelAgent.git
```

## 一条主流程

```text
图片或 MCAP
    ↓
读取相机与帧
    ↓
人工参考帧 / 已批准参考轮廓
    ↓
SAM3 生成候选
    ↓
VLM 复查类别、遗漏和冲突
    ↓
Agent 做受约束的局部修改
    ↓
候选结果叠加到原图
    ↓
人工接受 / 修改 / 拒绝
    ↓
人工真值（可进一步进入训练审批）
```

安全边界：候选不是人工真值；人工真值不是训练批准；训练产物也不会自动替换正式模型。

## 目录结构

| 目录 | 用途 |
|---|---|
| `source/agent/` | 工作流、VLM/Agent 协作和服务端接口 |
| `source/web/` | 自动标注与人工复核页面 |
| `source/sam3_service/` | SAM3 本地/远程服务 |
| `source/sam3/` | SAM3 运行所需源码，不含权重和演示素材 |
| `tools/` | 安装、启动、停止、自检、验收和离线工具 |
| `tests/` | 回归测试 |
| `docs/` | 主教程、架构和必要证据 |

旧实验截图、逐轮调试记录、临时数据和模型权重不进入 `release` 分支。

## 验收

```powershell
.\.venv\Scripts\python.exe tools\repro_doctor.py --json
powershell -ExecutionPolicy Bypass -File tools\verify_checkout.ps1
```

首次真实验收只处理一帧，并依次确认：数据读取、SAM3 候选、VLM 复查、Agent 修改、结果叠加、人工确认和运行记录。

## 必须另行获得的资源

仓库不包含：

- `sam3.pt`；
- 私有图片、MCAP、OSS 数据和人工真值；
- VLM 地址、模型名和密钥；
- 内部 LoRA 或其他模型权重。

因此，“代码已下载”不等于“模型效果已完整复现”。

## 文档

- [同事完整复现教程](docs/LabelAgent_同事完整复现教程_20261123.md)
- [架构与数据边界](docs/ARCHITECTURE.md)
- [可复现安装与验收](docs/REPRODUCIBLE_SETUP.md)
- [图片、MCAP 与四路相机导入](docs/capture-comparison.md)
- [本地 GPU 说明](docs/local-gpu.md)
- [托盘组件证据复核](docs/PALLET_BOUNDARY_COMPONENT_WORKBENCH_ROUND752_20261123.md)

## 历史与发布关系

- `release`：同事安装、演示和复现；单提交、精简树。
- `reproduce`：完整可复现开发快照和详细过程证据。
- `labelagent`：持续开发分支。
- `main`：不作为当前交付入口。

需要查旧实验时到历史分支，不要把旧产物重新复制回 `release`。

# LabelAgent 当前交付与完整复现说明（2026-11-27）

本文是当前版本的总入口，回答四个问题：同事应该下载什么、系统实际包含什么、哪些结果已经验证、出现问题后怎样恢复。历史轮次文档仍保留用于审计，但不需要逐份阅读才能完成复现。

## 1. 交付入口

| 使用者 | 分支 | 内容 | 是否含模型权重 |
|---|---|---|---:|
| 普通同事、演示与验收 | `release` | 精简代码、安装脚本、测试和主教程 | 否 |
| Linux GPU 服务器部署 | `linux` | 服务器端网页、SAM3、VLM 的部署与连接工具 | 否 |
| 研发人员 | `labelagent` | 完整研发历史、实验工具、失败记录和证据 | 否 |
| 获授权的模型维护者 | CNB `model-assets` | SAM3、Qwen3-VL 基座、LoRA 候选、许可证和 SHA-256 清单 | 是，Git LFS |

`main`、`reproduce`、`codex/*` 和 `backup/*` 均不是新的复现入口。历史分支不删除，因为它们承担回滚和审计作用。

## 2. 当前系统结构

```text
Windows 浏览器
    │  SSH 回环通道（不直接暴露服务器端口）
    ▼
Linux LabelAgent 网页后端
    ├── SAM3：点、框、文本提示和候选掩膜
    ├── 本地 Qwen3-VL-8B：常规语义复查
    ├── 外部专家模型：仅困难帧、默认关闭、失败回退本地
    └── Agent：只执行受约束的局部修改
            ▼
      机器候选 → 人工复核 → 人工真值 → 独立训练审批
```

运行时必须保留三道边界：机器结果是候选，不是人工真值；人工真值不等于训练批准；新权重不经过独立验收不会替换正式模型。`test24` 只用于评测，不能进入训练。

## 3. 已纳入稳定交付的能力

- 图片与 MCAP 读取、逐帧浏览和结果叠加；
- SAM3 文本、正负点和框提示；
- 人工多边形、矩形、椭圆、自由线、节点编辑、补区域与擦区域；
- 本地 Qwen8B 语义复查、受控 Agent 修改、失败回退和运行记录；
- 标签注册表在全新克隆中的创建、增加、自定义类别和重启持久化；
- Linux 服务器统一托管网页、SAM3 与 Qwen8B，本机只使用浏览器；
- 候选、人工真值、训练批准和评测集之间的隔离门禁。

## 4. 已完成但尚未部署的研究结果

Qwen3-VL-2B 的 Round375 LoRA 后训练确实执行完成，候选适配器和哈希清单已进入受控模型资产分支。但该轮主要使用机器教师样本，只能证明训练和回读流程成立，不能证明其优于 Qwen8B，更不能自动部署。SAM3 的场景微调仍处于数据与验收门禁阶段；当前正式路径主要依靠提示、候选复用、ROI、几何约束和人工修正。

斜坡在训练与分析中保留 `slope` 标签，以便统计困难样本；下游通行语义允许与 `floor` 合并。已有斜坡正样本、困难负样本和盲审工具，但尚无证据支持“跨工厂已完全解决”。托盘边界凹凸、孔洞、粘连和小目标仍是后续重点。

## 5. 最短复现流程

### Windows / 通用验收

```powershell
git clone --branch release --single-branch https://github.com/linweichen-ai/LabelAgent.git
cd LabelAgent
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File tools\verify_checkout.ps1
```

模型权重按 `docs/MODEL_ASSETS_AND_RUNTIME.md` 单独取得，不能把代码克隆成功误报为模型效果复现成功。

### Linux 服务器

```bash
git clone --branch linux --single-branch https://github.com/linweichen-ai/LabelAgent.git
cd LabelAgent
chmod +x tools/*_linux.sh
tools/setup_linux.sh --web-only
tools/start_linux.sh --sam-url http://127.0.0.1:8766 --no-browser
```

完整 GPU 安装、服务拆分、SSH 隧道和停止方式见 `docs/LINUX_SETUP.md`。首次只处理一帧，依次检查读取、SAM3、VLM、Agent、叠加显示、候选保存和人工确认。

## 6. 标签类别复现故障

全新克隆首次启动时，后端会创建项目级标签注册表；前端读取同一注册表，而不是使用浏览器内旧缓存。增加类别后需确认保存成功并重新读取。若出现“类别与标签不一致”，先运行复现自检并检查项目标签文件，禁止通过修改历史标注或把未知类别强制映射成已有类别来绕过错误。

## 7. 验收口径

完整复现至少要有以下证据：

1. `tools/repro_doctor.py --json` 的必需项通过；
2. `tools/verify_checkout.ps1` 或 Linux 对应测试通过；
3. 网页、SAM3、VLM 和任务状态均可读；
4. 一帧真实数据从候选生成到叠加显示完整跑通；
5. 关闭并重启后，类别、候选身份和审计记录没有漂移；
6. 没有把候选写成人工真值，没有把 `test24` 写入训练集。

单张效果图、同批数据提升、服务返回 HTTP 200 或训练进程结束，都不能单独代表完整复现或效果达标。

## 8. 清理与恢复

可清理对象仅限：已由远端分支和全新浅克隆回读覆盖的临时克隆、重复构建目录、下载残片、浏览器缓存和可确定重建的实验中间文件。清理前必须记录绝对路径、大小与恢复来源；清理后记录实际释放空间。

不得清理：原始数据、人工真值、唯一模型权重、密钥、当前正式工作台、尚未提交的独有代码、无法从远端恢复的实验结果，以及任何其他用户的文件。代码仓库不能替代数据和模型备份。

## 9. 继续开发前的固定检查

- 从 `labelagent` 新建隔离分支，不直接在 `release` 上实验；
- 所有机器产物保持 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`；
- 记录成功与失败实验的配置、输入身份、结果、原因和回滚点；
- 同时检查质量、误检、漏检、延迟、稳定性和跨数据集泛化；
- 通过门禁后再将必要代码和文档整理进 `release`，模型资产仍单独管理。

相关论文、官方实现、模型文档和本项目采用边界见 `docs/REFERENCES.md`。

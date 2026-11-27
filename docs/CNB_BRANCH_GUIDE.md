# CNB / GitHub 分支说明

## 复现入口

请下载：**`release`**

```powershell
git clone --branch release --single-branch https://cnb.cool/seer-robotics/EIC/BEV/LabelAgent
```

GitHub 使用同名入口：

```powershell
git clone --branch release --single-branch https://github.com/linweichen-ai/LabelAgent.git
```

该分支用于提供当前经过测试、带复现说明的版本。首次运行请继续阅读仓库首页和
`docs/REPRODUCIBLE_SETUP.md`。模型权重、原始MCAP、人工真值、账号凭证和本地
运行状态不随Git分支发布，需要按文档单独准备。

## 其他分支

| 分支 | 用途 | 是否建议复现者下载 |
|---|---|---:|
| `release` | 当前 Windows / 通用复现入口 | 是 |
| `linux` | Linux 服务器部署入口 | Linux 部署时使用 |
| `labelagent` | 持续开发与验证 | 否 |
| `model-assets` | CNB 受控模型资产与哈希清单 | 仅获授权且需要权重时使用 |
| `reproduce` | 旧复现分支，保留历史 | 否 |
| `main` | 早期主线 | 否 |
| `codex/*` | 历史实验和阶段记录 | 否 |
| `backup/*` | 历史备份 | 否 |

历史分支暂不删除，以保留回滚与审计记录；普通复现者只需认准 `release`。

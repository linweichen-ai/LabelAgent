# CNB / GitHub 分支说明

## 复现入口

请下载：**`reproduce`**

```powershell
git clone --branch reproduce --single-branch https://cnb.cool/seer-robotics/EIC/BEV/LabelAgent
```

GitHub 使用同名入口：

```powershell
git clone --branch reproduce --single-branch https://github.com/linweichen-ai/LabelAgent.git
```

该分支用于提供当前经过测试、带复现说明的版本。首次运行请继续阅读仓库首页和
`docs/REPRODUCIBLE_SETUP.md`。模型权重、原始MCAP、人工真值、账号凭证和本地
运行状态不随Git分支发布，需要按文档单独准备。

## 其他分支

| 分支 | 用途 | 是否建议复现者下载 |
|---|---|---:|
| `reproduce` | 当前复现入口 | 是 |
| `labelagent` | 持续开发与验证 | 否 |
| `main` | 早期主线 | 否 |
| `codex/*` | 历史实验和阶段记录 | 否 |
| `backup/*` | 历史备份 | 否 |

历史分支暂不删除，以保留回滚与审计记录；复现者只需认准 `reproduce`。

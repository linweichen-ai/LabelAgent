# Round759：外部专家设置真实浏览器验收

## 验收范围

在隔离工作目录和端口 `18768` 启动当前 `labelagent` 代码，只检查新增的外部专家模型设置入口。没有加载图片、MCAP 或任何工厂数据，也没有配置 API Key、调用外部模型、训练、推理或部署正式工作台。

## 过程与结果

- 第一次启动前创建两个隔离目录时使用了当前 PowerShell 不接受的参数组合，目录未建成，服务进程也没有启动；改为分别创建两个明确目录后正常启动。正式端口和正式数据未受影响。
- 在真实 Edge 页面中打开“设置”，确认外部专家区域可见，包含 OpenAI 兼容、OpenAI Responses、Claude Messages 和 Gemini 原生接口四种选项。
- `允许显式调用=false`、`只记录困难帧分流建议=false`，状态显示“专家通道与困难帧分流均关闭”。
- 页面只有 `LABELAGENT_EXPERT_API_KEY` 环境变量名字段，没有专家明文 Key 输入框。
- 切换夜间主题后重新打开设置，布局、文字、输入框和状态提示均可见；自动化复核同时验证暗色状态。
- 隔离服务完成后已关闭，端口 `18768` 不再监听。

## 证据

- 结构化回执：`docs/evidence/workbench-stability-20261117/round759-expert-settings-acceptance.json`
- 暗色界面截图：`docs/evidence/workbench-stability-20261117/round759-expert-settings-dark.png`
- 回执 SHA-256：`118f020030bdcec920e3bcf9dc9684ba1b8223e5926caa1836b95773150d344d`
- 截图 SHA-256：`a3e475efa4c53196ac431ae4ad4d8fbecf352f9202fb65248555783fc5fc6396`

## 结论与边界

当前代码已具备可用的专家模型设置入口，且默认关闭、安全提示和明文密钥隔离在真实浏览器中成立。本轮只证明界面与本地配置状态工作正常，不证明外部模型已连通，也不证明分割效果提升。正式部署仍应等待合成图连通性测试和外部模型固定样本 A/B。

# Round521：干净克隆轻量浏览器复现

## 目标

验证外部用户不复用现有虚拟环境、不安装 CUDA/PyTorch、不持有模型权重或数据时，能否从公开仓库下载代码并运行真实浏览器复核门禁。

## 新增入口

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_public_browser_check.ps1
```

该脚本：

- 自动定位标准或 Python Launcher 已注册的3.12解释器，包括 Astral/uv 运行时；
- 在 `.repro/public-browser-venv` 建立隔离环境；
- 只安装固定版本的 Pillow 12.3.0 与 Playwright 1.65.2；
- 执行 `pip check` 和真实 Microsoft Edge 合成复核；
- 不读取模型权重、MCAP、人工真值或工作台状态，不需要 Node.js 或 GPU。

## 干净克隆实测

从 GitHub `reproduce` 全新克隆提交 `4ceade7dad5a3273ffaaeceaa3c8c751580563b7` 后直接运行上述命令：

- 首次运行成功；本机已有 pip 下载缓存时总耗时约11.25秒；
- 第二次重复运行成功，总耗时4.56秒；
- 真实浏览器自检分别约3.16秒和2.34秒；
- 外部请求0、页面错误0、临时目录残留0；
- 克隆副本的受跟踪工作区保持干净；
- 训练授权、自动批准、批准训练均为false，真实人工决定写入0。

主工作区完整标准验收仍通过：Python共1505项（1504通过、1项按设计跳过），网页逻辑172/172通过。

## CNB 发现

CNB 已登录的主工作区能够推送并读回同一提交，但未继承仓库级专用凭证的全新目录执行 `git clone` 会返回“仓库不存在”，有无 `.git` 后缀结果一致。因此当前 CNB 是内部镜像，不具备匿名公开克隆证据；复现指南已把 GitHub 改为外部首选，并明确 CNB 需要组织权限。

## 边界

本轮证明公开合成复核页可以从干净克隆启动，不证明SAM/VLM精度、真实数据端到端推理或后训练完成。首次运行耗时会受网络与 pip 缓存影响，11.25秒不是无缓存下载基线。

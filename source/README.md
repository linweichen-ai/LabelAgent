# 代码结构

- `agent/`：MCAP 工作台、视觉大模型调用、本地自动标注与受控 OpenCV 编辑 agent
- `sam3/`：固定版本的 SAM 3 上游子模块；`weights/` 由使用者在本地准备
- `sam3_service/`：本项目的 SAM 3 HTTP 推理服务
- `web/`：浏览器工作台

依赖安装见项目根目录 README。自动标注结果保存在 `work/state.json` 中的候选区；逐帧复核后可在工作台保存为真值。无需 CVAT 或 Docker。

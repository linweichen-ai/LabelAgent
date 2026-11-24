# LabelAgent Linux 兼容性验收（Round 777）

## 结论

Linux 兼容入口已经在 WSL2 Ubuntu 环境中完成一次全新副本验收。网页工作台能够安装、启动、读取状态并干净停止；默认只监听 `127.0.0.1`，远程访问建议通过 SSH 隧道完成。

本轮验证的是 Linux 工作台与外部 SAM3 服务的连接能力。验收使用只返回健康状态的本地模拟 SAM3 服务，因此**不代表真实 SAM3 权重、CUDA 推理或分割效果已在 Linux 上完成验证**。

## 验收环境

- 平台：WSL2 Ubuntu（Linux 内核）
- Python：3.12.16，由 `uv` 安装到用户目录，未修改系统 Python
- 验收提交：`43c01f953fc335e3189573c61006b9d9e332f609`
- 全新副本：`/home/seer/labelagent-linux-acceptance-v2-1795501700`
- 安装模式：`tools/setup_linux.sh --web-only`
- SAM3：本机 `127.0.0.1:18766` 健康检查模拟服务

## 实测结果

1. `setup_linux.sh --web-only` 安装成功，`repro_doctor` 输出 `READY`。
2. `start_linux.sh --sam-url http://127.0.0.1:18766 --no-browser` 启动成功。
3. 工作台首页 `http://127.0.0.1:8765/` 返回 HTTP 200，页面标题为“标注 Agent 平台”。
4. `/api/run/status` 返回 HTTP 200，状态为 `idle`。
5. `status_linux.sh` 确认托管进程存活且身份匹配。
6. `stop_linux.sh` 成功停止托管进程，随后 8765 端口不可访问。
7. 验收副本最终 `git status` 为空。

## 已修复问题

最初验收发现，启动脚本打开的生命周期文件锁被后台服务继承，导致停止脚本提示“另一个生命周期操作正在执行”。现在启动后台服务前会显式关闭该文件描述符，并用测试固定这一行为。

## 边界与后续

- 未运行真实 SAM3 GPU 推理。
- 未下载或复制模型权重、数据集、人工真值和 `test24`。
- 未调用外部 ChatGPT API。
- 未更改 Windows 正式工作台的运行进程。
- 在 Linux GPU 机器上正式使用前，仍需按 `docs/LINUX_SETUP.md` 配置 NVIDIA/CUDA、PyTorch 和 SAM3 权重，再做一帧真实分割验收。

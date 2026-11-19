# Round 304：工作台推理空间恢复

日期：2026-11-20

范围：本地工作台运行前存储门禁与实验产物冷迁移

结论边界：不修改生产候选、人工真值、训练集、模型权重或 OSS

## 问题

工作台、SAM3 和 VLM 均在线，但运行前检查返回 `storage_low`：D 盘仅约 79.91 MiB，而单帧运行
至少要求预留 264 MiB。因此点击运行会被保护性拦截，这不是模型离线或页面卡死。

## 可逆恢复

工作树 `artifacts/` 原先链接到 D 盘，共 6,282 个文件、1,278,720,015 字节。先完整复制到：

`C:\Users\seer\Documents\ChatGPT\Seer2clw\cold_storage\labelagent_stability_worktree_20261117-artifacts`

源和目标逐文件 SHA-256 比对差异为 0，再把原工作树路径重新链接到 C 盘冷存储；历史文档和脚本
仍通过原 `artifacts/` 路径读取。验证后释放 D 盘旧副本。另对 71 个已停止写入的远程桌面/PowerPoint
诊断日志记录路径、大小、时间和 SHA-256 后清空内容，逻辑大小 617,099,264 字节；正在增长的
联想托盘日志未处理。原位压缩旧 Mask2Former 恢复目录没有产生额外物理收益，已如实记录。

## 结果

| 项目 | 恢复前 | 恢复后 |
|---|---:|---:|
| D 盘可用空间 | 约 79.91 MiB | 约 1,154 MiB |
| 单帧 264 MiB 安全门槛 | 不通过 | 通过 |
| 工作台运行状态 | idle | idle |
| SAM3 文本服务 | 在线 | 在线 |
| SAM3 点提示服务 | 在线 | 在线 |
| VLM `labelagent-vlm8b` | 在线 | 在线 |

`/api/run/readiness` 已返回 `code=ready`、`ready=true`。当前仍低于建议的 2 GiB 长批次余量，因此
适合少量帧验收，不应立刻启动大批量推理、下载或训练。生产 `state.json` 哈希保持
`237844f73ff4495e575c6a3f966260b9385efbdf1edee7d0e5177518470ddfab`，与恢复前一致。

结构化证据：
`docs/evidence/workbench-stability-20261117/runtime-storage-recovery-round304.json`

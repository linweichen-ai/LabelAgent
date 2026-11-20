# Round350：恢复工作台磁盘余量并改进临时目录选择

日期：2026-11-20

## 发现

日志增长已经停止后，D 盘可用空间仍从约 53 MiB 降到约 38 MiB。只读盘点确认：

- `D:\AI\Temp\DiagOutputDir\RdClientAutoTrace` 有 117 个远程桌面诊断跟踪，逻辑大小约
  414.5 MiB，但已由 NTFS 压缩为约 29 MiB，并不是主要物理占用；
- `D:\Apps\LabelAgent\work\file_sessions.json` 约 35 MiB，原先未压缩；
- 工作目录还有 1847 个未压缩 JSON，逻辑大小共约 290 MiB；
- 这些 JSON 是会话、标注和过程元数据，不包含原始 MCAP、模型权重或图片数据。

## 无损处置

1. 对工作目录设置 NTFS 压缩继承；
2. 压缩 `file_sessions.json`、`state.json` 和 `files.json`，哈希前后一致；
3. 在网页服务停止、远端模型保持在线时，对 1847 个未压缩 JSON 逐文件计算 SHA-256；
4. 压缩后再次逐文件核对，哈希不一致数为 0；
5. 290.0 MiB JSON 实际存储降为约 67.7 MiB，释放约 223.9 MiB；
6. D 盘可用空间恢复至约 274 MiB。

没有删除、移动或重命名任何数据，也没有改写人工真值内容。

## 临时目录选择优化

原管理器只有在默认临时盘低于 256 MiB 时才切换到 C 盘。清理后 D 盘约 274 MiB，虽勉强高于
门槛，但仍不适合承受临时文件。现在的选择规则是：

- 默认盘有 2 GiB 以上余量时继续使用默认盘；
- 默认盘低于 2 GiB时，若本机应用数据盘同时超过 256 MiB且空间更多，则选空间更大的盘；
- 两边都不足 256 MiB 时仍保持原有失败关闭行为，不强行启动。

正式工作台因此继续使用：

`C:\Users\seer\AppData\Local\LabelAgent\runtime-temp\workbench-8765`

而不是因为 D 盘刚超过 256 MiB 就切回 D 盘。

## 验证

- 临时目录管理专项：26/26 通过；
- 正式工作台恢复后状态为 idle；
- 两路 SAM、正式 VLM、快速 VLM 继续在线；
- `file_sessions.json` 保持 NTFS 压缩属性；
- 正式 `state.json` SHA-256 仍为
  `9EF08118DD2BA0760CC93CF57BBEAAE4B3BED062D383DF4E6A454C389D4AB115`；
- 正式管理器回滚备份位于
  `D:\Apps\LabelAgent\.local\backups\runtime-temp-selection-round350-20261120`。

## 当前边界

D 盘约 274 MiB 仍远低于 2 GiB 运行建议值，因此单帧展示和小范围操作可以继续，但长批推理、
训练和大文件导入仍由空间门禁阻止。下一步应在不删除唯一数据的前提下继续清理已验证可恢复副本，
或把大批任务落到具有充足空间的卷。

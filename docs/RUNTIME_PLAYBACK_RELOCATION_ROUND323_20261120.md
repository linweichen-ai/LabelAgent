# Round 323：D盘紧急空间恢复与回放无损迁移

日期：2026-11-20

## 结论

D盘余量在并行项目写入期间从约322 MiB继续下降到1.89 MiB，LabelAgent已无法安全运行。没有删除
另一个项目的活动文件，也没有触碰数据集、模型、人工真值或OSS；只将LabelAgent自身31个可重建
旧回放文件无损迁移到C盘冷存储，并在原路径建立目录联接。

迁移后实际回放媒体支持HTTP Range读取，返回`206`和正确`video/mp4`；工作台150帧回放索引保持
ready。迁移释放D盘约409 MiB，随后并行项目又自行释放临时空间，最终5次稳定采样均为
`2142.77 MiB`，工作台存储预检恢复`ok`。不能把全部约2.1 GiB恢复都归因于本轮迁移。

## 迁移与核验

| 项目 | 结果 |
| --- | --- |
| 原路径 | `D:\Apps\LabelAgent\work\playback` |
| 冷存储 | `C:\Users\seer\Documents\ChatGPT\Seer2clw\cold_storage\labelagent-production-playback-20261120` |
| 文件数 | 31 |
| 总字节 | 428,897,429 |
| 逻辑大小 | 409.03 MiB |
| 哈希树 | `2f2239fd062ed04e4250f9b655fcd194d89b055af2522d5acea582cdb6762252` |
| 原路径兼容 | NTFS目录联接，目标为上述冷存储 |
| 媒体实读 | HTTP 206，1024字节Range，`video/mp4` |
| 回放索引 | 150帧，ready |

迁移前后文件数、总字节和逐文件SHA-256构成的哈希树完全一致。正式`state.json`哈希仍为
`237844f73ff4495e575c6a3f966260b9385efbdf1edee7d0e5177518470ddfab`。

## 收尾状态

- 工作台HTTP 200；
- 文本SAM与点提示SAM均HTTP 200；
- `labelagent-vlm8b`在线；
- 存储预检`ready=true`、`status=ok`；
- 没有启动推理、训练或长期GPU进程；
- 没有删除文件，迁移可通过解除联接并移回目录恢复。

结构化摘要：
`docs/evidence/workbench-stability-20261117/runtime-playback-relocation-round323.json`

# 本机量化 VLM Agent 功能验收

2026-11-03，北京时间，Qwen3-VL-2B NF4 与 SAM3 在同一张 RTX 5060 Laptop 8GB 上轮流运行。同一相机第 0 帧提供功能参考，第 76、151 帧完成规范生成后的候选分割和视觉复核；结果已导入独立未确认候选会话。

| 目标帧 | 托盘候选 | 人员候选 | 复核轮数 | 实际修改 | 帧耗时（含模型轮换） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 76 | 9 | 2 | 1 | 0 | 88.844 秒 |
| 151 | 14 | 4 | 1 | 0 | 84.016 秒 |

两帧均由模型返回 `done`，共 29 个候选。没有发生工具修改或多轮修正，不能将本轮称为已验证多轮改进。含参考规范生成的运行约 353.876 秒，不含初次 VLM 激活加载。VLM 模型分配显存峰值为 3,764,695,040 字节（约 3.51 GiB），不包含显示进程、驱动和其他应用。

实际模型进程顺序为 Qwen → SAM3 → Qwen → SAM3 → Qwen。管理器记录同一时间最多一个所属模型工作进程，并在切换前等待旧进程树退出。工作台已配置本机 VLM 地址及 compact 模式。

## 证据

- [实际 Agent 汇总](local_gpu_agent.json)：真实 VLM 规范生成、四次 SAM3 类别分割、两次目标复核及收敛状态。
- [运行与阶段记录](local_gpu_progress.json)：固定模型版本、NF4、失败尝试、预算和候选数量对比。
- [原始数据保护核验](local_gpu_preservation.json)：1,023 个原始文件、原 MCAP、6 个既有标注会话及类别列表的摘要一致；新会话没有真值。
- [浏览器核验](local_gpu_browser.json)：两帧候选数量、整数 ID、显示及刷新恢复通过，没有页面错误，未点击保存真值。
- 93 项回归检查通过；SAM3 环境和独立 VLM 环境均通过依赖一致性检查。

逐次模型输入、输出、操作审计、完整候选轮廓和预览保留在本机独立输出目录，不提交到仓库。实际计算使用工作台同一 `workflow.describe` / `workflow.run` 的隔离验收入口，随后导入工作台验证显示；未把功能参考自动保存为生产真值。

## 对比边界

参考轮廓来自已有 SAM3 未确认候选，明确标记为 `unconfirmed_functional_fixture`，不是人工二维真值。包内三维 `labels` 152 帧继续作为三维对比参考，不转为二维分割真值。本轮没有像素准确率、召回率或标注质量达标结论。

固定 `pallet/person` 的既有 SAM3 基线在两帧分别产生 11、21 个候选；本轮为 11、18 个。Agent 使用模型生成的 SAM3 短语，与固定提示不同；本轮 VLM 没有修改候选。因此数量差异不能归因于 VLM 修正，更不能解释为准确率提升。

2B 模型曾输出非法字段、虚构孔洞和过长文本；这些结果被拒绝并保留。当前依靠 compact 提示、每图 256 token、最多 768 输出 token 和 JSON Schema 约束完成流程。受约束的描述及理由使用英文，部分理由仍笼统或被长度约束截短；格式合格与 `done` 不证明视觉判断正确。候选仍需人工复核。

## 复现

依赖和管理器启动见 [本机运行说明](../local-gpu.md)。给定同一相机的参考轮廓和新的输出目录：

```sh
python tools/run_local_agent_acceptance.py --dataset work/datasets/example --reference-json work/reference-fixture.json --reference-kind unconfirmed_functional_fixture --frames 76,151 --prompt-profile compact --max-rounds 5 --output work/agent-acceptance-new
```

本机 JSON 状态使用 UTF-8 显式读写，中文路径可在 Windows 重载。原有状态在重启前备份，恢复后重新核对会话内容摘要。

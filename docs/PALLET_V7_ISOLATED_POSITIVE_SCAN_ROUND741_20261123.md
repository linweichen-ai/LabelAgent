# Round741：16 帧隔离候选生成与 v7 正例扫描

## 结论

从正式 152 帧序列中按时间均匀选择 16 个未绑定帧，通过纯 GET 冻结原图，再在临时隔离工作区
运行首批 SAM3+VLM 候选生成。共得到 227 个机器候选，其中 164 个为 pallet。v7 对 164 个
栈板全部 no-op，两次离线扫描结果一致；这仍不能证明质量提升。

诊断定位到统一根因：164/164 都未通过旧的“候选占整图至少 30%”门。实际候选面积占整图
中位数为 0.108%，95 分位为 0.623%，最大仅 2.051%；候选框面积最大为 4.787%。因此旧门
在多实例工业画面上几乎关闭了全部图像证据路径。下一步应设计 ROI 尺度的 v8 影子策略，
而不是直接降低正式阈值。

## 隔离与证据

- 新增纯 GET 快照工具，要求工作台 idle，绑定 path/topic/载入代次，前后二读失败关闭；
- 16 张原图及 SHA 仅保存在私有 `.repro`，不上传仓库；
- `run_new_dataset_smoke.py --defer-background --max-rounds 1` 只生成首批候选，不运行 Agent
  后台精修和类别审计；
- 单帧候选可见耗时 20.04–22.99 秒，平均 21.63 秒；
- 正式 `state.json`、`files.json`、`file_sessions.json`、`last_run.json` 前后 SHA 全部一致；
- 人工真值、训练批准、部署批准均为 false；正式 8765 在快照前后均读回 idle。

## 离线扫描

新增离线适配器，以 snapshot manifest SHA、原图 SHA、smoke summary SHA、每个候选 JSON SHA
形成两级绑定，再调用 v7；不访问任何服务，只对真实变化项写 before/after mask。

| 项目 | 数值 |
|---|---:|
| 帧 | 16 |
| 全部机器候选 | 227 |
| pallet 候选 | 164 |
| v7 真实变化 | 0 |
| 通过整图 30% 尺度门 | 0 |
| 双跑首版 manifest SHA | `1613abba94142546f0a238c1b91bf57cf6ae2af3cf6e26d339aafa9af14b4903` |
| ROI 诊断 manifest SHA | `eed8b247a765158b6c2eb39464d5a4dd7bfca6e90daec6aec8d6e1e0dddc0b26` |

## 下一步

1. v8 只作为影子策略，将尺度门改为候选 ROI/绝对像素联合门；
2. 保留 v7 的授权新增区域、add-only、零组件合并、256 像素预算和结构孔 100% 保留；
3. 先在本轮 164 个候选和 frame0 人工作用域离线评测；
4. 只有出现真实 changed row 且人工指标不退化，才讨论部署，当前 v6/v7 正式行为均不变。

## 验证

- 新增定向测试：8 项通过；
- UTF-8 固定环境全量测试：1757 passed、1 skipped、282 subtests passed；
- 独立只读终审：无 P0/P1，可提交。

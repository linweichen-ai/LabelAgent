# Round 567：托盘候选复核能力安全接入正式工作台

日期：2026-11-22

## 本轮结论

已经把此前在隔离环境完成验证的“托盘候选只读复核”能力接入正式工作台代码，但保持默认关闭：没有配置复核包时，入口不可见，接口返回 404，不会改变现有自动标注、人工真值或演示操作路径。

正式工作台重启约 9 秒，重启后页面、双机模型通道和既有数据状态均正常。此次接入没有启动训练、没有运行批量推理、没有写入人工真值，也没有修改 OSS 数据。

## 接入内容

只部署以下 5 个经过 Git blob 校验的文件：

1. `source/agent/server.py`
2. `source/agent/pallet_candidate_review.py`
3. `source/web/index.html`
4. `source/web/pallet-candidate-review-entry.js`
5. `source/web/pallet-candidate-review-entry.css`

功能边界：

- 仅提供静态、只读候选复核页面；
- 复核包目录必须通过显式环境变量配置；
- 未配置时工具栏入口隐藏、后端接口 404；
- 不含脚本、表单、写入按钮或外链；
- 页面资产继续执行路径、PNG、哈希和内容安全策略检查；
- 不会把候选自动升级为人工真值或训练数据。

## 可回滚部署

新增部署工具：`tools/deploy_pallet_candidate_review.ps1`。

部署工具在落盘前检查：

- 源文件必须与指定 Git 提交完全一致；
- 正式目录旧文件必须与已知基线 `5d16643` 一致，或已经是最新版；
- 新增文件在旧基线中必须不存在；
- 临时文件复制后必须通过 Git blob 回读；
- 已存在文件使用原子替换并生成备份；
- 任一步失败时按逆序恢复旧文件，并删除本轮新增文件。

本次正式备份目录：

`D:\Apps\LabelAgent\.local\deploy-backups\pallet-review-20261122-185448`

目录内的 `deploy-manifest.json` 记录了旧、新 blob 与备份路径。第二次执行部署工具返回 `already_current`，说明部署幂等。

## 验收证据

### 部署前

- `/api/run/status`：`idle`
- 正式状态文件 SHA-256：`9ef08118dd2ba0760cc93cf57bbeaae4b3bed062d383df4e6a454c389d4ab115`
- 4090 SAM、5090 SAM、4090 VLM、5090 VLM：全部 HTTP 200

### 部署后

- 正式页面：HTTP 200
- UI 版本：172
- `pallet-candidate-review-entry.js`：HTTP 200，`Cache-Control: no-store`
- `pallet-candidate-review-entry.css`：HTTP 200，`Cache-Control: no-store`
- 未配置复核包时 `/api/pallet-candidate-review`：HTTP 404
- 浏览器中复核入口 DOM 存在但不可见
- 浏览器控制台：0 条 error / warning
- `/api/run/status`：`idle`
- 四个模型端点：全部 HTTP 200
- 正式状态文件 SHA-256：仍为 `9ef08118dd2ba0760cc93cf57bbeaae4b3bed062d383df4e6a454c389d4ab115`

因此可以确认：本轮只增加了可选复核能力，没有改变现有候选、真值、数据选择或运行任务。

## 自动测试

相关 Python 测试：

```text
9 passed in 0.85s
```

部署工具还在隔离副本上完成两次实测：第一次完成 5 文件部署与备份，第二次返回 `already_current`。

## 页面响应基准

部署完成后，在正式工作台空闲状态下对 4 个只读接口各连续请求 100 次；没有调用 SAM、VLM 或 GPU 推理：

| 接口 | 中位响应 | P95 | 最大值 |
| --- | ---: | ---: | ---: |
| 工作台首页 | 15.280 ms | 17.734 ms | 49.399 ms |
| 推理状态 | 13.504 ms | 20.369 ms | 26.710 ms |
| 数据摘要 | 27.996 ms | 31.927 ms | 36.860 ms |
| 新入口脚本 | 13.810 ms | 26.108 ms | 27.200 ms |

本轮没有发现由可选复核入口引入的页面响应瓶颈。相对耗时最高的是现有数据摘要接口，但 P95 仍约 32 ms；当前没有必要为了几毫秒收益改动稳定路径。

## 后续启用条件

只有同时满足以下条件时，才会在正式工作台显示“托盘候选”入口：

1. 指定一个已冻结、已校验的候选复核包；
2. 复核包中的清单、页面和 PNG 全部通过哈希检查；
3. 明确保留“候选仅供人工复核、不写入真值”的边界；
4. 再做一次正式浏览器只读验收。

在此之前，正式工作台继续保持当前稳定、默认隐藏状态。

# Round 573–575：5090 SAM 旧协议兼容与密集托盘复验

日期：2026-11-22

## 结论

5090 SAM 此前对密集托盘多提示评测返回 HTTP 400，并非模型离线，也不是 1920×1536 原图损坏。根因是离线评测工具绕过了正式工作台已有的协议兼容层：

1. 5090 的 `/segment_text_multi` 仍接收提示词列表，4090 接收类别到提示词的映射；
2. 5090 没有 `/segment_point_groups_batch`，需要回退为多个 `/segment_points` 请求；
3. 单点回退时，传输层没有把 `/segment_points` 识别为专用点提示接口，错误回退到了默认端口。

修复后，同一冻结的 5 帧 × 4 相机、共 20 张 1920×1536 图在 5090 上完整跑通：

| 指标 | 修复前 | 只修文本协议 | 完整修复后 |
| --- | ---: | ---: | ---: |
| 完成图像 | 0/20 | 20/20 | 20/20 |
| 定位门通过 | 0/20 | 7/20 | 20/20 |
| 实例隔离门通过 | 0/20 | 7/20 | 20/20 |
| 点提示兜底调用 | 0 | 13 | 13 |
| 点提示兜底成功 | 0 | 0 | 13 |
| 工作台写入 | 0 | 0 | 0 |

完整修复后一次全量运行约 15 秒。20/20 是“3D 投影定位与实例隔离门”通过，不等于人工像素边界质量已经通过，也不能直接进入训练集。

## 修复内容

### 多提示文本协议

离线评测改为复用正式工作台的 `sam_text_multi` 兼容层：

- 先尝试新版映射协议；
- 只在服务明确返回旧协议错误时，回退为提示词列表；
- 缓存已确认的服务协议；
- 保留有界超时，不允许评测请求无限等待。

单帧最小验证回执：

- `legacy_protocol=true`；
- `image_encoded_once=true`；
- 4 个提示词只做 1 次图像特征编码；
- 返回 3 个有效候选结果。

### 点提示批处理协议

离线评测改为复用正式工作台的 `sam_point_groups` 兼容层：

- 新服务使用 `/segment_point_groups_batch`；
- 旧服务返回 404 时，自动逐组调用 `/segment_points`；
- 只有“路由不存在”才允许回退，忙碌或畸形响应不会被掩盖；
- 每个请求继续受相同超时上限约束。

### 点提示服务路由

传输层现在把以下三类请求统一路由到 `sam3_point_url`：

- `/segment_points`；
- `/segment_points_batch`；
- `/segment_point_groups_batch`。

这修复了旧服务批处理回退后误连默认端口的问题。

## 冻结输入与安全边界

输入仍使用 Round 568 冻结清单：

`./.repro/round568-dense-pallet-heldout-20261122/manifest.json`

约束保持不变：

- 5 个连续标签帧；
- 4 路相机，共 20 张原始分辨率图；
- `evaluation_only=true`；
- `training_return_allowed=false`；
- `human_truth_write_allowed=false`；
- 不读取或写入 OSS；
- 不修改正式工作台状态。

## 结果证据

最终报告：

`artifacts/round575-dense-pallet-box3d-5090-point-route-fixed-20261122/results.json`

读回校验：

- 报告 SHA-256：`d3e38bf44a4b55ded58974e00323ae79212c72f4ca6dc20cabebf755afd1e43c`；
- 联系表 SHA-256：`8fcf1b074d3e5599db1741b2a6d09c7736153fe0f74ac96009f41ab905218713`；
- 20/20 行确认使用 5090 旧文本协议兼容；
- 13/13 兜底行确认使用逐组点提示兼容；
- 兜底错误 0；
- `approved_for_training=false`；
- `production_deployed=false`。

失败与中间结果没有删除：

- 原始 HTTP 400：`artifacts/round568-dense-pallet-box3d-5090-failed-20261122/`；
- 只修文本协议：`artifacts/round573-dense-pallet-box3d-5090-compatible-20261122/`；
- 首次点提示兼容但路由错误：`artifacts/round574-dense-pallet-box3d-5090-full-compat-20261122/`；
- 完整修复：`artifacts/round575-dense-pallet-box3d-5090-point-route-fixed-20261122/`。

## 正式环境读回

正式环境只部署了已验证的单点路由这一处最小修复，没有用工作树整份覆盖正式 `workflow.py`：

- 备份：`D:/Apps/LabelAgent/.local/deploy-backups/sam-point-route-20261122-194700/workflow.py`；
- 备份 SHA-256：`fce6c39adbbba02cc307505c486f413c4469f4d0b70fb93deca169003a20aa3f`；
- 部署后正式 `workflow.py` SHA-256：`490df12e41b928c69660e22ac1bc49dce2a654467dd5e5d95c1b8a723284b562`；
- 重启耗时约 9.8 秒；
- 隔离传输读回确认 `/segment_points` 解析到 `sam3_point_url`，而不是文本 SAM 地址。

部署复验结束后：

- 正式首页 HTTP 200；
- 正式工作台 `http://127.0.0.1:8765` 状态为 `idle`；
- 5090 SAM `http://127.0.0.1:38766/health` 为 `status=ok`、`model_loaded=true`；
- 4090 SAM 与两路 VLM 健康检查也全部为 HTTP 200；
- 正式状态文件 SHA-256 仍为 `9ef08118dd2ba0760cc93cf57bbeaae4b3bed062d383df4e6a454c389d4ab115`；
- 没有启动新的常驻训练或 GPU 占位进程。

## 测试

协议、路由、评测与工作台相关测试：

```text
311 passed, 85 subtests passed in 6.28s
```

## 后续边界

1. 5090 现在“可调用且能完整跑通”，但仍需要人工像素级复核，不能把定位门结果当成边界精度；
2. 4090 与 5090 的模型候选形态不同，后续比较应使用相同人工真值计算边界 F1、粘连率与孔洞率；
3. 在人工门禁通过前，本轮掩码只保留为离线候选，不接入正式真值或训练回流。

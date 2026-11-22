# Round 563：托盘候选复核页隔离接入

## 结论

本轮把 Round 561/562 的托盘可见性候选复核页接入 LabelAgent 后端，但保持**默认关闭、只读、未部署**。只有显式设置 `LABELAGENT_PALLET_CANDIDATE_REVIEW_DIR` 时，隔离实例才会开放 `/pallet-candidate-review/`；正式 8765 实例未设置该开关，也未重启、未改状态。

真实浏览器验收通过：5 个候选实例中，4 个显示候选掩码，1 个因为当前帧被遮挡而抑制二维掩码；两张 640×512 预览完整加载，页面按钮数为 0。

## 本轮增加的保护

- 默认关闭：没有显式目录配置时接口返回不可用。
- 清单约束：只接受 `labelagent.pallet_visibility_candidate_bundle.v1`。
- 安全标记：要求 `production_deployed=false`，且人工真值、训练数据、工作台写入均为 0。
- 内容校验：预览和掩码必须在清单内、路径不能越界、PNG 签名正确、SHA-256 完全一致。
- 页面校验：拒绝脚本、表单、按钮、链接和外部网络地址。
- 浏览器约束：返回严格 CSP、`no-store` 与 `nosniff`，页面无法调用工作台接口或提交表单。
- 失败关闭：文件缺失、计数不一致、资源被篡改时不降级展示。

## 验收证据

- 浏览器验收：`artifacts/round563-pallet-candidate-review-isolated-http-20261122/acceptance.json`
- 页面截图：`artifacts/round563-pallet-candidate-review-isolated-http-20261122/review-page.png`
- 浏览器验收结果：`passed=true`
- 页面统计：候选 5、输出掩码 4、抑制掩码 1、待复核 5
- 图片读回：2/2 完成，均为 640×512
- 页面操作按钮：0
- 自动化回归：24 passed，11 subtests passed
- 正式工作台验收期间状态：`idle`

## 边界

本轮没有把页面部署到正式工作台导航，没有写入人工真值，没有生成训练批准，也没有自动接受任何候选。该页面只是把 3D 深度、LiDAR 与时序路由的结果以可读方式展示出来，供后续人工复核。

## 下一步

在继续保持默认关闭的前提下，可增加一个只在开关有效时才显示的导航入口，并在隔离实例验证从主工作台进入、返回以及刷新行为。正式实例是否启用仍应经过独立验收，不随代码提交自动开启。


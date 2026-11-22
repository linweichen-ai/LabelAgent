# Round 565：托盘候选跨视角覆盖审计

## 结论

本轮不再只检查两张问题图，而是把 3 个序列、4 路相机共 12 张图全部纳入同一份机器可读审计。结果为：

- 12/12 张完成 3D 框定位；
- 10 张正常隔离视角全部安全跳过；
- 2 张托盘粘连视角全部触发后续深度、LiDAR 与时序路由；
- 误触发 0，漏触发 0；
- 触发路由共产生 5 个待复核候选，没有向其他 10 张图扩散。

这说明当前冲突门在这组已曝光开发样本上能把计算集中到真正存在粘连的视角。但 12 张同批次样本仍不是独立泛化证据，不能据此宣称新数据集上已经达到生产精度。

## 联表约束

审计工具同时读取并绑定：

1. Round 551 的 12 视角定位结果；
2. Round 555 的全量冲突触发结果；
3. Round 560 的 5 个候选路由结果。

它校验三份报告的 SHA-256、复合身份（序列、相机、帧 ID）、计数与只读标志。只要出现重复身份、负对照被路由、触发图没有候选、候选可写真值/训练集或接触表被修改，构建就会失败。

## 浏览器验收

- 顶部统计：12、10、2、0/0；
- 表格行数：12；
- 安全跳过：10；
- 冲突触发：2；
- 页面按钮：0；
- 接触表尺寸：1280×3456，完整加载；
- 外部网络请求：0；
- 总结果：`passed=true`。

## 证据

- 覆盖报告：`artifacts/round565-pallet-candidate-coverage-audit-20261122/coverage.json`
- 可视化页面：`artifacts/round565-pallet-candidate-coverage-audit-20261122/index.html`
- 12 视角接触表：`artifacts/round565-pallet-candidate-coverage-audit-20261122/contact-sheet.jpg`
- 浏览器验收：`artifacts/round565-pallet-candidate-coverage-browser-20261122/acceptance.json`
- 浏览器截图：`artifacts/round565-pallet-candidate-coverage-browser-20261122/coverage-page.png`
- 回归：26 passed，11 subtests passed。

## 下一步

当前最有价值的下一步是引入未参与阈值选择的新序列或新批次作为独立验证集，继续检查误触发与漏触发。若没有新的人工像素真值，应继续把结果标为候选覆盖审计，而不是精度或训练通过。


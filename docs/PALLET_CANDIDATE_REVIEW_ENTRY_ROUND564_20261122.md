# Round 564：托盘候选复核入口与默认隐藏验收

## 结论

本轮在主工作台顶部加入了一个由后端安全状态驱动的“托盘候选”入口。入口只在候选目录已显式配置、清单与所有图片哈希通过、且后端声明只读与未部署时出现；默认配置下保持隐藏，不改变正式工作台的视觉和操作流程。

## 用户可见行为

- 开关有效：顶部显示“托盘候选 5”，点击后在新窗口打开只读复核页。
- 开关无效或未配置：入口不可见，徽标为 0，后端状态接口返回 404。
- 新窗口页面仍无按钮、无保存、无训练、无部署入口。
- 入口使用 `noopener`，不会让复核页控制原工作台窗口。

## 浏览器验收

验收同时启动两套隔离实例：

1. `18767`：显式绑定 Round 561 候选包；
2. `18768`：同一份代码，但不配置候选目录。

自动化读回结果：

- 候选页：5 个实例、4 个输出掩码、1 个抑制项；2/2 张 640×512 图片加载完成；按钮数 0。
- 启用实例：入口可见，链接为 `/pallet-candidate-review/`，徽标为 5，目标为新窗口，点击后打开未部署只读页。
- 默认实例：入口隐藏，没有 `data-ready`，徽标为 0，接口状态码为 404。
- 总结果：`passed=true`。

## 证据

- 验收结果：`artifacts/round564-pallet-candidate-review-entry-browser-v2-20261122/acceptance.json`
- 启用截图：`artifacts/round564-pallet-candidate-review-entry-browser-v2-20261122/workbench-entry.png`
- 默认隐藏截图：`artifacts/round564-pallet-candidate-review-entry-browser-v2-20261122/workbench-entry-hidden.png`
- 复核页截图：`artifacts/round564-pallet-candidate-review-entry-browser-v2-20261122/review-page.png`
- 相关回归：24 passed，11 subtests passed。

## 正式环境边界

正式 8765 实例没有设置候选目录，没有重启，也没有部署本轮入口。验收结束后两个隔离实例均已关闭；正式工作台读回为 `idle`，正式状态文件 SHA-256 仍为 `9ef08118dd2ba0760cc93cf57bbeaae4b3bed062d383df4e6a454c389d4ab115`。

## 下一步

下一轮应扩大候选包验证范围，优先加入新的序列和更多相机视角，确认入口计数、预览布局和遮挡抑制逻辑在更大样本上仍一致，再决定是否对正式环境启用。


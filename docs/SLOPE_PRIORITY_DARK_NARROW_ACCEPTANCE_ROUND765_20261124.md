# Round765：斜坡优先复核窄屏夜间验收

## 结果

在隔离端口 18770、620×900 窄屏和夜间主题下，真实 Edge 浏览器验收通过。专项复核弹窗完整可见，复核包选择、加载、刷新、六个筛选按钮和安全边界说明均未被裁切；frame 95、140 仍排在最前，并带有清晰的橙色“优先核对”视觉提示。

结构化读回显示：32 帧队列首四项为 `95、140、74、77`，“优先复核”筛选精确返回 `95、140`；页面错误、失败请求、非预期 HTTP 错误和真值写请求均为 0。正式工作台、人工真值、训练集、模型和 `test24` 均未修改。

## 证据

- 回执：`artifacts/round765-slope-priority-browser-dark-narrow/result.json`
  - SHA-256：`178acb25a9bfc28c7af3a24af43242fc63600705dd921e895af27822e42a8f0c`
- 截图：`artifacts/round765-slope-priority-browser-dark-narrow/review-packet-priority-queue.png`
  - SHA-256：`3edd6581f9a1d566191681ec0cf1864578d44735f7893fd8287b7fe5dd2f7514`
- 验收脚本现在会自动寻找当前 Codex Playwright 运行时，并支持受控的视口、夜间主题和只读优先队列模式。
- 验收结束后隔离端口已关闭。

## 边界与下一步

本轮只证明优先队列在窄屏夜间模式下可用，不代表斜坡识别准确率提高。优先帧仍需真实人工签核，且专项决定不会自动成为整帧人工真值或训练授权。当前代码尚未部署到正式 `D:\Apps\LabelAgent`。

# Round520：标准验收浏览器依赖快速失败

## 问题

Round519 已把公开合成真实浏览器门禁纳入标准验收，但执行顺序是先跑完整 Python 回归，最后才检查 Node.js 和 Microsoft Edge。干净 Windows 环境若缺浏览器依赖，用户需要等待约1500项测试结束后才看到失败，体验上会像“跑了很久才卡住”。

## 修改

- 标准脚本在完整回归前检查 Node.js 可执行文件并读回版本；
- 先运行公开合成真实 Edge 浏览器自检，再进入完整 Python 回归；
- Playwright 无法启动 Edge 时，给出“安装或修复 Microsoft Edge”这一可执行提示；
- 新增静态回归，保证 Node 和真实浏览器检查始终位于长 Python 套件之前；
- 保留 `-SkipWebTests`，但使用它仍表示网页逻辑和真实浏览器均未验收。

## 验收

标准命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\verify_checkout.ps1
```

结果：

- 环境检查与 `pip check` 通过；
- Node.js `v24.22.3` 预检通过；
- 真实 Edge 合成自检约2.5秒完成，外部请求0、页面错误0；
- Python共1501项，1500项通过、1项按设计跳过；
- 网页逻辑172/172通过，草稿攻击拒绝39项；
- 临时浏览器目录运行后残留0；
- 训练授权、自动批准、批准训练均为false，真实人工决定写入0。

## 边界

本轮改善的是失败速度和错误可读性，不改变SAM/VLM推理、标注真值或训练结果。通过公开合成浏览器门禁仍不等于真实数据端到端精度验收。

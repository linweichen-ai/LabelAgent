# Round390：ego 逐像素人工复核包（2026-11-22）

## 结论

已从四路相机各固定 5 个时点（0、38、76、114、151），生成 20 帧本地人工复核包。每帧包含
机器候选二值掩膜、原图叠加预览、来源图像哈希和结果状态；另生成四相机×五时点总览图。

20/20 帧获得候选几何，0 帧调用失败；其中 14 帧进入人工复核态，6 帧因 SAM 未覆盖全部正点
而标记 `blocked_sam_warning`。所有文件仍是机器候选，不是人工真值，不批准训练、不允许自动合并。

## 分相机结果

| 相机 | 观察候选 | 人工复核态 | SAM 警告阻断 | 失败 |
| --- | ---: | ---: | ---: | ---: |
| camera0 | 5 | 4 | 1（frame151） | 0 |
| camera1 | 5 | 0 | 5 | 0 |
| camera2 | 5 | 5 | 0 | 0 |
| camera3 | 5 | 5 | 0 | 0 |

camera1 的 5 帧均保留掩膜和叠加预览供诊断，但 `candidate_available=false`；不会因为“有图可看”而
绕过警告门禁。

## 本地复核包内容

- `candidate_masks/`：20 张严格二值机器候选掩膜；
- `overlays/`：20 张只用于人工查看的叠加图；
- `contact_sheet.jpg`：四相机×五时点总览；
- `review_queue.json`：逐帧哈希、分数、面积、警告状态和人工边界声明。

真实图像、掩膜和叠加图只保留在本地未跟踪目录
`artifacts/round390-ego-pixel-truth-review20-v2-20261122/`，不上传 GitHub/CNB。

## 助手目视预检（不是人工真值）

助手查看了 20 格总览图：候选总体位于画面下方可见的叉车本体区域；camera1 与人员手臂、身体接触
较多，五帧全部被 SAM 警告阻断符合保守策略；camera0 frame151 也因提示覆盖不完整被阻断。
总览中没有看到整片地面或整个人被完整吸入，但缩略总览无法证明像素边界正确，更不能替代人工
逐像素确认。

## 可复现命令

```powershell
.\.venv\Scripts\python.exe tools\build_ego_pixel_truth_review_packet.py `
  --image-root artifacts\round385-ego-camera-templates\local-review `
  --output artifacts\round390-ego-pixel-truth-review20-v2-20261122 `
  --point-sam-url http://127.0.0.1:28766
```

工具拒绝覆盖已有目录、只允许 loopback point-SAM、单请求 5 秒超时；不会写工作台状态，也不会生成
空白“真值”或人工确认收据。

## 验证

- 新增 3 项工具测试；
- 完整 Python：`1233 passed, 219 subtests passed`；
- 网页 Node：`167 passed`；
- 正式四份状态/历史 SHA-256 在 20 次 point-SAM 调用前后完全一致；
- review queue SHA-256：`82A57825F0F4A496052EE865A9CFCFE2D1C70371B2A31477B9BE9BE12A51F07B`；
- contact sheet SHA-256：`07F3E72A748F8691BD137A1264B55DFDEBE42A7B27F8B1ECD0D84565A1920709`。

## 下一步

人工在本地逐帧绘制 ego、person、floor 三类二值真值并签署哈希绑定收据；随后使用 Round389 的验收
工具计算 IoU、召回、精度和误纳率。未完成这一步前继续保持正式开关关闭。

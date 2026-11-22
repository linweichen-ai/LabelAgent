# Round509：疑难项放大复核建议

独立助手基于8项 RAW/MASK ONLY/OVERLAY 放大证据复核后：`likely_ok_after_zoom=5`、`uncertain_needs_human=3`、`obvious_issue=0`。没有发现明显类别错误、严重粘连或大块漏切；3个不确定项均来自黑色覆盖件重遮挡且触及画面边缘，而不是已经观察到的标注错误。

独立审阅原始回复中一项候选ID有笔误；验证器使用Round508私有manifest的精确ID集合失败关闭并以权威ID校正，错误ID没有进入正式回执。完整ID和理由只在 `.repro/cvat-task3-zoom-triage-round509.json`，SHA-256 为 `1c9c9ab262977328232bf04d899273c817918c5023432b6ffb4baee750644257`。

公开回执 `artifacts/round509-cvat-task3-zoom-triage-20261122/receipt.json` 的 SHA-256 为 `2a531e5876774043fbf6ff30de37b29e2fb2e9576f8f1c092d621b973e17619f`，不含候选ID或理由。新增验证器和2项测试确保 `likely_ok_after_zoom` 永不授权训练，错误/缺失ID会失败关闭。

当前仍是助手建议：人工真值否、人工签字0、自动批准否、训练授权否。下一步3个不确定项需要更高分辨率原图或相邻帧辅助人工复核；其余5项也必须在进入训练前获得正式人工签字。

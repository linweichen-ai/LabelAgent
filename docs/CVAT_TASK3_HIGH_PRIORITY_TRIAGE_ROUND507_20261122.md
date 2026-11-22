# Round507：高优先级栈板实例独立助手初筛

Round506 的12个高优先级代表项已完成独立只读视觉初筛：`likely_ok=4`、`needs_human_review=8`、`obvious_issue=0`。

4项在当前页面未见明显大块漏切或粘连；8项受画面边缘、货物、货叉或黑色覆盖件遮挡，单页不足以确认精细边界，需要单项放大三联图。该判断仅为助手初筛，不是人工真值；`human_signoff_count=0`、`training_authorized=false`、`automatic_approval=false`。

完整候选ID和理由只保存在 `.repro/cvat-task3-high-priority-assistant-triage-round507.json`，SHA-256 为 `3c865ccd5ca8d869fb2e1bb5378d4addf7752d380fccc3e59d20b3941b7f1102`。公开回执为 `artifacts/round507-cvat-task3-high-priority-assistant-triage-20261122/receipt.json`，SHA-256 为 `abbb27ca560b30d41e59daa47b24734d941755a1ba237f7b8d7b9072da5647bf`，不含候选ID或理由。

新增验证器 `tools/validate_cvat_assistant_triage.py` 和2项测试：助手 `likely_ok` 永不授权训练；未完整覆盖全部高优先级候选时失败关闭。

下一步为8个疑难项生成 raw / mask-only / overlay 单项放大图，再进入人工签字门。即使4项 `likely_ok`，训练前仍必须逐项人工签字并单独生成训练授权回执。

# Round511：81项人工签字模板

已为全部81个人工栈板实例生成正式签字模板。每行绑定 candidate ID、帧号、源图SHA、规范几何SHA和聚焦证据图SHA；允许决定仅为 `pass/reject/uncertain`，并要求明确 reviewer 和 reviewed_at。

模板初始状态全部为空：待复核81、人工签字0、训练授权否、自动批准否。助手的 `likely_ok`、放大建议和时序建议均不能自动填入人工决定字段。

私有模板 `.repro/cvat-task3-human-signoff-template-round511.json` 的 SHA-256 为 `28c66746d3b7c76117a3eb79b64e54cb3da7c116b5b94df6b37843e3480f066d`。公开回执 `artifacts/round511-cvat-task3-human-signoff-template-20261122/receipt.json` 的 SHA-256 为 `91a4a87401734e17fb168f1e8504423b80f940b0ce492b1f2e9dbc77c8f0d3a1`，不含候选ID。

新增工具 `tools/prepare_cvat_human_signoff_template.py` 和2项测试：模板必须81项全部待定且不授权训练；test24交集非零时失败关闭。下一步将模板接入只读证据页面和受控签字导入器，签字与训练授权继续分离。

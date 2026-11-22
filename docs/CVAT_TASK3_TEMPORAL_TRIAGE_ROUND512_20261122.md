# Round512：3项时序证据助手复核

Round510 的3个重遮挡栈板实例已完成独立时序复核，三项均被建议为 `likely_ok_with_temporal`，`still_uncertain` 和 `obvious_issue` 均为0。第三项只采用帧117/119/121/123的连续证据，明确排除发生场景切换的帧125。

该结果只代表助手建议，不是人工真值：人工签字0、自动批准否、训练授权否。完整候选ID和理由只保存在 `.repro/cvat-task3-temporal-triage-round512.json`，其 SHA-256 为 `a5fbfdd80999baeef2a8ae9c77598b3e59cb0cb628dd9b353027e841d4611e11`；公开回执 `artifacts/round512-cvat-task3-temporal-triage-20261122/receipt.json` 的 SHA-256 为 `130b0dd07d05c70951dd3767e6e6fd001fb2bb37d1e323739acb468bad43111b`，不含候选ID和理由。

新增校验器 `tools/validate_cvat_temporal_triage.py` 和2项测试。校验器严格绑定 Round510 manifest 哈希，要求助手决定完整覆盖且只能使用允许枚举；候选ID笔误、重复、缺失、未知决定或空理由均失败关闭。下一步仍由人工在 Round511 的81项签字模板中逐项给出 `pass/reject/uncertain`，助手建议不得自动写入。

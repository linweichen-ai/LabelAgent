# 本地交接同步器

`tools/maintain_live_handoff.py --config <private-config.json> --watch` 每60秒检查已提交代码、原有CNB进度分支、三段对话文本及冻结验证结果。配置文件保留本地，包含repo/control/handoff/evidence/video/guide路径、producer_branch/cnb_progress_branch/publish_branch、threads数组（id与path）以及interval_seconds。

先在隔离副本配置origin、cnb、producer三个remote及两个远端的现有凭据助手，切换到专用交接分支；不要在正在标注或由其他任务使用的工作目录启动。两个云端分支都采用正常推送并回读摘要，禁止强推。凭据使用现有官方登录，不写入配置或代码。

本地交接包含独立源码ZIP、经过摘要核对的视频、真实截图、脱敏可见对话文本及SHA256清单。公开分支只提交白名单汇总与代码，不上传本地媒体、原图或标注。

同步不运行模型、不改实验配方、不确认真值。监测按具体PID及创建时间核对后台进程，不把旧进度文件等同于活动进程。原评分全部完成后才接受完整588图结论，指标仍为同采集稀疏参考一致性。

control目录中的STOP文件停止任务；删除STOP后用相同命令重新启动。内核文件锁拒绝重复启动。网络或推送失败时保留本地结果和错误记录，下一轮重试。电脑休眠或进程退出期间不能持续更新。

验证：`python -m unittest discover -s tests -p test_live_handoff.py -v`。包含真实本地Git双端发布、重复运行不产生提交、源代码更新传播、远端并发提交拒绝覆盖、媒体篡改拒绝、隐私字段白名单与不完整评分拒绝。

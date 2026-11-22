# Round508：8个疑难栈板实例放大三联图

已为 Round507 的8个 `needs_human_review` 项生成单项 `RAW / MASK ONLY / OVERLAY` 三联图和4张双项汇总页。中间独立掩码能直接显示多部件、孔洞和遮挡切口，右侧叠加图固定使用2像素外轮廓和1像素孔洞线。

私有目录为 `.repro/cvat-task3-triage-zoom-round508/`；候选ID、图像和几何均未上传。公开回执为 `artifacts/round508-cvat-task3-triage-zoom-20261122/receipt.json`，SHA-256 为 `e37cba7b7da37bae42eec330b6f699f59a98abf54e66600b8cf209f267ee6b3b`。私有manifest SHA-256 为 `20606b67436e35807996ca38a1325cf87ba3ebbf15302e703fd287dc200ff29b`。

构建器严格绑定 Round502 候选清单、Round506复核包、Round507助手初筛及其公开回执；每项重新核对 manual+pallet 来源和源图SHA。任何哈希漂移、未知候选或非8项疑难集合都会失败关闭。新增2项测试覆盖正常生成8项但不授权训练，以及初筛哈希漂移拦截。

首张真实汇总页已现场打开：原图、独立掩码和叠加图均可读；木栈板上的大面积橙色主要来自浅色木材与绿色半透明层/橙色轮廓的视觉混合，不是整图误分。当前仍为 `human_review_pending`，人工签字0、自动批准否、训练授权否。

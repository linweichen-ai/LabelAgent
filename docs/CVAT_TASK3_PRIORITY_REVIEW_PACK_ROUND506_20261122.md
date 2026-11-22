# Round506：Task3 分级聚焦复核包

## 结果

已将 Round505 的46个视觉代表项按风险等级重新编排为13页私有复核图：高优先级12项、中优先级17项、常规17项。每项继续显示原始裁剪和2像素薄线轮廓叠加，并在顶部注明风险分数和触发信号。

本机目录：`.repro/cvat-task3-geometry-priority-review-round506/`，共13张复核页和1份私有 manifest，总计约3.14 MB。公开仓库未上传图片、候选ID或逐项结果。

## 失败关闭门禁

构建前必须同时满足：

- Round503复核manifest状态仍为待独立复核；
- Round505优先级审计与Round503绑定同一候选manifest；
- 优先级审计没有自动决策或训练授权；
- 每个代表项都存在对应聚焦图，且现场SHA-256与manifest一致；
- 路径必须是安全相对路径，禁止跳出私有复核根目录。

任一候选缺失、图片哈希漂移或门禁变化时整包失败，不输出部分“成功”结果。

## 公开回执

`artifacts/round506-cvat-task3-geometry-priority-review-20261122/receipt.json`

- 回执SHA-256：`d4bc76078239dc57faad644f804553efab3c940562b445506cc855ad966d4940`；
- 私有包manifest SHA-256：`7603e3d636810b98e16f48004b0c4763c32f2e2bdde341c739e396a1019c0454`；
- `automatic_decisions=false`；
- `human_truth=false`；
- `training_authorized=false`；
- `private_candidate_ids_published=false`；
- `private_review_images_published=false`。

## 实现与下一步

- 工具：`tools/build_cvat_geometry_priority_review_pack.py`
- 测试：`tests/test_build_cvat_geometry_priority_review_pack.py`

定向测试覆盖正常生成高优先级页面但不产生决策，以及聚焦图字节哈希漂移时失败关闭。首张真实高优先级页面已现场打开，分组横幅、原图和轮廓叠加均可读。

下一步先对3张高优先级页面中的12项做独立助手视觉初筛，再把 `likely_ok / needs_human_review / obvious_issue` 作为私有建议写入新的哈希绑定回执。该建议不能成为人工真值或训练授权。

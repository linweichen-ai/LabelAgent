# Round504：Task3 人工栈板实例视觉去冗余审计

## 结论

对 Round502 的81个人工栈板候选做只读视觉近重复筛查：

- 图像：81张；
- 两两比较：3,240对；
- 严格联合阈值命中：99对；
- 连通视觉簇：46个；
- 建议优先复核的代表项：46个；
- 自动删除：0；自动通过：0；训练授权：否。

簇大小分布：35个单例簇、5个二项簇、2个三项簇、1个五项簇、2个八项簇、1个九项簇。结果说明81帧中存在明显连续/近似视角冗余；先看46个代表项，再回看每个多项簇中的其余样本，可以减少重复劳动，但不能据此删除训练样本或把代表项当人工签字。

## 方法

每张图像在本地重新核对 SHA-256 后，计算：

- 64位 pHash；
- 64位 dHash；
- 32×32灰度缩略图平均绝对差。

只有同时满足 `pHash Hamming≤4`、`dHash Hamming≤4`、`MAD≤0.035` 才建立近重复边，再按连通分量形成视觉簇。该阈值比之前跨集合泄漏筛查更严格，目的是压缩复核重复度，不是证明语义完全相同。

## 私有/公开边界

包含候选ID、簇成员和最近图像对的完整审计只保留在：

`.repro/cvat-task3-manual-instance-diversity-round504.json`

- 大小：27,927字节；
- SHA-256：`d6e5a415025636829f46a7eb280e02f35743c2ba6ddf624e92bcf7d6db46c789`。

公开仓库仅保存汇总回执：

`artifacts/round504-cvat-task3-manual-instance-diversity-20261122/receipt.json`

- 回执 SHA-256：`88dc8f0ec402f29c4d012be1d7c03562ebb4738d9f380af56429b565a221f18c`；
- 不包含候选ID、簇成员、原图、裁剪图或本机路径；
- `private_cluster_ids_published=false`；
- `raw_images_published=false`。

## 实现与边界

- 工具：`tools/audit_cvat_manual_instance_diversity.py`
- 测试：`tests/test_audit_cvat_manual_instance_diversity.py`

定向测试覆盖近重复成簇但不删除样本，以及候选清单哈希漂移失败关闭。审计前必须验证 Round502 私有清单与公开回执哈希一致、候选来源为 manual、类别为 pallet、图片字节哈希现场一致且候选图像不重复。

本轮结果为 `assistant_triage_only`。近重复只能用于安排复核顺序，不能替代人工/独立签字，也不能自动决定训练集合。下一步先对46个代表项做质量分级，再对多项簇做一致性抽查；通过项仍需进入独立签字门和训练授权门。

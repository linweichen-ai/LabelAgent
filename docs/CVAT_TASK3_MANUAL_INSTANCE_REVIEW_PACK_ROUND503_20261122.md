# Round503：Task3 人工栈板实例聚焦复核包

## 结果

已将 Round502 的 81 个本机私有栈板实例候选渲染成 14 页聚焦复核图，每项同时展示原始裁剪与轮廓叠加裁剪。

- 候选数：81；
- 页面数：14（每页最多6项）；
- 轮廓线宽：2像素，避免粗线遮挡边界细节；
- 每张源图在渲染前重新核对 SHA-256；
- 每个候选再次要求 `cvat_metadata.source=manual`；
- 候选私有清单与 Round502 公开回执做精确哈希绑定；
- 81个候选 ID、帧、图像和规范几何哈希均唯一；
- 与冻结 test24 的24张图片做实际 SHA-256 交集，交集为0；
- 候选外区域仍为 `ignore`，禁止未标区域采负样本；
- 当前状态：`pending_independent_review`；
- 自动通过数：0；训练授权：否。

本机复核包位于：

`.repro/cvat-task3-manual-instance-review-round503-gated/`

其中包括 `review_page_01.jpg` 至 `review_page_14.jpg` 以及 `review_manifest.json`。这些页面包含原始图像裁剪和轮廓，因此全部留在 `.repro`，没有上传到公开仓库。

## 公开回执

公开仓库只保存不含图片、坐标和候选 ID 的统计回执：

`artifacts/round503-cvat-task3-manual-instance-review-20261122/receipt.json`

- 候选私有清单 SHA-256：`1033e8ad16fc0e4c8b03ef246557ae321273b4188b92bc3171c9604558f32f99`；
- 候选公开回执 SHA-256：`04d353e2b0620dad82b5e92c0d2e39a1f752e03511181973cabb7f9e4bad2f37`；
- 冻结 test24 清单 SHA-256：`8ed726bd93585329676dc46da533c02a154501211b1846b9ee3c4b1014bf9ee2`；
- 私有复核 manifest SHA-256：`519250bb839667155dcd55c8591f21f17a125e277a5553832f4a56683358e28a`；
- 14页聚合 SHA-256：`c123e091317598035f78b5362db838588413383478bc6572609b025567ea7d62`；
- 公开回执 SHA-256：`89985971f28d41171f69773a2b65f11ea5191720fb81e871ef240235c15fc195`；
- `private_review_images_published=false`；
- `private_geometry_published=false`；
- `raw_images_published=false`。

## 复核标准

每个实例必须分别确认：

1. `is_pallet`：对象确为栈板；
2. `contour_complete`：主要可见区域轮廓完整；
3. `no_merge`：没有把周围其他对象粘进来；
4. `no_major_omission`：没有明显漏掉大块栈板区域。

任何一项不通过即不进入训练。复核决定必须绑定候选私有清单哈希和复核页哈希；不允许仅凭页面名称或旧草稿复用。

## 实现与验证

- 构建器：`tools/build_cvat_manual_instance_review_pack.py`
- 定向测试：`tests/test_build_cvat_manual_instance_review_pack.py`

4项测试通过：正常生成薄线私有复核页；非人工来源失败关闭；允许负采样的危险策略失败关闭；候选清单哈希漂移失败关闭。首个真实复核页已实际打开检查，原始裁剪和叠加裁剪均可读，2像素轮廓没有覆盖关键边界。

独立只读审阅提出的关键门禁已纳入：清单/回执身份锁死、manual+pallet逐项复验、candidate ID与frame/shape ID绑定、训练几何规范哈希、每项独立聚焦图哈希，以及冻结test24真实交集检查。签字回执与训练授权仍保持两道独立门；复核页面或助手判断不能直接启动训练。

本轮未修改 CVAT、真值、候选轮廓、权重或生产工作台，未启动 GPU 训练。下一步是独立逐实例复核和相似帧去冗余，只有通过项才进入小规模偏标注微调。

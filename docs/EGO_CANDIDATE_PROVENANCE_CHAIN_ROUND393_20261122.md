# Round393：ego 候选—真值证据链绑定（2026-11-22）

## 结论

逐像素验收器原先能证明 ego/person/floor 真值经过人工哈希确认，却没有证明“被比较的候选”就是
Round392 机器复核包里的原候选。误把真值文件填到 `candidate_mask` 时，理论上可能得到虚假的
1.0 IoU。

本轮将清单和人工收据升级为 v2，把机器候选复核包、候选掩膜、三类人工真值和人工收据串成
一条完整证据链。旧 v1 清单和旧 v1 收据明确失败关闭，不会静默兼容未绑定候选的旧格式。

## v2 证据链

1. 清单 `candidate_provenance` 绑定原始 `review_queue.json` 路径和 SHA-256；
2. 复核队列逐样本绑定相机、候选路径、候选掩膜 SHA-256、可用状态和候选安全边界；
3. 清单中的候选文件必须与队列记录解析到同一个受限本地路径，内容哈希也必须一致；
4. 候选、ego、person、floor 四个掩膜必须使用四个不同文件路径；
5. 人工收据 v2 同时绑定整个候选队列 SHA-256、逐样本候选 SHA-256 和三类真值 SHA-256；
6. 候选队列、候选文件、真值文件或收据任一被替换，验收都会失败。

这解决的是证据来源和防误替换，不代表人工真值已经产生，也不会自动批准候选、训练或部署。

## 真实队列兼容验证

对 Round392 本地 `review_queue.json` 做只读验证：

- schema 与候选安全边界合法；
- 20 个样本全部被读取；
- 14 个 `candidate_available=true`，6 个为阻断候选；
- 队列 SHA-256 为
  `D0D4510D9622BD68943E28FA89E098354F660F2695F4C9C2ECDE0C2AF82E3C19`；
- 未创建人工真值，未运行正式像素指标门禁。

## 失败关闭测试

新增测试覆盖：

- 候选直接指向 ego 真值文件；
- 生成清单后替换候选掩膜；
- 篡改候选复核队列；
- 人工收据未绑定正确队列；
- 人工收据候选哈希被替换；
- 继续提交旧 v1 清单或旧 v1 收据。

以上路径均明确拒绝。输出报告仍不包含图像路径或像素几何，并固定
`approved_for_training=false`、`eligible_for_candidate_merge=false`。

## 验证

- 定向 Python：`9 passed, 2 subtests passed`；
- 完整 Python（`PYTHONUTF8=1`）：`1238 passed, 219 subtests passed`；
- 网页 Node：`167 passed`；
- 正式工作台 HTTP 200，SAM、主 VLM、快速 VLM 均 ready；
- 四份正式状态/历史 SHA-256 与 Round392 完全一致；
- 没有部署、没有写候选/真值/审核状态/训练批准。

## 使用更新

后续人工像素验收必须使用：

- `labelagent.ego_pixel_truth_manifest.v2`；
- `labelagent.ego_human_pixel_truth_receipt.v2`；
- 原始未修改的 `labelagent.ego_pixel_truth_review_packet.v1` 队列。

模板已同步更新：

- `docs/templates/ego_pixel_truth_manifest.example.json`；
- `docs/templates/ego_human_pixel_truth_receipt.example.json`。

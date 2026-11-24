# LabelAgent 人工复核实验交接

更新：2026-11-24。阅读本页即可确定本轮实验的范围、版本、结果和接手步骤。

## 同步范围与版本

本轮已推送至 GitHub 与 CNB 同名分支 `sync/signoff-handoff-20261124`，没有合并或覆盖任何主线。此前交接 HTML 中“尚未同步”的描述是推送前快照；以本页为准。

- GitHub：https://github.com/linweichen-ai/LabelAgent/tree/sync/signoff-handoff-20261124
- CNB：https://cnb.cool/seer-robotics/EIC/BEV/LabelAgent/-/tree/sync/signoff-handoff-20261124
- 实验基线：`107a12f`。
- 端到端测试：`a792ff458c7c1820db045db53cf4fc95eba39d18`。
- 时间戳一致性审计：`39b8579`。
- 首份交接 HTML：`9b6649fc6304844c20ffb54e7383cc54a3a6587c`。
- 同步前远端实查：GitHub `labelagent=da3b6f92c561c3f56e8da867ed16f2f11c559676`；CNB `labelagent=a1c0c660bd43f0414d92a6fce5567bcfeae5cab9`。两边主线历史不同，本轮不尝试统一。

本分支是历史实验的可复现快照，不能当成最新主线。提交祖先关系未包含某提交，也不能证明主线没有通过 cherry-pick 或重写吸收其等价内容。

## 核心 idea

离线人工复核页面允许导出的 JSON，应能通过正式签核导入校验器。需要同时满足：完整身份与哈希绑定、合法人工字段、无训练授权、失败时状态不变。人工 pass 仅表示人工判断，不是训练授权。

已经证明的是“旧版本存在前后端校验不一致”；尚未证明整个系统或模型质量提升。需要先复核最新主线是否已有修复，再开展修复实验。

## 代码入口与数据契约

- `tools/build_cvat_human_signoff_reviewer.py`：构建离线 HTML，内嵌 JS 和聚焦图；含草稿导入/导出与正式导出。
- `tools/validate_cvat_human_signoff_import.py`：正式导入门禁。
- `tests/test_reviewer_export_import_gate.py`：Node VM 执行生成页面的真实 JS，把导出 payload 交给 Python 导入器。
- `tools/audit_signoff_timestamp_parity.py`：17 个时间戳向量的 Python/JavaScript 一致性审计。
- `tests/test_cvat_human_signoff_drafts.js`：草稿攻击与状态恢复测试。

Round503 review manifest 含 81 项 `review_items`，Round511 template 含 81 行 `rows`。绑定字段为 `candidate_id`、`frame_id`、`source_image_sha256`、`canonical_geometry_sha256`、`focused_image_sha256`。草稿仅恢复 `decision/reviewer/reviewed_at/note`，还需携带 manifest/template SHA 和全量身份绑定。未知、重复、缺失、漂移、重复 JSON 键、授权字段注入应整包拒绝；失败不得部分改变当前进度。

正式输出要求 `status=human_review_complete`、`human_signoff_count=81`、训练与自动批准标志为 false。不得自动填人工决定；草稿可不完整，正式导出不可不完整。

## 本次复跑结果

- 端到端测试：2/2 通过。
- 81 项完整合成输入：导出后导入成功，pass/reject/uncertain 各 27 项，训练授权 false。
- 缺 decision：正式导出禁用。篡改授权、绑定、完成计数：导入拒绝。
- 时间戳审计：17 向量，3 不一致；退出码 1，属于预期发现缺陷，不是全绿验证。

| 时间戳 | 页面 | 导入器 |
|---|---|---|
| `2026-02-30T12:00:00+08:00` | 接受 | 拒绝 |
| `2026-11-22T24:00:00Z` | 接受 | 拒绝 |
| `2026-11-22T12:00:00.1234567Z` | 接受 | 拒绝 |

历史会话还用这三个时间戳各跑过完整导出→导入，均可导出但导入拒绝。本次复跑了上述审计和两项 E2E，未重新执行那段临时三例驱动。

导入拒绝不意味着人工内容已经被删除；原导出文件或草稿仍可用于纠正时间。问题是阻断工作流与增加返工，而非已证实不可恢复的数据丢失。

## 复现

在此分支仓库根目录执行，环境需 Python、Pillow、Node.js：

```bash
python -m unittest tests.test_reviewer_export_import_gate
python tools/audit_signoff_timestamp_parity.py
node tests/test_cvat_human_signoff_drafts.js
```

第一条预期通过；第二条在此快照预期退出 1 并报告 `UNSAFE_PAGE_LOOSER_THAN_IMPORTER`。第三条为接手者建议回归，本轮同步操作未复跑。不要通过管道丢失脚本真实退出码。

脚本的 `PARITY` 标签目前按 unsafe 数为零计算，不足以证明双向完全一致；验收时必须同时检查 `mismatch_count==0`。提取 JS 使用正则，若后续函数结构或依赖改变，应更新 harness，不能把提取失败当作校验失败。

## 下一轮最小实验

1. 在新的隔离 worktree 核对最新主线实现与此快照差异。只读源码，不打开真实数据。
2. 从正式导入器规则提取共享测试向量：闰年/日期边界、00–23 小时、偏移、必须带时区、小数秒位数、空白与类型。
3. 如缺陷仍存在，只修页面时间校验及相关测试；不改人工决定、不放松导入器。
4. 双端运行同一批向量，要求 mismatch=0；对有效向量运行 export→import，对无效向量确认页面不能正式导出。
5. 回归草稿往返、失败状态不变与授权注入测试。需要实际浏览器验收时只使用合成材料。
6. 提交独立分支并报告测试，不自动合并主线。下一轮推送须依当轮用户授权，不把本次同步授权无限延伸。

## 必须遵守的范围

不读取 test24、真实签字、凭证或私有图片；不训练、改真值/权重、部署或上传私有样本。生成器的合成测试可创建临时图像。无网络依赖、外部脚本、localStorage。reviewer 文本不具备身份认证或密码学签名效力。

本轮同步的是审计、测试和交接记录，不是主线合并、网站部署或模型实验。仓库 HTML 文件链接通常显示源码；本 Markdown 交接页可由仓库直接渲染，推荐给下一位 AI 阅读。

## 其他历史方向（不可混作本轮实验结果）

Round385 ego 四相机归一化点模板曾做只读审阅；camera1 可见面积双峰不能用来自动触发扩张。遮挡真值、person/floor 误纳入率及默认关闭 evaluator 的状态不变测试仍应独立核实。历史回答关于 NaN 的一处说明不准确：`not (0 <= NaN <= 1)` 实际为 True，会拒绝；不能据此直接宣称该条件允许 NaN。请以目标提交源码与复现实验为准。

主线后续 LiDAR、语义 canary、盲复核方向没有在本轮完整审阅，不能据提交标题概括其有效性。

# Runtime closed-loop 审查修复回执（2026-11-11）

## 范围与边界

本次只修改当前 LabelAgent 工作树中的运行时哈希账本、机器 schema、人工门禁同步工具和测试；没有覆盖 `task_workspaces/sfl_cpd15y_20261110` 的原契约，也没有调用 8767 的任何写接口。8767 仍保持 frame0 一个 slope 机器候选、truth/reference 为 0。

## 六项审查结论

1. **Receipt 幂等**：当前跟踪版本本身已没有 `generated_at`。现在进一步按 `last_event_id` 查找并复用既有 receipt；相同历史重复调用返回同一内容、同一哈希且只保留一个文件。若相同最后事件对应的输入、事件数或链尾身份冲突，则拒绝读回。
2. **事件重放**：duplicate `event_key` 查找保持在 transition 校验之前，并逐字段比对 state、actor、action、label layer、confidence、model、prompt、payload、artifacts 和 metrics。即使状态已经推进，完全相同的历史事件仍返回原 `event_id` 和 `idempotent_replay=true`。真正的新一轮复检必须改变动作、轮次或证据。
3. **拒绝/修复路径**：新类使用 `open_set_candidate → ontology_review_required → ontology_approved|ontology_rejected → remediation_required`。普通候选驳回写 `candidate_rejected → remediation_required → quality_checked`；旧 `human_rejected` 也不再是无法修正的终态。`closed_loop_human_gate.py` 已把 `reject_candidate` 映射到显式 `candidate_rejected`。
4. **Schema 收紧**：event 与 receipt 均关闭未知字段；state、actor、label layer、哈希、序号、时间、模型、prompt、artifact、metrics 和 receipt 字段均有枚举/格式/必填约束。`quality_checked`、`version_committed` 和 slope payload 也有状态特定约束；模型 actor 不能省略模型/prompt 版本。实际生成 bundle 已通过 Draft 2020-12 校验。
5. **质量与版本证据**：候选生成前必须有 `quality_checked` 且列出真实 checks。SAM/VLM/Agent 事件必须携带模型名/版本和 prompt 版本。所有 artifact 必须有小写 SHA-256。`version_committed` 必须记录 `version_id`、`parent_version`、`train|val|test`、`split_group_id`、ontology 版本、truth SHA、manifest SHA 和与输入一致的 real/sim 来源；manifest SHA 必须引用事件 artifact，同一 split group 跨 train/val/test 会被拒绝。
6. **Slope 安全契约**：slope hard case 必须记录 `semantic_label=slope_visible`、`human_pixel_truth_available`、`approved_for_training`、traversability、geometry sufficient 和 artifact ID。`traversable/non_traversable` 必须同时引用 `lidar_geometry` 与 `vehicle_constraints`；没有人工像素真值禁止 `evaluation_completed`，没有人工像素真值也不得批准训练。

## Receipt 与完成口径

`minimum_closed_loop_complete` 现在验证质量门、SAM、首次 VLM、Agent、再次 VLM、最终候选、绑定该候选 event/artifact 的人工接受及真值提交的先后关系，不再只检查状态是否曾出现。`human_gate_passed` 也只承认这种精确绑定；早期参考帧确认不计入。`training_ready` 只有在链接真值之后紧接一个字段完整、manifest 可核对且 split group 无泄漏的 `version_committed` 时才为 true。

## 验证

- 运行时账本、人工门禁、单图登记、高层 1.1 契约、真实 slope 事件及 vertical 三阶段流程：标准环境运行 38 项，37 项通过、1 项仅因标准环境未安装开发依赖 `jsonschema` 而跳过；schema 专用环境中的 vertical 测试 2/2 通过（同时拒绝缺 split 与缺 manifest artifact 的版本事件）。
- `source/agent/closed_loop.py` 与 `tools/closed_loop_human_gate.py`：Python 编译通过。
- `schemas/labelagent-closed-loop.schema.json`：JSON 解析通过。
- 当前工具在测试夹具上实际写出的完整文件（不裁剪 root、写后原样回读）：Draft 2020-12 schema 校验通过；这不是工厂真实 bundle 的通过声明。
- 四项负向 schema 样例均按预期拒绝：未知 event 字段、receipt 缺质量门字段、模型 actor 缺模型/prompt 版本、slope 无人工像素真值却声明完成评测。

## Vertical 工具补充复核

- 根 Schema 已声明 `registration`、`workbench_readback` 和 `bundle_sha256`；新增测试直接验证工具实际生成的初始与最终 bundle，而非手工拼装简化对象。
- 候选初次进入 `human_review_required`；完全相同重跑的 receipt、bundle 和输出字节保持不变。
- 模拟加入人工真值、参考帧和哈希有效的人审回执后，对此前 `candidate_final` 写入精确 event/artifact 绑定，再追加 `human_accepted → truth_committed → version_committed`。
- 原始 bundle 和既有 event 文件逐字节保持不变；最终状态写入新的不可变 sibling snapshot，数据 manifest 的文件 SHA-256 与事件 artifact 一致。

这只证明契约与状态机修复通过本地回归，不代表 8767 已产生人工真值、完整闭环已完成或服务器部署已更新。

## 真实完整 bundle 独立复核更正

此前 `SCHEMA_AND_REAL_BUNDLE_VALID` 只验证了抽取的 `schema_version/input/events/receipt` core，不能证明完整文件有效，该说法已撤回。当前 Schema 已正式声明 `registration`、`workbench_readback` 和 `bundle_sha256`，并新增整文件只读验证器 `tools/validate_closed_loop_bundle.py`，验证器同时核对文件前后 SHA 和 bundle 内部哈希，禁止投影 core。

对两份既有真实文件原样回读的结果仍为 **invalid**，且文件未变化：

- `frame0-gate-bundle.json`：文件 SHA-256 `1D5B1555CBB04FC6520FDF2DA08B6291A9C01F86BD799C9454694F3D2B8F8565`；receipt 缺 `quality_gate_observed`、`dataset_version`、`dataset_split`。
- `frame0-gate-replay.json`：文件 SHA-256 `417D24E01948C60C97ACF787067D204685CE0700DD913803C759AFE51592011A`；同样缺上述三个 receipt 字段。

两份旧文件的 `bundle_sha256` 自校验均正确，但这不等于 Schema-valid。需要由当前工具针对真实 frame0 生成新的不可变 sibling snapshot 后，再对该新文件原样验证；旧证据不得覆盖。

# LabelAgent 工厂数据闭环事件协议（目标架构）

> 状态：契约已落地，流水线尚未跑通。本文、事件 schema 和示例只定义目标协议与不可绕过门禁；它们不证明模型效果、微调完成、生产部署或最小闭环完成。`examples/closed_loop_unknown_candidate_event.json` 的对象名、哈希、时间、置信度和 URI 全部是显式占位值，不得作为真实数据或生产结果引用。

## 最终验收目标锁定（2026-11-11）

项目最终交付固定为持续运行的工厂 Label 闭环：生产数据持续到达后依次经过接入与质检、SAM/其他模型候选、VLM 已有类识别与疑似新类发现、Agent 拆分/合并/重分割/改类/去重、按需人工复核、不可变真值与 ontology 版本、分项评测、训练和可回滚部署，新模型再进入下一轮数据。单个模型服务、一次真实请求或某批数据的一次性自动标注都不是最终交付。

以下约束作为当前及后续工作的最高层验收门禁：

1. VLM 对新类别只能产生带证据的 `unknown_candidate`；人工批准前不得修改正式 ontology，批准后必须生成新的 `ontology_version`，并能追溯重跑受影响样本。
2. 机器预标注、Agent 候选与人工真值严格分层；Agent 的 `done` 只表示本轮停止，不表示正确，也不授权写入真值。
3. 仿真数据沿同一协议进入，但必须记录模拟器、场景、随机种子和版本。real、simulation 与 mixed 分项评测，真实工厂数据始终是最终验收集。
4. 双 5090 只属于基础设施里程碑。服务通过后先验收一帧真实数据的完整事件链，再做小批量和送标成本统计，最后才进入持续接入与版本回灌。
5. 从现在起保留事件、模型、prompt、Agent 工具动作、数据/ontology 版本、人工决定与效果变化的时间戳和哈希，为潜在专利提供可追溯证据；在完成独立评估和正式程序前不宣称专利成果。

## 与当前实现的关系

LabelAgent 现在有两层互补证据：

- `source/agent/closed_loop.py` 与 `schemas/labelagent-closed-loop.schema.json`：当前运行时使用的逐帧追加式哈希账本和回执；
- `source/agent/closed_loop_contract.py` 与 `schemas/labelagent_closed_loop_event.schema.json`：跨组件交换的 1.1 版高层事件协议。

运行时输入中的 `source_kind=sim` 映射为事件协议的 `source.origin=simulation`，但原始账本字段不原地改名。移植事件协议不能覆盖既有账本、候选、人工真值或服务状态。

## 目标

工厂真实数据和仿真数据持续进入后，由 LabelAgent 完成输入质检、候选生成、VLM 识别、Agent 纠错、按需人工复核、真值提交、不可变数据集版本、评测、训练和可回滚部署。预标注、Agent 候选和人工真值始终分层；`done` 只表示一轮停止，不表示正确。

双 5090 上的 SAM/VLM 服务健康只是基础设施里程碑。验收顺序为：真实服务请求 → 一帧完整事件链 → 小批量与人工成本统计 → 持续接入与版本回灌。

## 状态机

```text
INGESTED
  ├─ QC_FAILED -> QUARANTINED
  └─ QC_PASSED -> CANDIDATES_GENERATED
                    -> VLM_REVIEWED
                       ├─ KNOWN_LABEL -> AGENT_ITERATION <-> VLM_REVIEWED
                       │                  └─ REVIEW_REQUIRED
                       └─ UNKNOWN_CANDIDATE -> ONTOLOGY_REVIEW_REQUIRED
                                               ├─ REJECTED
                                               └─ ONTOLOGY_VERSIONED
                                                    -> REPROCESS_REQUIRED

HUMAN_APPROVED
  -> TRUTH_COMMITTED
  -> DATASET_VERSIONED
  -> EVALUATED
  -> TRAINED
  -> DEPLOYMENT_CANDIDATE
  -> PROMOTED / ROLLED_BACK
```

协议保留 `auto_accepted` 状态供未来经过书面批准的自动接收策略使用，但当前部署没有这样的阈值。当前 `closed_loop_contract.py` 因而采用更严格默认：`truth.committed` 必须从 `human_approved` 进入，并携带已批准、带时间戳的人工复核记录。

## 不可绕过的门禁

1. 每帧包含数据集、批次、帧、相机/topic、采集时间、real/simulation 来源和内容 SHA-256；相同 `idempotency_key` 不得产生冲突真值。
2. VLM 只能提交 `unknown.proposed`，必须包含证据、候选名称、最相近类别、差异和置信度，且 `ontology_mutation_allowed=false`。
3. 拆分、合并、删除、重分割和改类记录前后对象、工具参数、模型/提示版本与原因；超轮次、冲突或质检失败进入人工。
4. 机器候选、Agent 最终候选和人工真值分别保存。当前运行时没有人工批准不得进入 `truth.committed`。
5. train/val/test 按批次或序列隔离并生成不可变版本；real 与 simulation 分开记录来源、权重和指标。
6. 模型、ontology、提示和数据版本均可追溯；新模型未通过主指标与泄漏护栏不得晋级，晋级后必须可回滚。

## 最小闭环验收

一帧真实数据至少可回读：

- `frame.ingested`：输入身份、来源、采集时间与内容哈希；
- `quality.checked`：通过或隔离理由；
- `candidate.generated`：SAM 候选、权重与代码版本；
- `vlm.reviewed`：已有类判断或 unknown 证据；
- `agent.action`：至少一轮可审计动作或明确 no-op；
- 再次 VLM 复核证据；
- `human.reviewed`：当前严格运行时必须存在；
- `truth.committed`：与候选严格分层；
- `dataset.versioned`：不可变版本、split、ontology 与来源明确。

只有这些事件能以同一 `run_id` 和 `correlation_id` 串联回读，底层哈希链校验通过，且最终回执明确，才称“最小闭环跑通”。服务健康、单次推理成功或界面出现候选均不等价。

## 新类别规则

unknown 候选至少记录：建议名称/同义词、证据图或掩码及哈希、最相近既有类与差异、置信度、冲突意见、建议动作和人工 ontology 审核状态。人工批准后生成新 `ontology_version`，再对受影响样本可追溯重跑；禁止原地修改历史真值。

## Slope 与可通行性

`slope_visible` 是视觉语义；`traversable` 是融合决策，两者不得混同。没有 LiDAR 几何、坡度、宽度、边缘落差、障碍物和车辆约束证据时，只能保持 `unknown` 或 `review_required`。如果报告 slope IoU、Dice 或边界 F1，必须有独立人工像素真值。

## 仿真数据

仿真事件必须记录模拟器、场景配置、随机种子和传感器版本。真实、仿真、混合训练分别报告，禁止用仿真提升代替真实工厂验证。

## KPI

- 自动可接受率；
- 人工分钟/帧及节省比例；
- unknown precision/recall 与新类确认率；
- 错误标签泄漏率；
- 端到端延迟与单帧成本；
- Agent 平均轮次、超限率和冲突率；
- real/simulation 分项指标与部署漂移。

无真实分母或独立人工结果时，指标为 `null`，不得填 0。

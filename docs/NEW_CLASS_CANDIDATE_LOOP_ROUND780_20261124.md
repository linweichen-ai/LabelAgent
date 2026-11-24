# Round780：新类别候选闭环

## 问题

此前 VLM 能返回 `unknown_candidates`，页面也会提示“疑似新类待人工命名”，但单帧结果没有形成跨帧候选池。这样既容易把偶发物体误当新标签，也可能把 `forklift`、`other_robot` 之类已有标签或同义词再次注册。

## 本轮实现

新增 `tools/build_new_class_candidate_pool.py`，处理流程为：

`机器未知类输出 → 已有标签别名拦截 → 显式同义词合并 → 同帧去重 → 跨帧/跨序列统计 → 人工下游价值复核模板`

核心规则：

- 只合并配置文件中明确列出的同义词，不让模型自行改名或扩充本体。
- 同一帧重复输出 `rack/shelf/货架` 只计一次，避免通过重复输出伪造“多次出现”。
- 至少跨 2 个不同帧出现，才进入人工下游价值复核队列；单帧出现只保留为证据不足。
- `车辆`、`pallet`、`person`、`floor`、`slope`、`ego`、`other_robot`、`cargo_floor` 及其显式别名永远进入“已有标签冲突”，不会成为新类别。
- `absent` 提议不进入候选池。
- test24 记录、重复 observation ID、不安全训练边界、模糊别名配置全部失败关闭。
- 即使人工在复核模板中选择“接受且具有下游价值”，校验结果也只生成 `manual_registry_proposals`，不会自动改标签注册表。

配置和输入示例：

- `docs/templates/new_class_policy.example.json`
- `docs/templates/new_class_observations.example.json`

## 真实历史输出只读审计

对 Round191 的两个历史影子输出进行只读审计：

- 机器在开放集字段中提出 `other_robot`；
- 其中 1 次为 `present`，另 1 次为 `absent`；
- `other_robot` 本来就是正式已知标签，因此被归入 `known_label_alias_not_new_class`；
- 新类别组 0，人工新类队列 0，自动注册 0，训练批准 0。

这是一个有意义的负结果：当前这两条历史输出不能证明发现了新类，反而证明必须先做已知标签别名拦截。

候选池 SHA-256：

`618362cf46b2520255701230576a5dd3f59aa1279cf0ee21806ecad25185a45a`

## 安全边界

- `candidate_only=true`
- `human_truth=false`
- `approved_for_training=false`
- `test24_used=false`
- `automatic_registry_mutation=false`
- `registry_mutated=false`
- 外部 API、训练、推理和正式部署均为 0

本轮没有声称发现任何真实新类别，也没有把货架、椅子、箱子等普通物体自动加入项目标签。

## 验证结果

- 新类别聚合、别名拦截、同帧去重、test24 拒绝、边界拒绝和人工复核校验：8/8 通过。
- 项目主测试集：1875 通过、1 跳过、286 个子测试通过。
- JSON 模板和结构化证据可解析，`git diff --check` 通过。

## 下一步

后续影子运行可把新的 `unknown_candidates` 按示例格式追加到观察清单。只有某一候选跨帧持续出现，并由人工确认类别语义及下游价值后，才允许提出单独的人工注册变更；注册完成后仍需另行采集和审批训练样本。

## Round788：工作台只读导出接通

新增 `GET /api/new_class_observations`，直接从当前数据和图像源已经持久化的参考帧机器草稿中导出候选池输入：

- 序列路径只输出 SHA-256 前缀，不暴露原始本地或服务器路径；
- 旧记录中不规范的自然语言存在性证据一律降级为 `uncertain`，不猜测为 `present`；
- 明确的 `present/absent/uncertain` 原样保留；
- 发现 test24 记录立即拒绝整个导出；
- 接口只读，不生成文件、不修改标签注册表、不批准训练。

人工标注页增加“导出新类别观察”按钮；有记录时直接下载 JSON，没有记录时明确提示为空，不再要求用户手工拼接接口地址。

因此新类别链路现在可以从“工作台单帧发现”连续进入“跨帧候选池”，不再需要人工抄写 JSON。候选仍必须经过跨帧支持门和人工下游价值复核，注册表自动修改数仍为 0。

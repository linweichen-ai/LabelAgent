# Round 297：逐类别候选风险提示（2026-11-20）

## 目的

原工作台只显示“某类别已生成 N 个候选”。这会把“已经生成”误读为“这些候选都可靠”。
本轮将 Agent 和类别盲审已经绑定到当前候选的风险直接汇总到类别卡片，但仍严格保持：

- 风险记录不自动删除候选；
- 风险记录不自动修改类别；
- 风险记录不写入人工真值；
- 未经人工确认的候选不批准进入训练或部署。

## 实现

- 按类别统计 `candidate_class_audits` 与 `review_queue` 中绑定当前候选的风险记录；
- 同一类别、同一候选在两个来源重复出现时只计一次；
- 有风险的类别使用琥珀色提示，并显示“已生成 N 个 · M 个需复核”；
- 点击卡片后说明风险来自 Agent 或类别盲审，必须由人工决定；
- 明确提示系统不会据此自动删除候选或改写真值。

## 自动测试

运行：

```text
node --check source/web/app.js
node --test tests/test_web_polling.js tests/test_web_candidate_audit.js
```

结果：139/139 通过，0 失败。新增用例覆盖风险汇总、跨来源去重、普通候选与风险候选状态区分，
并检查不自动删除候选或改写真值的边界文案。

## 真实批次 B 003 验收

隔离工作台只读恢复先前生成的真实 003 候选，没有重新推理、没有占用 GPU。页面读回：

| 项目 | 结果 |
| --- | --- |
| 数据身份 | SFL-CPD15-Y · 批次 B（1104）· 问题包 003 · 单图抽样 |
| 候选总数 | 17 |
| 类别分布 | pallet 13、person 2、floor 1、ego 1 |
| person 卡片 | `person：已生成 2 个 · 2 个需复核` |
| Agent 复核 | complete |
| 类别盲审 | complete |
| 人工真值 | 0 |
| 参考帧 | 0 |
| 页面错误 / HTTP 错误 | 0 / 0 |

两个 person 风险分别来自：一个 Agent 提议的高风险 person→ego 互改被门禁拒绝；另一个极小候选
在类别盲审中因证据不足转人工复核。候选保持17个，没有根据风险记录自动删除或改类。

结构化证据：

- `docs/evidence/workbench-stability-20261117/candidate-coverage-risk-round297.json`
- `docs/evidence/workbench-stability-20261117/candidate-coverage-risk-round297.png`
- `docs/evidence/workbench-stability-20261117/candidate-coverage-risk-round297.cjs`

验收脚本第一次只等待8个卡片出现，早于按需候选几何读回，因而短暂读到“尚未评估”；随后把验收
条件加强为候选已绑定到页面后再判断。第二次正式验收通过。这是测试时序修正，不是把失败结果隐藏。

## 生产部署与回滚

只替换 `source/web/app.js`、`source/web/style.css`、`source/web/index.html`。部署前备份：

```text
D:\Apps\LabelAgent\.local\backups\candidate-coverage-risk-round297-20261120-050633
```

部署后开发/生产哈希逐项一致：

| 文件 | SHA-256 |
| --- | --- |
| app.js | 17BF2F7639D791F39C317566A4834294C452B126C68B257D86978EE28C6B9A7E |
| style.css | D61CC0E4E5B6BFC95B37BA38B2FB6D38189125C2CD6C5365927FF0F417179B8A |
| index.html | A38A4D108A33771559712A0CEA02E947F33F602C3C18131409BA19542CFDCB77 |

HTTP 已返回 `app.js?v=198` 与 `style.css?v=119`。部署前后正式工作区均为 CVAT Task3、
`/CVAT/Task3/images`，人工真值帧数均为1；没有切换生产数据、运行推理或修改模型权重。
隔离端口18793在验收后已关闭。

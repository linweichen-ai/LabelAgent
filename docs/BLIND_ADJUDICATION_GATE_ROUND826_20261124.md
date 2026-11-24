# Round826：困难帧先盲判、后对照门

## 结论

Round825 已经把 11 张模型分歧帧排到人工复核队列最前，但原页面仍允许复核人在填写前主动展开机器建议，存在确认偏差风险。Round826 增加失败关闭的盲判门：当前帧全部核心字段完整填写之前，鼠标点击或快捷键 H 都不能显示机器建议；完成独立判断后才允许展开对照。

该门只约束复核顺序，不把复核记录升级为像素真值或训练授权。

## 规则

显示机器建议前必须具备：

1. 候选结论；
2. 斜坡存在性；
3. 类别冲突；
4. 托盘质量；
5. 八个项目类别的存在性；
6. 非负整数的可见托盘实例数。

任一字段缺失时，建议面板立即关闭并提示“请先完整填写本帧独立判断，再查看机器建议”。机器建议不会预选、复制或提交任何答案。

## 真实工作区

最终入口：

`C:\Users\seer\Documents\ChatGPT\Seer2clw\task_workspaces\labelagent_stability_worktree_20261117\artifacts\round826-blind-priority-adjudication-workspace-20261124\reviewer.html`

| 项目 | 结果 |
|---|---:|
| 总帧数 | 34 |
| 优先分歧帧 | 11 |
| 机器建议 | 34 |
| `assistant_hint_blind_gate` | `true` |
| 页面 SHA-256 | `c03b4ad4423b41d2c970a37eb8231ee943213edde953bb3493d42f7bcae4f87d` |
| `candidate_only` | `true` |
| `human_truth` | `false` |
| `approved_for_training` | `false` |

## 验证边界

- 定向 Python：22/22 通过；
- 全量 Python：2009 通过、1 跳过、294 个子测试通过；
- 网页逻辑：199/199 通过；
- 生成页面通过独立 JavaScript 编译检查；
- 工厂图片外发、外部 API 调用、test24 使用、训练、正式部署均为 0；
- Round825 v2 继续保留作可追溯历史，但新的人工复核应使用 Round826 入口。

> 后续更新：真实浏览器验收发现并修复 JSON 实体初始化失败、机器置信度泄露和粘性操作栏遮挡。最终人工入口已升级为 Round830，详见 `docs/BLIND_ADJUDICATION_BROWSER_ACCEPTANCE_ROUND830_20261124.md`。

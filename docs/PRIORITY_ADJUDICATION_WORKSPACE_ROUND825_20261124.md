# Round825：困难帧优先人工复核工作区

## 目的

将 Round824 中已经通过哈希审计的 11 张斜坡分歧帧放到 34 张困难复核集最前面，缩短人工形成可用裁决数据的路径。此改动只改变离线复核顺序，不改变 SAM3、Qwen8B、Astra 或 Agent 的正式推理结果。

## 实现

- `build_mixed_hard_adjudication_workspace.py` 新增可选参数 `--priority-triage`。
- 优先队列必须同时绑定原始复核包和本地 Qwen 建议文件，并通过 `audit_second_assistant_slope_triage.py` 全部边界检查。
- 复核顺序固定为高置信度分歧 3 张、中置信度分歧 6 张、低置信度分歧 2 张、其余 23 张原顺序。
- 页面只显示中性的“优先复核”徽标，不会自动选择 `present`、`absent` 或任何类别答案。
- 机器建议仍默认折叠，只有人工按 H 或主动展开后才显示。
- 回执新增 `priority_review_count` 和 `priority_triage_sha256`，便于复现和防止队列漂移。

## 真实工作区读回

生成目录：

`C:\Users\seer\Documents\ChatGPT\Seer2clw\task_workspaces\labelagent_stability_worktree_20261117\artifacts\round825-priority-adjudication-workspace-v2-20261124`

读回结果：

| 项目 | 结果 |
|---|---:|
| 困难帧总数 | 34 |
| 优先分歧帧 | 11 |
| 高 / 中 / 低优先级 | 3 / 6 / 2 |
| 复制图像 | 68 |
| 工作区文件 | 71 |
| 优先队列 SHA-256 | `2c92ad1020b327578051b140e69f4f4990fab347940c11711482521f20783325` |
| `candidate_only` | `true` |
| `human_truth` | `false` |
| `approved_for_training` | `false` |

前 11 项全部为优先分歧帧，第 12 项起没有优先标记，说明排序没有把普通样本混入优先区。

## 验证与失败记录

- 定向 Python：22/22 通过。
- 全量 Python：2009 通过、1 跳过、294 个子测试通过。
- 网页逻辑：199/199 通过。
- 第一次真实命令行构建暴露直接运行脚本时无法导入审计模块，未写出工作区；已加入仓库根目录入口兼容和命令行回归测试后重建成功。
- 第一次全量测试因 Windows GBK 和残留的 Git 临时环境配置出现 18 项环境失败；隔离复测 12/12 通过，按 UTF-8 且清除损坏临时 Git 配置后全量通过。
- 第一次网页测试为 198/199，唯一失败是 Node 子进程误用无 Pillow 的系统 Python；显式使用项目虚拟环境后为 199/199。
- 第一版生成页进一步被独立 JavaScript 编译检查发现换行转义错误；该无效目录只保留作失败证据。v2 页面 SHA-256 为 `12fc8944cff96c6b5039e252e3ec67999cf8b9547aa2d7f2b138b22517592b31`，并已通过实际 `new Function` 语法编译和自动回归测试。

## 当前结论

该工作区已经能让人工先处理最有训练价值的模型分歧帧，但它没有产生人工裁决，也没有授权后训练。下一门槛仍是人工逐帧确认；在完整人工记录通过校验之前，后训练提案工具会继续失败关闭。

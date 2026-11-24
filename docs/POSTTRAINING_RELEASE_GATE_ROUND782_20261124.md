# Round 782：后训练发布门禁补强

## 结论

项目已有一轮 Qwen3-VL-2B 离线 LoRA，但它使用机器教师数据，只能证明训练流程执行过，不能替代人工真值训练，也不能直接部署。最新 34 帧困难集仍在人工复核阶段，因此本轮没有启动新训练。

新增 `tools/gate_posttraining_release.py`，把“数据可训练、训练真实完成、对照评测可比、效果不退化、可回滚”五件事绑定为一个失败即关闭的门禁。它只做审计，不训练、不推理、不部署。

## 必须同时满足的条件

1. 训练样本全部经过人工复核并明确批准，机器候选不能冒充人工真值；
2. `test24` 永久只用于评测，不能进入训练；
3. 数据清单、训练配置、训练日志、适配器均有 SHA-256 绑定；
4. 有训练步数、GPU 数量、GPU 型号和显存使用证据；
5. 基线与新模型使用相同协议、相同样本和人工参考；
6. 总体 F1 达到最小提升，同时斜坡误检不能突破保护线；
7. 旧模型有已验证的回滚制品和恢复命令。

即使全部通过，状态也只是 `eligible_for_isolated_release_review`，不会自动授权生产部署。

## 当前状态

当前 34 帧 Round779 困难集仍为 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`，所以门禁应当阻止开训。这是预期保护，不是程序故障。完成人工复核后，才可生成训练清单、运行隔离后训练，并用 Round781 的三臂协议做同样本对照。

示例输入：`docs/templates/posttraining_release_manifest.example.json`。

## 实测与边界

- 当前示例按真实阶段边界填写为34张候选、0张人工复核、0张批准训练，门禁返回 `blocked`，退出码为2；未启动训练、推理或部署。
- 新增6项门禁定向测试；连同既有后训练、Linux和复现检查共17项通过。
- 按 Windows UTF-8 环境运行项目测试：`1887 passed, 1 skipped, 286 subtests passed`。
- 直接从仓库根目录无边界收集测试会额外收集上游 SAM3 的可视化脚本，并因未安装可选 `matplotlib` 在收集期中止；项目正式回归范围为 `tests/`，该问题不属于本轮逻辑回归。
- 外部 API 调用0次，正式工作台写入0次，人工真值写入0次，`test24` 使用0次。

结构化证据：`docs/evidence/workbench-stability-20261117/round782-posttraining-release-gate-summary.json`。

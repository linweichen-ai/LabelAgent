# Round 852：困难集跨文件身份门禁修正

## 背景

一次准备性复核发现，旧的本地未发布审计误把早期 `round776` 复核包与当前 `round779` 的 Qwen 建议、混合队列组合在一起。旧工具分别验证了每个文件的结构，却没有强制检查它们是否绑定同一份复核清单，因此没有立即失败。

这不影响已生成的机器建议或正式工作台，但如果继续使用错误报告，可能让后续三路实验比较不同样本集合。

## 修复

`audit_mixed_hard_experiment_readiness.py` 新增两层失败关闭：

1. 本地 Qwen 预测包的 `manifest_sha256` 必须与传入复核包的实际 manifest 哈希一致；
2. 混合队列和全外部队列的 `source_suggestions_sha256` 必须与本地 Qwen 预测文件哈希一致。

任一绑定不一致，相关组件会明确报告 `review manifest binding differs` 或 `queue source suggestions binding differs`，三路质量评测门禁保持关闭。

## 正确冻结包复核

使用 `round779` 的 34 帧复核包重新审计：

- manifest SHA-256：`4cd7eadb244cadad83208b4f2d9f304da457830b839070dd796edf77f53dad50`；
- 本地 Qwen：34/34，准备完成；
- 混合外部队列：17/17，准备完成；
- 全外部队列：34/34，准备完成；
- 人工复核：0/34，未完成；
- 外发授权：未授权；
- 外部结果：尚未生成；
- 本次外部调用：0；
- `test24` 使用：0；
- 正式部署：0。

定向测试覆盖正确绑定、复核模板进度、完整门禁、manifest 错配和队列源错配，共 5/5 通过；项目全量回归 `2063 passed, 1 skipped, 294 subtests passed`。

## 当前结论

三路实验的机器侧输入已经正确对齐，但质量比较仍不能开始：需要先完成 34 帧人工复核，并获得开发图片外发授权。机器结果继续保持 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`。

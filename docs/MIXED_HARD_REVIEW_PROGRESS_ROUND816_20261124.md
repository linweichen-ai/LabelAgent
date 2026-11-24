# Round816：困难集人工复核状态显式化

## 修复内容

旧版就绪审计把未填写的 `adjudication-template.json` 直接交给正式提交校验器，因此显示 `unsupported adjudication schema`。该提示容易被理解为复核包损坏，实际情况只是尚未从 `reviewer.html` 导出正式复核文件。

新版审计会区分：

- 空白或部分填写的复核模板；
- 已导出但仍含不确定项的完整复核；
- 可用于基准评测的完整复核。

对于模板会明确给出总数、已完成数、剩余数、复核入口和预期导出文件名，不会自动填入任何机器判断。

## 当前实测状态

- 样本总数：34；
- 已完成人工复核：0；
- 剩余：34；
- 复核入口：`artifacts/round803-diverse-hard-adjudication-resumable-20261124/reviewer.html`；
- 需要导出的文件：`mixed-hard-adjudication.json`；
- 本地Qwen结果：34/34就绪；
- 混合路由队列：17/17就绪；
- 全外部队列：34/34就绪；
- 外部图片授权：尚未获得；
- 本轮API调用：0；
- 本轮真实图片上传：0。

最新只读审计位于 `artifacts/round816-mixed-hard-readiness-audit-20261124/readiness.json`，SHA-256 为 `5129abc97dd3db356d4b19cc77147098a3817f54d3a8ebc37cabc15c8d1033c7`。

## 安全边界

复核模板和结果始终保持 `candidate_only=true`、`human_truth=false`、`approved_for_training=false`。即使完成34帧复核，也只形成声明式开发基准，不会自动成为训练真值、不会触碰test24、不会自动部署。

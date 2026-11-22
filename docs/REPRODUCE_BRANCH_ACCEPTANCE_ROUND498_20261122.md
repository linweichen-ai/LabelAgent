# Round498：CNB/GitHub 公开复现分支同步验收

日期：2026-11-22

## 问题

README 要求外部使用者克隆 `reproduce`，但 CNB 与 GitHub 的该分支此前都停在
`f398675ac893e89d76dcd9953a6eaf255d4e2101`，落后当前开发线 44 个提交、涉及 291 个文件。
继续只推 `labelagent` 会导致文档中的“最短复现路径”拿到旧代码。

## 发布前验收

确认旧 `reproduce` 是当前提交的祖先，可以无冲突快进。随后在待发布代码上执行完整
`tools/verify_checkout.ps1`：

- `repro_doctor`：READY；
- Python 版本与核心依赖：通过；
- `pip check`：无依赖冲突；
- Python：1,425 项通过、1 项跳过；
- 浏览器逻辑：167/167 通过；
- CUDA：可用；
- SAM3 权重是可选项，本轮没有把权重带入 Git。

## 发布与读回

经过验收的 `2f24b662c16bccb0db7e18d25d804d7665f6d3f0` 已快进到：

- CNB `labelagent`；
- GitHub `labelagent`；
- CNB `reproduce`；
- GitHub `reproduce`。

四个远端引用均重新读回为相同 SHA。模型权重、原始数据和凭证均未上传；外部克隆现在能拿到
最新代码、受控 LoRA 构建/验收工具及最新复现文档，但仍需按说明自行提供有授权的模型权重。

机器回执：
`artifacts/round498-reproduce-branch-publication-20261122/receipt.json`

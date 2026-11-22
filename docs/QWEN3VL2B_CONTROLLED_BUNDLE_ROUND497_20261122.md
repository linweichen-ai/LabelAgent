# Round497：2B VLM LoRA 本地受控复现包

日期：2026-11-22

## 结果

基于 Round496 的完整性清单，已把 Round375 的 Qwen3-VL-2B LoRA 候选构建成确定性本地 ZIP：

- 文件：`.repro/qwen3vl2b-round375-controlled/labelagent-qwen3vl2b-round375-lora.zip`；
- 大小：36,191,268 字节；
- ZIP SHA-256：`cde1b62b4537915fe8f426eed8f99bd06d9bccb2657d55ec2394ea01d18dec01`；
- 成员：12 个适配器文件，加独立清单和使用说明，共 14 个；
- 基座身份：`Qwen3-VL-2B-Instruct`；
- 权重 SHA-256：`ad81bd3929ffb915b07986e783fe2116defe51c45d3ff562d86e5d164f59ad4f`。

构建后重新打开 ZIP，逐成员核对文件集合、大小和 SHA-256；随后从 ZIP 中取出 safetensors，
完整扫描 392 个 LoRA 张量，全部有限，196 个 A 张量均非零、196 个 B 张量均发生更新。

## 可复现工具

- `tools/build_qwen3vl_controlled_bundle.py`：只接受已通过 Round496 门禁的本地候选，构建
  确定性 ZIP，并在完成后逐成员回读；
- `tools/verify_qwen3vl_controlled_bundle.py`：接收方只需 ZIP 和预期整包 SHA，即可独立核对
  安全路径、成员集合、逐文件哈希、基座身份和可选的完整张量扫描。

构建回执：
`artifacts/round497-qwen3vl2b-controlled-bundle-20261122/receipt.json`

独立验收：
`artifacts/round497-qwen3vl2b-controlled-bundle-20261122/verification.json`

## 分发边界

GitHub 仓库当前为公开仓库，且项目明确约定模型权重不随 Git 发布。因此 ZIP 位于被忽略的
`.repro` 本地目录，`remote_uploaded=false`；本轮没有把权重推到 GitHub/CNB，也没有改变正式
8B VLM。公开仓库只同步构建器、验收器、文档和不含权重的小型回执。

该包仍是机器教师 LoRA 离线候选，不是人工真值模型，也未获生产部署批准。若未来决定受控
交付，应通过权限受限的制品库传输 ZIP，并要求接收端使用本轮工具和 SHA 完整复验。

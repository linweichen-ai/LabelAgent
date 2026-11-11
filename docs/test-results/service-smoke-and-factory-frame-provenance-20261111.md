# 双服务 smoke 与真实工厂帧来源回执（2026-11-11）

## 结论

仓库演示图 `truck.jpg` 已完成 SAM/VLM 服务加载与接口 smoke，但它不是工厂数据，也不是闭环证据。随后从批次 A 单独选取并核验了一张 `cameras_640x512` 工厂图；该图已登记到不可变输入账本，并真实请求了 SAM 和 VLM 服务。目前账本仍只有 `received`，`minimum_closed_loop_complete=false`，不能声称完整 LabelAgent 闭环跑通。

## 仓库演示图 smoke

- 输入：`source/sam3/assets/images/truck.jpg`，271,475 bytes，SHA-256 `941715e721c8864324a1425b445ea4dde0498b995c45ddce0141a58971c6ff99`。
- SAM：HTTP 200，0.09 秒，score 0.94921875，1 个 part；结果 SHA-256 `99cec29c463ce9810dcd0e3f1c299791a6b914ff3feaa9c0429d2306396a8f9b`。
- VLM：真实 CUDA 服务收到 1 张图，结构化请求通过；输出明确为 `full_agent_verified=false`。
- 5090 回执目录：`/home/clw/run/labelagent-5090/service-smoke-demo-1htiovlp`。

这组证据只能说明模型已加载且接口能接收真实图片请求，不能替代真实工厂帧、候选复核、Agent 动作或人工门禁。

## 真实工厂帧

- OSS 原始路径：`oss://seer-kubeflow/dataset/robot_data/SFL-CPD15-Y/20260920_cs-heli_Dev_000001/000/bevannotation_data/cameras_640x512/camera_0/1789897506782450197.png`
- 编号目录：`000`
- `frame_id`：`1789897506782450197`
- camera：`camera_0`
- 采集时间：`2026-09-20T09:45:06.782450197Z`（由纳秒级 `frame_id` 解释）
- 图像：640×512 RGB PNG，416,542 bytes
- SHA-256：`34e5094d24db487b3fcd5c458ac810cde5d58293f89fe0b92a2933c72a146812`
- 标签交叉核对：同名标签 `frame_id` 一致，18 个目标，类别含 `Pallet`、`Pedestrian`；四路相机均存在该帧，并列在 `specified_annotation_frames.csv` 中。

`ossutil ls` 读回对象大小为 416,542 bytes，ETag 为 `5E5B2FB31645FE726FD109F9C62E0E63`。`ossutil cat` 默认会在对象流后附加 elapsed 文本，必须加 `-q`；使用 `ossutil cat -q` 后，OSS 流、4090 落地副本和本机单图三者 SHA-256 完全一致。

不可变输入登记结果：

- `input_id`：`bc706b57eedaea59e3deda1e25eb5a20002e90b61089083e84d2f98130c5600b`
- 状态：`received`
- 事件数：1
- 回执 SHA-256：`87d4316cb4e11424b278b5f0918fcc05bfc5b94f709fcb9bf957d0b6412064a2`
- 本地登记回执：`C:\Users\seer\Documents\ChatGPT\Seer2clw\work\clfi\registration.json`

## 同一工厂帧的模型请求

SAM 使用 Codex/assistant 选择的黄色坡板内部正点 `(390, 220)`；来源固定记录为 `assistant_selected_prompt` / `operator_seed`。它不是人工选择、不是 human seed，更不是人工真值，也不属于纯自动候选发现。

- SAM：HTTP 200，0.065 秒，score 0.98046875，1 个 part、103 个外轮廓顶点、0 个洞、无 warning。
- SAM 结果 SHA-256：`0be59d677c0a910fbf79437484f25cdfdd8256aed4e347a656447bbe9f642593`
- VLM：真实服务收到同一张工厂图，结构化图像请求通过，prompt/completion/total tokens 为 414/31/445。
- VLM 输出明确为 `full_agent_verified=false`。
- 5090 回执目录：`/home/clw/run/labelagent-5090/factory-frame-1789897506782450197-AWkFGAWh`。

## 尚未完成的闭环步骤

当前 VLM 请求只验证了原图图像接口，并未读取或复核 SAM mask；因此以下步骤尚无真实证据：SAM 候选感知的 VLM 分类与漏误标判断、Agent 拆分/合并/重分割/改类/no-op 动作、第二轮 VLM 复核、人工接受或驳回、真值提交、不可变数据版本。下一步必须沿用上述 `input_id` 完成这些事件，且人工接受前不得写生产 ontology 或真值。

机器可读明细见 `service-smoke-and-factory-frame-provenance-20261111.json`。

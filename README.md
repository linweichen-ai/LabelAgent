# 视频全景自动标注原型

## 最短复现路径

**从 CNB 或 GitHub 复现时都只下载 `reproduce` 分支。** `labelagent` 是持续开发分支，其他
`codex/*` 和 `backup/*` 分支仅保留历史记录，不作为复现入口。

Windows + NVIDIA GPU 用户可直接执行：

```powershell
git clone --branch reproduce --single-branch https://cnb.cool/seer-robotics/EIC/BEV/LabelAgent
cd LabelAgent
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
# 将获得授权的 sam3.pt 放入 source\sam3\weights\
powershell -ExecutionPolicy Bypass -File tools\start_local.ps1
```

若使用 GitHub，复现入口同样为 `reproduce`：

```powershell
git clone --branch reproduce --single-branch https://github.com/linweichen-ai/LabelAgent.git
```

随后打开 `http://127.0.0.1:8765`。环境、依赖、权重及服务可用性可由 `tools/repro_doctor.py` 读回检查；完整代码回归使用 `tools/verify_checkout.ps1`。详细步骤、VLM 配置边界、验收口径和最小复现记录见 [可复现安装与验收](docs/REPRODUCIBLE_SETUP.md)。模型权重、MCAP、人工真值、运行状态与密钥不随 Git 发布，也不会被启动脚本上传。

Linux 用户使用独立的 `linux` 分支：

```bash
git clone --branch linux --single-branch https://github.com/linweichen-ai/LabelAgent.git
cd LabelAgent
chmod +x tools/*_linux.sh
tools/setup_linux.sh --web-only
tools/start_linux.sh --sam-url http://SAM服务器地址:8766 --no-browser
```

完整 GPU 安装时去掉 `--web-only`。完整安装、SSH 隧道、停止方式和故障排查见 [Linux 兼容版教程](docs/LINUX_SETUP.md)。浏览器必须访问 `http://127.0.0.1:8765/`，不要直接打开 `source/web/index.html` 的 `file://` 静态页面。

当前稳定范围、分支职责、模型资产、后训练状态和清理恢复方式见 [当前交付与完整复现说明](docs/CURRENT_RELEASE_AND_REPRODUCTION_20261127.md)；服务和数据边界见 [架构说明](docs/ARCHITECTURE.md)；相关论文与官方实现见 [参考文献与上游依赖](docs/REFERENCES.md)。

首次交接给团队同事时，建议直接使用包含操作截图、安装清单、完整验收流程和常见故障处理的 [同事完整复现教程](docs/LabelAgent_同事完整复现教程_20261123.md)。

无需模型权重、MCAP 或私有标注，也可用公开合成数据真实验证离线人工复核页的草稿导出、刷新恢复、篡改拦截和训练授权门禁。`tools/setup_windows.ps1` 已安装所需浏览器验收依赖，标准 `tools/verify_checkout.ps1` 会自动执行该真实浏览器自检；该过程只访问本机回环地址，不会产生人工真值或训练批准。

若只想先验证公开复核页面，不准备安装 CUDA/PyTorch、Node.js 或模型权重，可直接运行 `tools/setup_public_browser_check.ps1`。它在 `.repro` 下建立独立的轻量环境，仅安装固定版本的 Pillow 与 Playwright，完成后给出真实 Edge 浏览器回执。

托盘边界候选的六联“所有权”证据可转换成工作台只读复核包。下面的命令只复制已绑定哈希的证据图，不修改候选 mask，也不会写入人工真值或训练集：

```powershell
.\.venv\Scripts\python.exe tools\build_pallet_boundary_component_review_pack.py `
  --review-manifest <第一批所有权复核目录>\manifest.json `
  --review-manifest <第二批所有权复核目录>\manifest.json `
  --output work\component_review_packs\pallet-ownership-review
```

重新打开自动标注页，在“组件证据复核”中可查看；“倾向可补 / 不确定 / 倾向驳回”仅保存在当前浏览器会话。格式、安全门禁与验收证据见 [Round752 记录](docs/PALLET_BOUNDARY_COMPONENT_WORKBENCH_ROUND752_20261123.md)。

团队内部收到已授权的 LoRA ZIP 时，可用 `tools/verify_qwen3vl_controlled_bundle.py` 独立核对
整包摘要、逐文件哈希、基座身份和全部 LoRA 张量；公开仓库仍不分发权重。详见
[受控 LoRA 复现包记录](docs/QWEN3VL2B_CONTROLLED_BUNDLE_ROUND497_20261122.md)。

本机 8GB 显卡可通过按需进程轮换运行 Qwen3-VL-2B NF4 与 SAM3，工作台保持在线。两帧真实功能验收、候选导入及原始数据保护已经完成；本轮没有实际修改候选，不代表准确率提升。见 [本机运行说明](docs/local-gpu.md)、[结果与局限](docs/test-results/local_gpu_results.md) 和 [当前进度](docs/progress/dataset-comparison.md)。

仓库已经包含 `source/sam3` 的 SAM 3 源码，克隆后无需再初始化子模块。安装依赖需要 Python 3.12 和支持 CUDA 的 NVIDIA 显卡：

```bash
git clone https://github.com/linweichen-ai/LabelAgent.git
cd LabelAgent
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e ./source/sam3
```

SAM 3 权重不在仓库中。按[上游说明](https://github.com/facebookresearch/sam3#installation)申请 Hugging Face 访问权限，将 `sam3.pt` 放到 `source/sam3/weights/`。已有本地权重也可直接放在该路径。模型权重因体积和许可原因不随仓库发布。

在两个终端分别启动独立的 SAM 3 推理服务（默认端口 `8766`）和标注工作台：

```bash
.venv/bin/python -m source.sam3_service.server
SAM3_URL=http://127.0.0.1:8766 .venv/bin/python -m source.agent.server
```

SAM 3 服务按首次请求加载 `source/sam3/weights/sam3.pt`，提供 `/health`、`/segment_points`、`/segment_text` 和可选的 `/segment_text_multi`。点提示供参考帧交互标注使用；文本提示供自动流程生成候选。多个类别默认优先使用多提示接口，在服务端只编码一次图像；旧服务不支持时自动退回逐类请求，不会丢弃任务。可通过 `SAM3_HOST`、`SAM3_PORT` 和工作台的 `SAM3_URL` 指向其他主机；工作台本身通过 HTTP 调用模型服务。

双 4090 部署使用的可复现 FastAPI 入口为 `source/sam3_service/remote_fastapi_server.py`，并依赖同目录的 `sam_multi_helpers.py` 与 `response_cache.py`。该入口额外提供 `/prefetch_text_multi` 和 `/prefetch_text`：只缓存完整图片字节与完整提示完全一致的 JSON 响应，默认最多 24 项、128 MiB，缓存位于 CPU，不保存 GPU 特征。相同在途请求会合并；后台预取遇到不相关模型任务时返回 HTTP 202，不排在人工前台请求之前。工作台仍会执行斜坡几何过滤和 ROI 回退，缓存命中不会自动写候选、人工真值或训练批准。

Windows 端通过已配置的 `factory-4090` / `factory-5090` SSH 别名连接远端服务时，可使用 `tools/manage_remote_workbench.py start --no-browser`、`status` 和 `stop` 管理页面与安全通道。4090 是主路径；5090 只是小图可选加速，默认最多探测 3 秒，失败后立即继续使用 4090。确需等待正在启动的 5090 服务时，可设置 `LABELAGENT_FAST_ROUTE_WAIT_SECONDS`，有效范围为 1～20 秒。该管理器只停止自己记录并验证过的本地页面或隧道进程，不会停止远端模型或其他用户的 GPU 任务。

后端使用 Flask，前端静态资源位于 `source/web`，Python 依赖共用 `autolabel/.venv`。浏览器打开 `http://127.0.0.1:8765`，点击顶部文件夹图标，通过绝对路径或文件选择添加 MCAP，再点击文件行的「加载」。点击顶部齿轮图标配置 OpenAI 兼容的视觉大模型 API。页面下方的操作台随人工或自动标注模式切换，可拖动操作台上边缘的分隔条调整高度，两种模式分别保留本次页面内调整后的高度。点击图像依次添加多边形顶点，闭合后保存真值。自动模式查看参考帧时，在「参考帧准备」中生成真值描述；查看待处理帧时，「自动标注过程」顶部工具栏提供开始自动标注、保存结果、清除结果、下载记录及返回参考帧准备的入口。SAM 3 服务生成候选后，编辑 agent 校验 GPT 操作并调用 OpenCV 处理。候选可在主画面复核，通过「保存结果」保存为真值。

当前标注页面支持多边形、矩形、椭圆、自由线/套索与点选 SAM。多边形可按 Enter 或双击闭合；自由线可选“精细、平衡、快速”三档，默认精细档优先保留边界细节。SAM 支持正负提示点，生成期间继续加点会合并为最新点集并只补算一次。页面还支持类别、逐帧播放和保存；MCAP 支持相机话题中的 JPEG/PNG 或 H.264 图像消息。点选提示通过独立 SAM 3 服务生成候选轮廓。编辑 agent 仅接受受控的删除、改类别、补区域、擦区域、合并和拆分操作；坐标及 shape ID 无效时记录工具错误。

SAM 3 上游源码已完整放入 `source/sam3`；本项目的 HTTP 服务位于 `source/sam3_service`，模型权重保存在本地 `source/sam3/weights/sam3.pt`。参考帧点选分割已在 RTX 4090 上通过实际图像验证。无需 CVAT 或 Docker。

本原型尚未完成完整的全景分割约束处理（`stuff/thing`、像素唯一归属、跨帧实例 ID）；必须在实际数据和服务上验证后才能作为全景真值生产流水线。

播放使用 `mcap` 读取消息、PyAV 将 H.264 按采集时间戳封装为 MP4，再由浏览器原生视频播放器按 1× 播放，支持不均匀帧间隔。首次点击播放会准备缓存（`work/playback`），后续直接使用；JPEG/PNG 话题会转码成 H.264。无采集时间戳时使用 MCAP 的 `log_time`，时间戳不递增会报告错误。暂停、拖动和逐帧操作会加载对应的原始分辨率图像，再进行人工标注。播放时显示采集平均帧率。

页面支持 0.1×、0.5×、1×、1.5×、2× 倍速。加载 MCAP 后画面区保持空白；点击「添加窗口」打开对话框，选择图像或点云窗口，再勾选一个或多个同类型消息源；图像源按时间戳同步并透明叠加，点云源合并显示（多源点云需处于相同坐标系）。所有窗口在主视区自动宫格平铺。双击窗口或点击右上角放大按钮可放大/还原，Esc 可还原，× 关闭窗口。图像窗口右上角的 ✎ 按钮可进入标注模式，再次点击可退出；图像窗口的 ✎ 会在原窗口位置打开该窗口第一个图像源的标注视图，退出后恢复原有叠加画面；点击「多边形」进入绘制，再点击「闭合」，绘制期间用窗口右上角按钮放大。

所有窗口按消息时间戳对齐，共用倍速和底部唯一的播放时间轴，显示当前时间与总时长；底部控制栏整合播放/暂停、停止、帧号、倍速和时间轴，左右方向键用于逐帧切换，用于播放与逐帧定位。点云当前支持 protobuf Foxglove PointCloud 的字段与姿态解析，以 100 ms 时间窗聚合，最多显示 5 万点，显示约 10 次/秒；按住鼠标拖动旋转、滚轮缩放。高倍速时会跳过部分预览更新，原始 MCAP 不变。添加窗口对话框只列出当前类型可显示的消息。MCAP 需要 summary 索引。连续图像已验证 JPEG 不等间隔时间戳；原始图像支持 rgb8、bgr8、rgba8、bgra8、mono8。当前解码器接入 protobuf，ROS 点云字节字段解析已提供，但 ROS2 消息解码入口尚未接入。

「文件管理」支持添加多个 MCAP、逐项加载和删除，浏览器选择文件支持多选。列表保存在 `work/files.json`，上传副本位于 `work/uploads`。删除仅移出列表，保留磁盘文件；删除当前文件会卸载画面。切换文件时，各文件的真值与自动候选分别保存于 `work/state.json`。同一路径重复添加会合并。

参考帧可以点击「点选目标」，在目标内添加绿色正点，点击「排除点」后在背景或邻近目标上添加红色负点，再点击「生成轮廓」。候选轮廓会作为本帧多边形加入列表，需人工检查后点击「保存当前帧真值」；「↶」可撤销最近一个提示点。点选推理调用本项目服务端的 SAM 3 交互式图像预测器。当前保存格式是单个外轮廓多边形，复杂目标的多连通区域或孔洞尚不能无损保留；后续需要改为像素 mask 存储。

页面顶部可手动切换「人工标注」和「自动标注」。人工模式的操作台提供手工多边形、补标、擦除和真值保存；自动模式在同一操作台显示过程视图。参考帧准备包含两个阶段：输入 GPT、GPT 描述与关键词。待处理帧的过程视图包含四个阶段：SAM 3 分割、GPT 复查输入、GPT 决策输出、Agent 标注结果。描述与关键词在参考帧准备阶段生成，后续帧复用；描述缺失或失效时，开始自动标注会提示先返回参考帧生成描述，不会在任务中自动生成。平时跟随主画面当前帧，自动标注运行时跟随正在处理的帧；可逐轮查看实际 Prompt 和工具执行结果，点击过程图放大。最终候选仍需在主画面复核并保存真值。

新任务的 GPT 实际输入图像、SAM 3 分割叠加图及 Agent 各步结果图保存在 `work/inspection`；文字记录保存在 `work/last_run.json`、`work/model_debug.json` 和 `work/description_debug.json`。过程面板按当前文件、相机和所选帧筛选记录，描述生成记录仅在参考帧准备中展示，逐帧过程不再混入该记录。旧任务未记录的中间图像或 Prompt 会明确显示缺失提示。当前仅保留最近一次自动任务和最近一次描述生成的过程记录，图像文件按内容指纹复用。

生成描述时，GPT 固定接收参考帧真值原则：参考真值优先、先比对原图和真值再判断、包含和排除须有标注证据、限定类别但不擅自扣除区域、谨慎解释未标注区域、提炼可迁移规则、输出前检查一致性，以及分别输出完整描述和简短 SAM 3 关键词。参考真值可以来自人工或自动标注，经人工确认后作为依据。参考帧中纳入地面的同类内容（如黄色标线）在其他帧继续纳入，明确排除的物体（如机器人）继续排除；参考帧未出现的独立物体默认不纳入地面，有已配置的明确类别时按该类别处理。每轮 GPT 复查同时接收待处理帧原图、候选叠加图、原始参考帧、参考真值轮廓图及参考真值几何数据；文字描述与参考证据冲突时以参考真值为准。提示词版本更新后，旧描述不再复用，需在参考帧准备中重新生成后才能开始自动标注；已有标注和历史记录保留。模型仍可能判断失误，保存真值前可检查候选。

自动标注开始时，SAM 3 按类别关键词生成初始候选。此后每轮 GPT 复查可以针对已有候选使用 `sam_points` 逐步提供当前帧正点和负点，也可以使用受控 OpenCV 工具修改标注 mask；禁止使用 `sam_text` 对整幅图像重新分割或用空 ID 创建新候选。SAM 3 点提示结果要由下一轮 GPT 比较是否改进，未改进时再通过 OpenCV 做局部补区域或擦区域。修复提示词要求保留正确区域、依据参考真值证据定位局部缺陷、使用当前帧原始像素坐标、限制修改范围，并在下一轮检查修复及退化情况；无法可靠判断或定位时保持候选不变并结束，不输出人工复核请求。界面允许设置 1～12 轮，默认上限为 6 轮；Agent 在完成、无有效修改或检测到循环时会提前停止。自动标注按钮显示当前轮数和耗时，运行中再次点击可请求停止。工具修改的是标注 mask，原始图像用于观察。局部修复的语义质量仍需在实际数据上验证。

GPT 每轮复查输出会先由 Agent 校验状态、操作名、候选 ID、操作必需字段、坐标和缺口判断。协议不合规时，Agent 将具体校验错误和上一份输出附加到原请求，最多要求 GPT 修正 2 次；修正请求复用该复查轮的图像，不执行此前不合规的操作，也不增加复查轮数。每次响应分别记录在自动标注过程的 GPT 输入与输出阶段，可看到校验错误、反馈内容和重试次数。重试仍失败时，该帧停止在最近的有效候选，并将校验失败原因留在流程记录中；网络或服务调用故障不进行协议修正重试。

自动模式的过程面板提供「清除结果」：按主画面当前帧清除自动候选、Agent 复查记录、模型调用、运行事件、过程图和页面上的运行文本。参考帧描述作为共享输入保留，但已清除帧的过程阶段不再显示它；其他帧的记录与已保存的人工真值也保留。任务运行中不可清除。清除后可从该帧重新运行自动标注。

## 本地采集数据集导入与测试

四路相机、二进制 PCD 和三维标注采集包的转换、SAM3 全量基线及已有三维框对比，见 [导入与复现说明](docs/capture-comparison.md)。工作台也支持 JPG、JPEG、PNG 图片导入。

新图片批次可先用 `tools/run_new_dataset_smoke.py` 做隔离验收。它从现有 `state.json` 只读复用模型连接和标签表，在临时工作区完成批量上传、首帧检查、VLM 可见类别识别、SAM/Agent 候选及结果叠加图；临时工作区结束后删除，结果始终保持机器候选，不写入正式文件库、人工真值或训练集。示例：

```bash
python tools/run_new_dataset_smoke.py image-a.png image-b.png \
  --source-state /path/to/work/state.json \
  --output-dir /path/to/smoke-output --max-rounds 2
```

本轮数据和测试进度见 [进度记录](docs/progress/dataset-comparison.md)。原始数据、逐对象标注和权重保留在本地。

本机约束解码的后续优化见 [实测对照](docs/test-results/protocol-optimization.md)：相同参考输入的规范生成中位数从约 181 秒降至 16 秒，两帧完整功能重测约 100 秒。97 项回归通过；像素准确率和自然错误的多轮修正效果仍待验证。

[重复候选隔离实验](docs/test-results/duplicate-recovery.md)记录已知重复错误的真实多轮修正、空白负对照及扩大测试中的失败。实验默认关闭，当前工作台候选保留；108 项回归通过，实际像素准确率仍未评估。

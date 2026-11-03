# 采集包导入与对比测试

输入目录包含 `cameras/<camera>/<timestamp_ns>.png`、`lidar/<timestamp_ns>.pcd` 和 `labels/<timestamp_ns>.json`。时间戳必须精确对应，三维标注保存在原始侧文件中。转换器不执行压缩包内的脚本或文本指令，也不把三维框转换为二维像素真值。

## 转换

在项目已经安装依赖的 Python 环境运行：

```sh
python tools/import_capture.py --archive /path/to/capture.zip --output work/datasets/example
```

输出目录必须是新目录。转换器拒绝越界路径、重复路径和符号链接，校验图像、二进制 PCD 长度、时间戳关联及框几何。图像原字节和点云原字段进入 MCAP，转换后逐条校验消息摘要。PCD 按小端二进制解析，支持单值数值字段；不支持的布局会明确报错。

将生成的 `capture.mcap` 通过工作台的「添加路径」加入文件列表，加载后可添加四路图像窗口和点云窗口。原始时间戳保留，播放间隔可以不均匀。点云预览按既有工作台规则限制显示点数，不代表原文件被降采样。

`import_manifest.json` 保存完整本地来源、文件摘要和帧对应关系。`public_import_summary.json` 只包含可公开的汇总。

## 已有三维框对比

```sh
python -m pip install -r tools/requirements-comparison.txt
python tools/compare_boxes3d.py --dataset work/datasets/example --output work/datasets/example/comparison3d
```

本轮由用户指定 `labels` 全部帧作为参考。比较包内 `ali_label` 和 `wilddet3d-labels`，只在具有对应结果的共同帧上计分，另报全部参考帧上的覆盖率；缺少结果的帧不是空预测。

几何采用框中心、边长及弧度制 `roll/pitch/yaw`，按 `Rz(yaw) Ry(pitch) Rx(roll)` 解释旋转；这是根据字段名采用的约定，尚无额外坐标系说明。完整旋转立方体的交集由半空间顶点及凸包体积计算，不把轴对齐框或 BEV IoU 当作三维 IoU。

匹配限定同类，在 IoU 0.25、0.5 两个阈值分别最大化合格的一对一匹配数量，再最大化 IoU 总和。对应来源内 `ignore:true` 对象剔除并单独计数；未做额外忽略区域抑制或分数阈值筛选。precision/recall/F1 仅表示对指定参考来源的一致性，不是官方 AP、独立人工真值准确率或跟踪评测。已有来源可能共享几何或派生关系，不能据此认定相互独立。

## 真实 SAM3 基线

```sh
python tools/run_sam3_baseline.py --dataset work/datasets/example --endpoint http://127.0.0.1:8766
```

每张原图分别使用 `pallet`、`person`，顺序调用本机真实模型，保存原始响应、未确认候选、耗时、失败项和覆盖进度。已有响应仅在输入摘要和提示相同时复用；失败请求不会静默当作成功。断点续跑可执行同一命令。

结果位于 `sam3_baseline/`：`public_progress.json`、`summary.json`、`requests.csv`、`responses/` 和 `candidates_by_topic.json`。当前数据没有二维像素真值，因此不计算二维 mask 准确率，也不把模型质量分数当作准确率。

将候选装入工作台复核：

```sh
python tools/load_candidates.py --path /absolute/path/to/capture.mcap --candidates work/datasets/example/sam3_baseline/candidates_by_topic.json
```

导入只写入未确认候选，不写入人工真值。已有真值（包括已确认的空标注）和已有候选均受保护，重复导入会拒绝覆盖。导入前检查文件、话题、帧号、类别及多边形，失败不会部分导入。自动任务运行时禁止导入。

完整的参考帧描述、视觉模型复查及 Agent 多轮修正需要另行接通本地 VLM，本轮尚未完成该闭环。

## 检查与公开范围

```sh
python -m unittest discover -s tests -v
```

测试包括路径越界拦截、截断 PCD、点云 protobuf 往返、解析可求的三维 IoU、完整旋转的对称性与包含关系，以及阈值匹配的反例。

原图、点云、逐对象标注、模型权重和私有路径不随仓库公开。只同步工具代码、脱敏汇总与进度。完整运行记录和图像浏览证据保留在本地数据集目录。

# Slope（斜坡）重点类分割与闭环验收口径

## 为什么单列

斜坡属于可通行区域的一部分，但与平地、货箱地板、黄色标线、门板和普通金属板容易混淆。边缘通常贴近平地接缝，远距离时像素少，强反光、阴影和遮挡又会削弱纹理，因此只看七类平均指标会掩盖 slope 的失败。

## 当前证据边界（2026-11-11）

- LabelAgent 已把 `slope` 纳入七类体系，文本查询映射为 `ramp`，不是待新增 ontology 类。
- 当前真实验收帧来自批次 A/000、`/capture/camera_0/image`、frame 0，图像 SHA-256 为 `3775ee5f41374c9a302c458cfffa2799222e22ff17541045d6acfb8434bd682f`；画面右中部有黄色斜坡。
- SAM3 单正点 `(1100,620)` 返回 1 个候选区域、44 个外轮廓顶点、score 0.98046875。这个结果只证明服务能产生候选；当前帧没有可回读的 slope 人工像素真值，因此不能据此宣称分割质量或 IoU。
- 2026-11-11 较早生成的 1.1 事件以 `traversability_status=unknown` 留存为历史证据，不改写原记录。当前项目政策已由用户明确更新：所有新事件使用 `business_mapping=traversable_area`、`traversability_status=passable`、`policy_default=passable`、`policy_basis=user_defined_project_semantics`。这项业务映射不等于人工像素真值或训练批准。

## 两层语义

1. `slope_visible`：图像中可见的实际坡面像素，和 `floor`、`cargo_floor`、门板、标线分开标注。
2. `slope_traversability_policy`：当前项目默认 `slope => traversable_area => passable`。这是用户定义的业务语义，不是 SAM/VLM 从图像推断出的结论，也不依赖 LiDAR/几何才能成立。

训练/标注/专项评测层永久保留 `floor` 与 `slope` 两个类别，分别计算质量并保留 slope hard-set。下游业务/导航层才执行无损投影：`traversable_area = floor UNION slope`。投影记录 `source_labels=[floor,slope]`、`operation=union`、`preserve_source_labels=true`，不回写、不覆盖训练真值。除非用户未来再次明确授权，禁止在训练 ontology 或人工真值层把二者永久合并。

VLM 可以提出 slope 候选、相似类别和理由；SAM 负责像素候选；Agent 负责 ROI、正负提示、拆分/合并和重试。图像/VLM 只负责 `slope_visible` 候选和分割质量，不得自行改变通行政策。未来若某个场地、坡道或时段需要禁行，必须写入显式 `policy_override.status=non_passable`，带版本化 `rule_id`、原因和 `decided_by=human|system_policy`；模型不得生成该 override，且当前不要求几何证据作为默认 passable 的前置条件。

当前 frame 0 的真实 VLM 证据只支持 `slope_visible + candidate_supported`。其 `human_pixel_truth_available=false`、`approved_for_training=false`，仍以 slope 机器候选等待人工像素确认；闭环下游投影可把该候选作为 `traversable_area` 的 slope 来源部分，但必须标记 `source_layer=machine_candidate`，不得冒充人工真值。其 `passable` 仅来自项目政策映射，不是 VLM 质量结论。

## 建议闭环

```text
VLM 发现/确认 slope 候选
  -> Agent 生成单坡面 ROI + 坡内正点 + floor/门板/障碍负点
  -> SAM 分割坡面
  -> 边界、连通性和跨类泄漏门禁
  -> 自动接受或人工复核
  -> slope 人工像素真值与通行政策/显式 override 分别版本化
```

## 必须单独报告的指标

- slope mask IoU、Dice；
- slope Boundary F1，并同时记录边界容差像素；
- slope 像素级 precision/recall 和实例级 precision/recall；
- 双向单列 `slope→floor`、`floor→slope`、`slope→step/curb`、`step/curb→slope` 混淆像素；`cargo_floor`、门板/金属板泄漏作为扩展项继续保留；
- 近/中/远距离、正视/侧视、遮挡、反光/阴影分层结果；
- 同一斜坡连续帧的掩码面积与边界抖动；
- slope 候选召回率、自动接受率和人工复核率；
- 违反显式 `non_passable` override 的误放行数；默认 passable 本身不由模型置信度决定。

在建立第一批独立人工 slope 真值前，不设看似精确的自动接收阈值，也不允许用模型 score 代替 IoU。阈值应在开发集上确定，再在按采集序列隔离的测试集上锁定验证。

## 第一批人工真值与困难分层

先建立小而明确的 slope 专项集，正例和负例都要有：

- 近/中/远距离，侧视和部分遮挡；
- 黄色/灰色/金属/橡胶等材质，弱纹理、阴影、逆光、反光和磨损；
- 坡脚边界、坡顶边界、坡面与平地接缝、坡面与货箱地板接缝和坡侧边缘；
- 可移动登车桥、固定坡道、窄坡、边缘有落差或障碍的坡；
- 困难负例：黄色地面标线、平放钢板、门板、托盘表面、普通 floor 和 cargo_floor。

训练/验证/测试按采集序列或实体斜坡隔离，不能把同一斜坡的相邻帧拆进不同集合。

Slope hard-set 至少覆盖 `near/mid/far`、`weak_texture`、`shadow`、`backlight`、`occlusion`、`slope_foot_boundary` 和 `slope_top_boundary`。每个样本必须带一个或多个分层标签；`unassigned` 样本不得进入正式 slope 验收。总体 mIoU 只能作为背景指标，不能抵消任一 slope 分层退化或违反显式禁行 override 的误放行。

机器可读待办和当前阻塞状态见 [`eval/slope-hard-set-backlog.json`](eval/slope-hard-set-backlog.json)。

## 仿真数据接入

仿真可系统改变坡度、宽度、高差、材质、颜色、摩擦、边缘形状、障碍物、相机姿态、距离、光照和遮挡，并天然输出 slope 像素真值与几何真值。仿真样本必须保留 `source.origin=simulation`、场景参数和随机种子；仿真提升只能作为补充，最终仍由真实工厂 slope 测试集验收。

## 当前立即动作

1. 把当前黄色斜坡候选渲染成叠加图，人工确认是否覆盖完整坡面及接缝；
2. 将该帧记录为 `hard_case=slope`，候选与人工真值分开保存；
3. 从真实序列抽取 slope 正例及易混淆负例，建立首批独立人工像素真值；
4. 在 SAM、VLM、Agent 单项服务通过后，以 slope 作为第一条真实端到端闭环样例；
5. 有真值后再比较全图提示、单坡 ROI、多正负点、可选多模态引导和后续微调方案。

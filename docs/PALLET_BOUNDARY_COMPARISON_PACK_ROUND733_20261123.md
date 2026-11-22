# 栈板边界清理候选对比包（Round733）

## 结论

已从正式 8765 工作台当前加载数据中，通过只读 GET 生成一份通用候选对比包。生成过程中没有调用 POST、没有修改工作台候选、人工真值或训练数据，也没有写入 `D:\Apps\LabelAgent\work`。

最终候选包位于：

`artifacts/round733-pallet-boundary-compare-live/`

它包含 33 个 `pallet` 当前候选与离线边界清理替代候选。所有条目仍是 `candidate_only`；它们不是人工真值、不是训练批准结果，也未部署到工作台。

## 输入和只读协议

- 服务：`http://127.0.0.1:8765`
- 请求帧：0、76、151
- 当前相机：camera1 topic（完整 topic 保留在私有 manifest）
- 数据路径字符串 SHA-256：`b10b2bcd75ecc9e53e0b7efa14421c03b77c493e5a4fb4fd8344d5a4118b3196`
- 文件载入代次：`1838435537374152`
- 网络方法：仅 `GET /api/bootstrap`、`GET /api/frame_state/<frame>`、`GET /api/frame/<frame>`
- 每帧在离线计算前后各读取一次 `frame_state`，候选集合或服务端绑定漂移即失败关闭。

生成器参数化 `base_url`、`frames` 和 `output`，没有硬编码当前数据集、帧号、候选 ID 或类别实例。它仅筛选标签为 `pallet` 的当前机器候选，并调用现有纯函数 `refine_pallet_candidate_geometry`；只有栅格 mask 真实发生像素变化时才进入包。

## 结果统计

| 项目 | 数值 |
|---|---:|
| 当前帧中的 pallet 候选 | 34 |
| 产生真实像素变化的替代候选 | 33 |
| 无变化、跳过 | 1（frame 0 / candidate 6） |
| frame 0 / 76 / 151 入包数 | 10 / 9 / 14 |
| 总变化像素 | 2,475 |
| 单实例最大变化像素 | 245 |
| 单实例最大变化比例 | 2.4787% |
| 最终文件数 | 133 |
| 最终占用 | 1,432,332 bytes（约 1.37 MiB） |

变化最大的五项：

| 帧 | 候选 ID | 变化像素 | 变化比例 | 面积净变化 |
|---:|---:|---:|---:|---:|
| 0 | 10 | 245 | 1.3662% | -67 |
| 151 | 3 | 213 | 1.6348% | +107 |
| 0 | 3 | 205 | 1.0689% | -51 |
| 151 | 9 | 198 | 1.1531% | 0 |
| 76 | 4 | 198 | 1.3280% | -4 |

本批次变化主要来自边界简化与局部边缘清理；统计仅说明当前候选与替代候选之间的像素差异，不代表替代候选质量优于当前候选。

## 身份与哈希

- 生成器 SHA-256：`cbead685c276211c8522475adfb25cdc1c78715a1f467f7c6c1334443a9b132a`
- manifest SHA-256：`2c2e6473b8173198aa87672fb0223ef9f344ff5a07d257cde09a8509abf4ed17`
- 规范文件树 SHA-256：`750cf1880019a70eaba5a613e2892166faac0ec13abab92b9c7a0426c2b334a6`

每条记录单独绑定：原始编码帧 SHA、整帧候选 artifact SHA、当前候选 fingerprint、current mask SHA、refined mask SHA、两张预览图 SHA。服务端只读 scope 仍会重新检查当前路径、topic、文件载入代次、帧 SHA 和候选身份；任一漂移返回 409。

## 文件结构

每个候选有四个文件：

1. `*-current-mask.png`：当前候选完整分辨率二值 mask；
2. `*-refined-mask.png`：边界清理替代候选完整分辨率二值 mask；
3. `*-current-preview.png`：当前候选局部 ROI 预览；
4. `*-refined-preview.png`：替代候选相同 ROI 预览。

局部 ROI 仅减少文件大小，不增加新的视觉信息。两张预览使用相同裁剪范围；完整 mask 仍保留原始图像分辨率。

## 测试与边界

定向测试覆盖：

- 所有远端请求均为 GET；
- 任意 base URL、帧和输出目录参数；
- 非 pallet 与无像素变化候选跳过；
- 候选 artifact / fingerprint 和前后读回一致性；
- 两次相同输入输出逐文件一致；
- 重复帧和已有输出失败关闭；
- 非整数或负数候选 ID 在写出任何资源前失败关闭；
- 新包能被通用 v1 schema 读取器完整读回。

本包没有复制到正式 `WORK/candidate_comparison_packs`，所以不会自动出现在已部署工作台；后续只有在单独审阅并明确部署后，才可作为只读对比包载入。

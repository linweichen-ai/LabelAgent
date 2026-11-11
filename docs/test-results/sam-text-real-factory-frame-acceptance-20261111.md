# SAM 真实工厂帧文本提示验收（2026-11-11）

## 结论

双 5090 上的受管 SAM 服务完成了一次真实 1920×1536 工厂帧 `/segment_text` 推理，全部服务级门槛通过。该结果只证明服务进行了真实推理，不代表分割质量合格，更不代表 LabelAgent 闭环完成。

## 输入与请求

- 来源：批次 A/000 的 `batchA_0920_000_clean_review.mcap`，topic `/capture/camera_0/image`，frame 0。
- 采集时间：`2026-09-20T09:45:06.782450Z`。
- 图像：1920×1536，263,785 bytes，导出文件扩展名为 `.png`，实际解码格式为 JPEG/JFIF。
- 输入 SHA-256：`3775ee5f41374c9a302c458cfffa2799222e22ff17541045d6acfb8434bd682f`。
- endpoint：`POST http://127.0.0.1:18766/segment_text`。
- prompt：`shipping container`。

## 验收结果

- 请求前、请求后 `/health` 均为 `status=ok, model_loaded=true`。
- HTTP 200，耗时 0.642 秒。
- 返回 1 个 result、3 个非空 part、66 个外轮廓顶点、0 个洞；score 为 0.80859375。
- 主 part 范围为 `(299,98)–(1182,679)`。另外两个断开 part 只有 5/6 个顶点，范围分别为 `(973,602)–(978,608)`、`(1047,459)–(1052,462)`，约 5×6 像素，明确标记为**疑似伪小块**；全部坐标虽在 1920×1536 范围内，但不能因此认为分割质量良好。
- 精确 HTTP 响应为 983 bytes，SHA-256 `e17f725b5d63524a0fab704debc44ef95edd3f4e8b71f8c416c505c8a7f08b3d`。
- 请求期间服务日志新增 4 行，包含 `/segment_text` 和 `/health` 的 200 回执；新增 traceback 数为 0。
- SAM PID 1131700 绑定 GPU0 UUID `GPU-8fc83bb8-cdd1-8da3-98b7-990f5de229ff`。
- SAM 进程显存由 5,580 MiB 增至 6,218 MiB；GPU0 总已用显存由 5,676 MiB 增至 6,314 MiB。

服务级验收判定：**推理通过；质量未验收**。

## 证据与边界

5090 完整回执目录：`/home/clw/run/labelagent-5090/sam-text-container-flzNz1dm`。目录保留原始输入、精确响应、请求摘要、前后 health、GPU/进程清单、坐标读回、日志增量以及 SHA-256 清单。

响应中有一个覆盖主体的大 part，同时存在两个仅 5～6 个顶点、约 5×6 像素的断开碎片；二者按疑似伪小块留证，未有人工作真值前不作最终真假裁定。没有人工真值 IoU、边界质量、漏分/误分比较，因此 `quality_accepted=false`，不得声称分割质量良好。本验收也不证明 SAM 已微调，不包含 VLM 候选复核、Agent 动作、二次 VLM、人工门禁、真值提交或不可变版本。

可复用的验收脚本为 `tools/smoke_sam_text_http.py`。它会将 HTTP 200 但无结果、空 part、非法点或越界坐标直接判为失败。机器可读回执见 `sam-text-real-factory-frame-acceptance-20261111.json`。

# Round764：斜坡困难帧优先人工复核

## 本轮解决的问题

Round763 已把 frame 95 和 frame 140 识别为最需要人工优先查看的两帧：95 只露出画面边缘的一小段坡面，140 被纸箱大面积遮挡。原工作台虽然已有完整的专项签核、原图哈希校验和追加式审计链，但队列仍按原始帧顺序展示，frame 140 排在很后面。

本轮没有新建第二套签核系统，而是在现有“专项复核”入口增加只读优先级侧车文件。工作台读取后会把尚未完成的优先帧置顶，显示“优先核对”和原因，并提供“优先复核”筛选。点击下一帧时也按优先队列前进。

## 当前实测结果

- Round160 共 32 帧，显式优先帧只有 2 帧。
- 实际队列开头为 `95、140、74、77`；“优先复核”筛选仅显示 `95、140`。
- 优先级侧车 SHA-256：`e91a21a5b7115d1ca9d8dbc963e26124ed33de2dd5fd40907f73ed6cec9aeec2`。
- 侧车绑定 Round160 清单 SHA-256：`1c67cb8dc22b225a7b3b0d149539ec210b7386c0a2a4f1016b7384160092a79c`。
- 侧车绑定 Round763 交叉复核 SHA-256：`d0697eaf57566272c3e5c681da95ebd81c797b145adaf716330de148d5aaa775`。

## 安全边界

优先级只是助手风险排序，不是人工结论。服务端会拒绝清单哈希、来源证据哈希、帧号、排名或候选边界不合法的侧车。所有记录继续固定：

- `assistant_only=true`
- `human_truth=false`
- `approved_for_training=false`
- `auto_submit_allowed=false`

助手建议不会预选人工决定，也不会自动提交。真正签核仍必须由人在工作台“专项复核”入口填写复核人并确认；专项签核本身也不会自动升级为整帧人工真值或进入训练集。

## 验证

- `tests/test_review_packet_core.js`：7/7 通过。
- `tests/test_web_polling.js`：95/95 通过。
- `tests.test_review_packet_api`：7/7 通过。
- Round160 真实清单与助手建议、优先侧车组合读回：优先帧精确为 `95、140`。
- 隔离端口 18769 的真实 Edge 验收通过：32 帧队列首项为 `95、140、74、77`，“优先复核”只显示 `95、140`，页面错误、失败请求、非预期 HTTP 错误和真值写请求均为 0。
- 浏览器回执：`artifacts/round764-slope-priority-browser/result.json`，SHA-256 `dfddf6f353011259efb6532925cfa80cef00ad3facdfcdaff761cc2100d1109f`。
- 截图：`artifacts/round764-slope-priority-browser/review-packet-priority-queue.png`，SHA-256 `181ac557eff7c3ba8fa7d88048dde72483fb0fc50a7cf3b39f60d0ccc3db54fb`。
- `git diff --check` 通过。

## 当前状态与下一步

本轮代码和优先侧车仍在开发工作树中，尚未部署到正式 `D:\Apps\LabelAgent`，也没有产生人工签核。下一步应补充深色和窄屏显示检查，并验证一帧签核后另一帧仍保持优先；通过后再决定是否受控部署。

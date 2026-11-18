# 工作台启动恢复就绪门禁：Round 213

## 一句话结论

工作台现在会把“基础数据已加载”和“上次工作帧已恢复”明确区分开：页面恢复期间显示“正在恢复上次工作帧”，SAM 点选按钮保持禁用，任何提前提交都会在浏览器内被拒绝且不会发送模型请求；只有最终候选帧、数据集和相机稳定后，页面才发出统一的 `workspace-ready` 信号并开放点选。

## 为什么要做这一轮

Round 212 的真实浏览器验收发现，页面启动时可能先出现第 0 帧，随后恢复到上次候选第 74 帧。旧代码把基础数据载入完成当成了整个工作台完成，之后再用一个异步定时器恢复上次运行结果。用户如果恰好在两者之间操作，会看到帧号跳转；虽然 Round 212 已能安全丢弃旧结果，但体验仍像页面“自己变了”，并且会浪费一次已经发出的 SAM 请求。

这一轮不取消服务端正在运行的 SAM。服务端 GPU 请求一旦发出，浏览器中止等待并不能停止后台计算，反而可能让用户马上提交第二次请求并遇到 busy。因此修复点放在请求发出之前：只有工作台完成最终恢复，才允许提交。

## 修复方式

### 1. 单一工作台就绪信号

新增 `window.labelAgentWorkspaceReady`：

1. 先等待原有 `window.labelAgentBootstrapReady` 完成数据、相机和基础页面载入；
2. 进入显式 `restoring` 状态；
3. 合并读取当前任务状态，若存在未展示的上次结果，完整等待 `recoverRunResult()` 和 `openCandidateFrame()`；
4. 最终帧稳定后切换为 `ready`，发布 `labelagent:workspace-ready` 事件，并返回精确的文件、相机和帧作用域。

状态读取增加了单飞 Promise。同一时刻若初始轮询和恢复流程都请求 `/api/run/status`，后者会等待同一个请求，不会再创建第二条并发状态读取。

### 2. 用户可见的恢复状态

恢复期间数据集栏显示：

`正在恢复上次工作帧，请稍候 · <文件> · <相机> · <帧数> · <图像路数>`

SAM 点选按钮同时显示“正在恢复工作帧…”并禁用。恢复完成后，数据集栏回到正常身份信息，点选按钮恢复为“SAM 点选（点目标即运行）”。

### 3. 请求前失败关闭

`generatePointMask()` 第一项检查就是 `workspaceRestoreState === 'ready'`。如果页面仍在恢复：

- 不读取或提交提示点；
- 不调用 `/api/segment_points`；
- 不占用 GPU；
- 明确提示“正在恢复上次工作帧，完成后才能运行 SAM”。

如果状态服务短暂不可用，工作台继续保持恢复状态，沿用既有低频自动重连；不会为了尽快显示而错误开放点选。

## 自动回归

| 项目 | 结果 |
|---|---:|
| 点提示与启动恢复定向网页测试 | 48/48 通过 |
| 全部网页测试 | 114/114 通过 |
| 全部 Python 测试 | 979/979 通过，6.511 秒 |
| JavaScript 语法 | 通过 |
| Python 编译 | 通过（仅上游 SAM3 已知 SyntaxWarning） |
| `git diff --check` | 通过 |

新增测试覆盖：恢复期间点提示产生 0 个 API 调用；状态读取单飞；统一就绪 Promise 必须等待结果恢复；旧的 `setTimeout(refreshAgentProgress, 0)` 路径已移除；浏览器资源版本提升到 `/app.js?v=168`。

## 真实 Edge 验收

验收使用全新 Edge 无头上下文，并在浏览器层暂缓 `/api/run/status`，使恢复窗口可稳定复现。没有拦截或调用真实 SAM，也没有保存人工真值。

| 时刻 | 页面状态 | 帧 | 点选按钮 | SAM 请求 |
|---|---|---:|---|---:|
| 基础载入完成、状态请求尚未释放 | `restoring` | 0 | 禁用，“正在恢复工作帧…” | 0 |
| 状态恢复与历史结果打开完成 | `ready` | 74 | 可用，“SAM 点选（点目标即运行）” | 0 |
| 再观察 500 ms | `ready` | 74 | 可用 | 0 |

具体读回：

- 数据文件：`batchA_0920_000_clean_review.mcap`；
- 相机：`/capture/camera_1/image`；
- 总帧数：152；图像源：4 路；
- 恢复期间强行调用 `generatePointMask()` 后，提示正确，`/api/segment_points` 请求数仍为 0；
- `workspace-ready` 返回的文件、相机、帧与最终页面完全一致；
- 控制台错误 0、页面异常 0、人工真值写请求 0；
- 最终页面资源为 `/app.js?v=168`。

本地验收截图保留在未跟踪目录 `artifacts/round213-workspace-ready-browser/workspace-ready.png`，其中含厂内画面，不上传 GitHub/CNB。

## 部署、健康与回滚

部署前备份：

`D:\Apps\LabelAgent\.local\backups\workspace-ready-round213-20261119-043126`

备份 SHA-256：

- `app.js`：`21343596b8e4b139451a1de599bdea33e6805452bdd3d69217d92681e88d38ac`；
- `index.html`：`225bbf08fad35d51ca6f30af4b1e78ff06c5153194ba31cae9332bb22e23c029`；
- `state.json`：`1dcd67a8475a8dd1974c9a9b3d4bf68a56e1abd30cf2a9559c12e849e3d86f96`；
- `file_sessions.json`：`202927e295cabc27b1f9583b6438d3e16c020809ee421bce1342c3aebc0367f5`；
- `remote_processes.json`：`5bc7053a1322a0f41e84797e5a871f406c9edc597fc539b30273c49e7fe29223`。

正式文件与工作树哈希一致：

- `app.js`：`8dda1637153b8a4140604c928c412d1c534d87198ba56fcfff564b841fda4098`；
- `index.html`：`4b4bca981521ead56e68b51bcd274591c81d8414eafb50c9c4bc30d87efe288c`。

只重启本机工作台：管理进程 PID `69944 -> 61156`，实际监听子进程 PID `35980 -> 52168`。4090 通道 PID `60456` 和 5090 SAM 通道 PID `8656` 均保持不变。工作台状态、运行状态、4090 SAM、5090 SAM 和 4090 VLM 五项探针均为 HTTP 200。

正式 `state.json` 与 `file_sessions.json` 的 SHA-256 前后保持不变。`remote_processes.json` 因工作台 PID 合法更新而变化，不属于标注或真值变化。

## 边界

- 本轮没有修改 SAM/VLM 权重、微调或后训练；
- 没有产生新的机器分割结果；
- 没有写人工真值、训练批准或 OSS；
- 第 74 帧显示的是既有机器候选，仍需人工复核；
- 本轮只解决启动恢复时序、可观察性和误触发，不代表模型精度提升。

机器可读证据见 `docs/evidence/workbench-stability-20261117/workspace-restore-readiness-round213.json`。

## 下一步

1. 继续检查数据集、相机和帧切换期间的其他异步按钮，统一复用精确作用域和就绪门禁；
2. 只对“尚未发到服务端”的浏览器等待做安全取消，已经进入 GPU 的请求继续等待并安全丢弃过期结果；
3. 模型效果优化继续使用独立人工真值和冻结评测集，不把本轮稳定性验收当作训练效果。

# Round493：微孔只读复核工作台生产验收

日期：2026-11-22

## 部署范围

部署到 `D:\Apps\LabelAgent` 的代码版本为 `a4faeab26d55435e54d46cd06c78a4732a0920c6`。
仅更新了工作台服务文件和两个新增前端脚本，并安装 Round488 的 24 个只读证据包文件；
没有重启或修改 SAM/VLM 模型服务。

覆盖前的三个运行文件已备份到：
`D:\Apps\LabelAgent\.local\backups\component-review-round492-20261122`。

部署后的五个代码文件 SHA-256：

- `source/agent/server.py`：`b6f260d021ce95eadd33285d00de439d922552307127e8bb527fc77ae45185ee`
- `source/web/index.html`：`f5490d6a6219c0ab2fd87a921d4fdb0ab8cd42e5994c89deca49e42e26c24574`
- `source/web/style.css`：`6a1bd1e9727c21614c2a170e591a7a85d5eb98d295cf68a1ab641471ab5a4339`
- `source/web/component-review-core.js`：`8829f938287ceb2ff970006c34e0b367876561a8700a4240f198de5aadc8fb4e`
- `source/web/component-review.js`：`ec86ec02880d71f7c8b285bd85f2cad56af38d3a1fb3e9231396116334e565b1`

## 线上接口验收

- 复核包：1 个；总项数 19；优先复核 1；延后人工复核 18。
- 聚焦图片：HTTP 200，返回 52,229 字节，带 `Cache-Control: no-store` 与
  `X-Content-Type-Options: nosniff`。
- 对复核包详情发起 POST：HTTP 405，确认没有写接口。
- 工作台任务状态：`idle`。
- 5090 SAM、4090 VLM、5090 快速 VLM：健康接口均为 HTTP 200。

## 真实浏览器验收

- 徽标显示 19，首屏显示 4 张已完成解码的证据卡片。
- 精确找到唯一优先项 `004-camera1-idx5-1793760663542839987-c002`。
- 图片真实显示后按钮才可用；暂存“不确定”后同会话刷新为 `1 / 19`。
- 新浏览器会话为 `0 / 19`，说明草稿没有跨会话或写回服务器。
- 组件复核 POST 请求 0，页面错误 0，控制台错误 0。

机器可读记录：
`artifacts/round493-component-review-production-acceptance-20261122/acceptance.json`

验收截图：
`artifacts/round493-component-review-production-acceptance-20261122/component-review-session-draft.png`

## 结论与边界

微孔聚焦复核现已在当前工作台可用，且候选、人工真值和训练数据边界保持隔离。这里保存的
仍只是当前浏览器会话草稿，不代表人工确认，不会自动改变 mask，也不会进入训练集。

# Round840：语义专家队列授权模板

## 目的

Round839已经把19张开发集困难图固定为8张哨兵和11张后续样本，但仍然不应因为队列存在就默认获得真实图片外发权限。本轮新增一个失败关闭的授权模板生成器，将未授权状态显式写成可审计文件。

## 安全边界

生成器会在产生模板前校验：

- 队列必须恰好是19张，首批8张、后续11张；
- 队列和每张图必须仍然是未授权状态；
- 不得携带图片路径、密钥或密钥字段；
- `candidate_only=true`、`human_truth=false`、`approved_for_training=false`；
- `test24_used=false`，且不得标记为生产部署。

生成的模板与执行队列的SHA-256绑定。当前队列哈希为`2b055827e1f2eba63052e4b59731530df495511ba6dc9148bc3f30e346ce7a48`，授权值固定为`false`，已授权上传数为0。

## 实测

- 授权模板的队列哈希与Round839执行队列完全一致；
- 外部运行器读取该模板时，在图片读取和API请求前以`matching explicit external-upload authorization is required`拒绝执行；
- 定向测试22/22通过；
- 完整Python回归2,044项通过、1项跳过、294个子测试通过；
- 真实图片外发0，外部API调用0，人工真值写入0，训练0，部署0。

## 当前结论

该文件是“未授权模板”，不是用户同意记录，也不能通过手工修改代替新的明确授权。只有取得针对这19张哈希队列的新授权后，才可生成独立授权记录并先执行8张技术哨兵。

结构化证据：`docs/evidence/workbench-stability-20261117/round840-semantic-expert-authorization-template.json`。

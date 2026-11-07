# 本机 Agent 约束解码优化

更新日期：2026-11-04（北京时间）。本轮改进的是实际运行耗时和关键词输出，二维像素准确率仍未评估。

## 同一输入的实测对照

在相同的 Qwen3-VL-2B NF4、两张参考图、原生产提示、1516 个输入 token、每图 256 token 和 768 输出 token 上限下，将自由文本的有限正则改为 JSON 原生字符串长度限制。描述、理由和关键词的长度上限保留，操作名、类别、候选编号和孔洞编号约束保留；关键词继续经过生产流程的短语校验。

先旧后新与先新后旧各测一次，另重复新方案两次。原方案 2 次中位数 **180.837 秒**；新方案 4 次中位数 **15.890 秒**，本组观测降低 **91.21%**。初次模型激活时间另记，不计入这些请求；未控制桌面后台负载，不将小样本计时解释为性能保证。

所有实际输入摘要和逐次耗时见 [对照汇总](protocol_optimization.json)。约束解析器原生字符串路径可以复用自由文本 token 过滤缓存，避免长有限正则的解析及筛选开销；参见 [格式约束组件源码](https://github.com/noamgat/lm-format-enforcer)。增加预处理、约束准备和生成阶段计时，便于后续定位等待。

新方案各次关键词均为 `pallet`、`person`，替代旧方案中的重复词组。两种方案的描述仍达到 700 字符上限，存在句子截断。尝试过的简化英文提示和 JSON 示例没有解决该问题，未安装这些提示修改。

## 两帧完整重测

继续使用第 0 帧未确认功能参考，第 76、151 帧完成真实 SAM3 生成和 VLM 复核。以下帧耗时包含模型轮换：

| 帧 | 前次耗时/秒 | 本次耗时/秒 | 本次候选数 | 实际修改数 |
| --- | ---: | ---: | ---: | ---: |
| 76 | 88.844 | 42.891 | 11 | 0 |
| 151 | 84.016 | 40.375 | 21 | 0 |

包含规范生成的前次运行约 353.876 秒，本次约 **100.037 秒**，均不含初次模型激活。这是不同时间两次完整运行的观察值；完整流程的 SAM3 关键词也已变化，不能仅据此分离各项耗时原因。

本次共 32 个未确认候选，分别为 11、21 个。VLM 各复核一轮并返回 `done`，没有执行修改；不能将候选数量差异归因于 VLM 修正。原工作台会话中的 29 个候选继续保留，本次结果独立保存。逐帧功能汇总见 [本次 Agent 结果](protocol_optimization_agent.json)。

## 验证与剩余问题

- 独立 VLM 环境共 97 项回归通过，包括 4 项使用实际约束解析器的测试。短字符串可提前结束，JSON 转义可通过；长度限制、非法编号、虚构孔洞及待确认 SAM 评估继续被校验。
- 优化代码已加载到工作台；第 76、151 帧显示和刷新恢复通过，无页面错误，未确认候选未保存为真值。[浏览器结果](protocol_optimization_browser.json)
- 1023 个原始文件、原 MCAP 和 6 个既有会话重新校验一致；保存状态文件与加载优化前的 SHA256 完全相同。[保护检查](protocol_optimization_preservation.json)
- 模型描述仍有不可靠的颜色和物体解释，复核理由可能截断或与 `done` 矛盾。功能结束不证明视觉判断正确，也没有实际多轮修正证据。
- 下一阶段在独立候选副本中注入可核验的重复实例，测试错误发现、修正和正确区域保留。该检测指标只针对已知注入错误，仍不作为真实二维像素准确率。

## 复现

在可用的本地 CUDA 模型服务上，使用新的输出目录：

```sh
python tools/benchmark_local_constraints.py --dataset <dataset> --reference-json <reference-json> --reference-kind unconfirmed_functional_fixture --variants legacy_regex,native_length --output <new-output>
python tools/benchmark_local_constraints.py --dataset <dataset> --reference-json <reference-json> --reference-kind unconfirmed_functional_fixture --variants native_length,legacy_regex --output <another-new-output>
```

完整重测继续使用 `tools/run_local_agent_acceptance.py` 的 compact 模式。参考仍为未确认功能示例，原始图像、逐对象轮廓、私有路径和模型权重仅保存在本机。

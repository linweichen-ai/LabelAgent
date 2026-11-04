# 当前采集数据的七类Mask2Former试跑

当前工作台类别列表已包含地面，但参考帧只确认了10个托盘、5个人员。`reference_labels`取类别列表与参考帧已标注类别的交集，`restrict_description`及自动标注入口只接受该范围。因此，本轮SAM3实际请求仅含pallet/person；添加floor类别名称不会自动生成地面参考。

已有Mask2Former权重映射七类：pallet、person、floor、slope、ego、other_robot、cargo_floor。floor是普通地面，cargo_floor是车厢地板，不能混同；这份权重不包含后来新增的“车辆”类。

本次冻结三张当前采集图，比较保留的R5 epoch20（384×512输入）与R11 epoch5候选（448×576，按原规则与R5融合）。先固定模型、输入和后处理再执行；CPU上6次真实推理，无新训练和GPU请求。另一轮较新的训练结果没有自动取代保留版，因此没有直接按最大轮次选择模型。

实际产生了地面和人员分割；目视存在墙面被判为地面、托盘明显漏检等迁移问题。七类模型不代表每张图一定预测出七类，也不代表能识别任意新增类别。本次没有完整二维真值，不报告PQ、mIoU或像素准确率，预测区域占比也不是准确率。

工作台、CVAT均未写入；参考帧及当前主题状态前后摘要一致，模型权重不变。图片及逐像素预测保留在本地交接目录，不上传仓库。对照图与原图供人工复核；原SAM3流程和原标注保持。

## 独立试跑入口

`tools/probe_trained_mask2former.py`提供仅CPU、本地权重推理入口；需要原有torch、transformers、numpy、Pillow环境，不自动下载模型。模型路径、输入图和新输出目录由调用者提供，不接管模型服务，不写标注。

```sh
python tools/probe_trained_mask2former.py --checkpoint <local-checkpoint> --image <local-image> --output-dir <new-private-output>
```

新入口在同图同R5权重上追加1次真实CPU推理，与训练项目现有推理的全景ID图逐像素一致。原类别分数0.5、绝对掩码0.5及重叠0.8阈值保持；未分配像素保留未知。

后续可以评估“Mask2Former生成多类初稿、SAM3细化、Agent复核”的组合。扩大当前SAM3的地面范围则需先人工确认参考帧地面标注，再重新生成参考描述；本次没有自动确认地面或切换默认模型。

[脱敏实测](mask2former-capture-probe.json)

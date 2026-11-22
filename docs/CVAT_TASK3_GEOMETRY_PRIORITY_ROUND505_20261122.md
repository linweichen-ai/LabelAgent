# Round505：Task3 栈板轮廓复核优先级

## 结果

对 Round504 的46个视觉代表项做几何风险分级，仅用于安排复核顺序：

- 高优先级：12项；
- 中优先级：17项；
- 常规复核：17项；
- 自动判错/通过：0；训练授权：否。

触发信号统计：23项接触图像边缘、16项包含多个部件、16项包含多个孔洞、4项轮廓顶点密度较高。分数分布为0分16项、1分1项、2分9项、3分8项、4分3项、5分7项、6分2项。

这些信号不等于标注错误。栈板孔洞、遮挡、被画面边缘截断都可能是合理情况，因此只把高风险项提前给复核者查看，不自动修改轮廓。

## 边界与产物

完整候选ID、指标和排序只保存在：

`.repro/cvat-task3-manual-instance-geometry-priority-round505.json`

- 大小：27,051字节；
- SHA-256：`022b56b08027a584fa6b4d66e95699e6a3807bfcc1fb474252b8de81288a45e6`。

公开汇总回执：

`artifacts/round505-cvat-task3-manual-instance-geometry-priority-20261122/receipt.json`

- SHA-256：`9434f28d1756bcf42d5a38bb8927422c92fd30440d4c0fefb4c00b96e0e3e775`；
- 不含候选ID、轮廓坐标、图片或本机路径；
- `automatic_decisions=false`、`training_authorized=false`。

## 验证与下一步

新增 `tools/audit_cvat_manual_instance_geometry_priority.py` 及2项定向测试：图像边缘截断会提升复核优先级但不会授权训练；多样性审计若绑定了不同候选清单则失败关闭。

下一步按“12个高优先级→17个中优先级→17个常规项”生成分组复核页，并把每项结论绑定候选、几何和聚焦图哈希。助手判断仍只能作为初筛；独立签字和训练授权继续分离。

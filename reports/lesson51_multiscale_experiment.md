# Lesson51：浅层多尺度残差结构实验

目标是在相近 Precision 条件下改善细且低对比度裂纹的覆盖，并检查是否增加伪连接。本轮仅增加多尺度残差模块，注意力机制留作独立后续消融。

## 结构

新增模型见 [multiscale_resnet_unet.py](../models/multiscale_resnet_unet.py)。原 [ResNet18 U-Net](../models/resnet_unet.py) 保留，现有训练、最终评价与 ONNX 脚本不修改。

```text
ResNet18 原始 e0（约1/2分辨率）
       ├──────────────────────────恒等旁路───────────┐
       └─1×1压缩（64→16）─┬─3×3 depthwise,d=1─┐    │
                          └─3×3 depthwise,d=2─┴─concat─1×1融合（32→64）─相加
                                                                     │
                                                       送入原解码器浅层skip
```

- 默认适配 `e0`，可独立配置 `e1`（约1/4分辨率）或两者。
- 各分支保持空间分辨率一致。dilation 1/2 的3×3卷积分别具有3×3/5×5的采样覆盖范围；后者不是完整密集5×5卷积。
- 使用通道压缩和深度卷积控制开销，默认每个适配器新增 **3,472** 个参数。
- 适配结果只进入 decoder skip；后续 encoder 层仍接收原始编码器特征，保留预训练主路径。
- 融合层零初始化，使初始适配器为恒等映射，加载同一基线权重后的初始输出一致。第一步只有融合层能收到非零数据梯度，更新后梯度可以进入前面的分支。
- 不新增 BatchNorm；冻结编码器时保持已有 BN 统计量固定，Layer4 参数和适配器仍可训练。

## 配对训练设计

两组为 `baseline` 与 `multiscale`，均从已有 `outputs/best_resnet18_frozen.pth` 开始，重新执行相同预算的10 epoch增量微调。该设计用于控制结构变量，不是分别从ImageNet重新训练全流程。

| 设置 | 两组共同配置 |
|---|---|
| 数据 | split seed42，240训练/60验证；不读取237测试集 |
| 训练输入 | 256×256随机裁剪，batch size1 |
| 可训练参数 | Layer4、decoder；多尺度组另训练新增adapter |
| 学习率 | Layer4：1e-5；decoder/adapter：1e-4 |
| 损失 | Weighted BCE(pos_weight=5) + 0.5 Dice Loss |
| 其他策略 | 不开启增强、困难采样、clDice或注意力 |
| checkpoint | 按256×256中心裁剪、阈值0.5的最高验证Dice保存 |
| 最终对比 | 完整60张验证图直接前向，扫描0.05～0.95、步长0.01 |

模型构建后重置训练种子，两组使用同样的DataLoader generator。每个epoch记录实际训练tensor内容的SHA256，运行结束会验证两组图像顺序和裁剪内容完全一致。

完整验证同时保存最高Dice点与Precision下限约束点。默认目标为 `P≥0.85`，从可行阈值中选择Recall最高者；若不可达，则保存最接近目标的点并标记 `precision_target_met=False`。达到同一下限不保证Precision完全相等，应同时查看实际Precision及gap。阈值扫描与骨架评价统一按float64严格大于阈值，避免临界值类型转换不一致。

## 运行方法

在项目根目录激活原虚拟环境，或直接使用 `.venv\Scripts\python.exe`。本实验额外需要 `scipy` 和 `scikit-image`；它们与原骨架实验依赖相同。

```powershell
.venv\Scripts\python.exe -m pip install scipy scikit-image
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_multiscale*.py"
.venv\Scripts\python.exe lesson51_multiscale_experiment.py --help
```

先做只用于流程检查的小样本运行，结果明确标为 `subset_smoke_validation`：

```powershell
.venv\Scripts\python.exe lesson51_multiscale_experiment.py --epochs 1 --train-limit 2 --val-limit 2 --output outputs/lesson51/smoke_e0_seed42
```

完整配对实验与扩展种子：

```powershell
.venv\Scripts\python.exe lesson51_multiscale_experiment.py --seed 42
.venv\Scripts\python.exe lesson51_multiscale_experiment.py --seed 43
.venv\Scripts\python.exe lesson51_multiscale_experiment.py --seed 44
```

默认结果目录为 `outputs/lesson51/e0/seed_42` 等，已有训练目录会拒绝覆盖。改变适配位置形成独立对照：

```powershell
.venv\Scripts\python.exe lesson51_multiscale_experiment.py --seed 42 --stages e1
.venv\Scripts\python.exe lesson51_multiscale_experiment.py --seed 42 --stages e0 e1
```

仅重新评价已完成的实验：

```powershell
.venv\Scripts\python.exe lesson51_multiscale_experiment.py --evaluate-only --output outputs/lesson51/e0/seed_42
```

重评使用保存的模型/数据配置，要求全部训练epoch完整、checkpoint哈希及图像/标注内容与训练记录一致。它会更新该次实验的验证CSV与报告，同时分别记录当前评价源码与运行环境；不会再次训练。失败或中断的训练不会被重评伪装成完整实验。

## 输出与诊断定义

每组保存 `best_model.pth`、`training_history.csv` 和 `full_validation_thresholds.csv`。实验目录另有 `experiment_config.json`、`validation_summary.csv`、`summary.md`，包含配置、输入权重/源码/数据哈希、配对数据校验、像素指标、骨架计数、参数量及本机前向延迟。

- 细裂纹：GT骨架上 `2 × distance_transform_edt(GT) ≤3px`。
- 低对比：RGB灰度的21×21邻域均值减当前灰度，截断到非负后≤0.0779；阈值沿用历史诊断定义，不在本轮重新调选。
- 覆盖容差：预测Mask按3×3全1邻域膨胀2次（8邻域2px容差）。无相应GT点时Recall为空，分母另存。
- 漏检候选片段：未覆盖GT骨架中8邻域连通分量大小≥5点的数量，不是人工确认的断点数。
- 伪骨架比例：预测骨架落在GT按3×3全1邻域膨胀3次范围之外的点数比例。
- 本轮在完整图像上重算上述统计，与此前中心裁剪的56.61%等结果不直接混用。

单种子结果只能作为初步筛查。后续应以多种子下的细低骨架覆盖、整体Dice、误检和计算开销共同判断，不能仅凭某一次Dice上涨宣称稳定改善。

所有新权重和本地输出仍位于已忽略的 `outputs/`。公开结果只选取汇总CSV和说明，不纳入原始数据、权重或测试图像。

## 已完成结果

已完成seed42/43/44配对实验，见[三种子结果](lesson51_multiscale_results.md)及原始汇总。当前e0配置没有显示稳定改善，保留为结构消融候选。

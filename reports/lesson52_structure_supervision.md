# Lesson52：以整图细裂纹覆盖为目标的结构监督对照

已有 `outputs/lesson41/worst_validation_examples.png`、`lesson45/thin_low_examples.png` 和 `lesson48/difficult_examples.png` 中，可见细低对比裂纹存在长段漏检。部分漏检区域在旧概率图上仍有弱响应，不能仅凭二值Mask就确认信息已被下采样完全丢失；同样，伪骨架比例本身也不能证明发生了路面纹理误检。

本轮首先比较相同 ResNet18 U-Net 的普通损失与 soft-clDice，重点检验结构监督能否改善上述漏检。编码器下采样保持相同，结果不能直接证明下采样是否为根因。

## 固定设计

| 项目 | 预先固定的设置 |
|---|---|
| 初始化 | 同一已有 `outputs/best_resnet18_finetuned.pth`；这是已完成Layer4微调后的增量对照 |
| 模型 | 原 ResNet18 U-Net，两组架构相同，未叠加多尺度或注意力 |
| 数据 | split seed42固定240/60；不读取237张测试集 |
| 训练 | 256×256随机裁剪，batch size1，每组10epoch，训练种子42/43/44 |
| 优化器 | Adam，Layer4为1e-5，decoder为1e-4，其余编码器冻结，已有BN保持eval |
| 基线 | Weighted BCE(pos_weight5)+0.5Dice Loss |
| 处理组 | 基线损失+0.25soft-clDice，soft-skeleton迭代10次 |
| 其他策略 | 不开启增强、困难采样或新结构；每轮核对两组实际训练tensor哈希 |
| 验证 | 每个epoch使用完整60张验证图，直接原图前向，无裁剪、缩放或滑窗 |
| 阈值 | 0.05～0.95、步长0.001，统一float64严格大于比较 |

阈值点采用 `Precision≥0.85` 时Recall最高的网格点。该下限不意味着两模型Precision完全相同，因此报告实际Precision与gap。

每组在完整训练结束后，先取各epoch中最高的P约束点Dice，再在距该值不超过0.002的epoch中选择**细低对比骨架Recall最高**者；同分时优先更高Dice与更早epoch。该0.002容差只用于各组内部选择，不保证两组间Dice不会下降，最后必须据实际配对结果判断代价。

实现只缓存Dice/细低Recall未被其他epoch同时超越的CPU权重，与保存所有epoch再穷举选择等价，避免保存大量冗余checkpoint。单元测试用随机候选集合核对了这一等价性。

## 运行

在项目根目录使用原虚拟环境：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe lesson52_structure_supervision.py --help
.venv\Scripts\python.exe lesson52_structure_supervision.py --epochs 1 --train-limit 2 --val-limit 2 --output outputs/lesson52/smoke_seed42
.venv\Scripts\python.exe lesson52_structure_supervision.py --seed 42
.venv\Scripts\python.exe lesson52_structure_supervision.py --seed 43
.venv\Scripts\python.exe lesson52_structure_supervision.py --seed 44
```

仅核对来源并重评已完整训练的结果：

```powershell
.venv\Scripts\python.exe lesson52_structure_supervision.py --evaluate-only --output outputs/lesson52/seed_42
```

训练拒绝覆盖已有非空结果目录。重评要求训练记录完整且数据/初始化/选中权重哈希匹配，更新本次验证结果并另外记录评价源码与环境。

## 输出

默认保存在 `outputs/lesson52/seed_<seed>/`，小样本运行明确标记为subset：

- `baseline/`、`cldice/`：训练history、选择说明、选中权重、P约束点Dice最佳权重、901点阈值曲线与完整验证概率。
- `validation_summary.csv`：选中权重分别在最佳Dice与Precision下限点的结果。
- `per_image_validation.csv`：原初始化模型与两组新权重在Precision下限点的逐图数据。
- `diagnostics/`：固定5张验证图的原图、GT、概率、二值Mask与TP/FP/FN叠加，不从测试集挑图。
- `experiment_config.json`、`summary.md`：数据/源码/权重校验、规则、环境与结果。

固定案例为 `IMG11-1.jpg`、`11122-2.jpg`、`11178-3.jpg`、`IMG39-1.jpg`、`11150.jpg`，它们都属于既有验证划分。图像选择在训练前固定，最终定量比较仍覆盖全部60张图。

细低对比骨架定义沿用已有方法：GT skeletonize，局部宽度≤3px，对比度为21×21灰度邻域均值减当前值并截断到非负，低对比阈值0.0779；覆盖用预测3×3膨胀2次计算。分母和实际命中点同时保留。

本轮不将伪骨架比例解释为已确认的路面纹理误检，也不将候选片段数解释为人工标注的断点数。结构监督能否提高覆盖、是否只是变粗或产生额外预测，需要联合指标与固定案例图判断。

## 解释范围

与lesson51相比，本轮改变了初始化阶段、整图epoch选择、细阈值和主选择目标，因此不能跨轮直接归因。主要结论只来自本轮配对基线与soft-clDice之间的差异。三个种子共享同一历史初始化，仍只覆盖增量微调的随机性；没有新的未接触测试集证据。

原脚本、原最终模型、既有报告和ONNX文件均保留。只将数字汇总整理到 `reports/`，训练权重、数据和全部输出不随代码上传。

## 已完成结果

已完成三个种子的配对训练、整图评价与固定案例图，见[结果汇总](lesson52_structure_results.md)。soft-clDice有小幅正向信号，尚未解决全部长段漏检。

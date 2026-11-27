# 最终实验记录与评价协议

本目录保留可公开的数字汇总，不包含数据集、权重、测试集原图或全部预测图。CSV 从已有实验文件按字节复制；`hard_sampling_summary.csv` 为六份训练 history 的确定性提取。模型、阈值及表格内容来自现有文件，本次文档整理未重新训练或运行模型推理。

## 最终模型与评价

- 数据划分：原训练来源 300 张，以 split seed 42 固定划为 240/60；测试来源 237 张。
- 最终选择：`ResNet18_Finetuned`，本地权重 `outputs/best_resnet18_finetuned.pth`，阈值 `0.78`。
- 训练路径：`lesson16_train_resnet_frozen.py` 的预训练编码器冻结训练，再由 `lesson17_finetune_resnet.py` 使用 256×256 随机裁剪进行 Layer4 微调。最终选中模型没有使用整图训练。
- 最终评价：`lesson50_final_evaluation.py` 对完整原图直接 PyTorch 前向，无缩放、裁剪或滑窗。RGB `/255`，模型内部做 ImageNet 归一化；GT 灰度 `>127`，预测概率 `>threshold`。
- 验证时对每个候选扫描 0.05～0.95、步长 0.01，以验证集 Micro Dice 选权重和阈值。测试沿用该选择，不重新搜索阈值。
- Micro 指标由所有像素 TP/FP/FN 汇总后计算；Mean Image Dice/IoU 为逐图指标的算术平均。
- 已复核 [test_per_image.csv](test_per_image.csv) 共 237 个不同文件名，TP=1,600,314、FP=202,763、FN=541,949，逐图与汇总一致。
- 测试来源图像在早期实验曾被查看，此次不属于完全未接触的盲测；模型和阈值共用同一验证集选择。

## 五个候选权重的映射

| CSV 名称 | 本地权重 | 生成入口 |
|---|---|---|
| ResNet18_Frozen | `outputs/best_resnet18_frozen.pth` | `lesson16_train_resnet_frozen.py` |
| ResNet18_Finetuned | `outputs/best_resnet18_finetuned.pth` | `lesson17_finetune_resnet.py` |
| Hard_Sampling_43 | `outputs/lesson47/seed_43/best_hard_sampling.pth` | `lesson47_multi_seed.py --seed 43` |
| clDice_42 | `outputs/lesson48/best_cldice.pth` | `lesson48_cldice_experiment.py` |
| Full_Image | `outputs/final_full_image/best_full_image.pth` | `lesson50_full_image_train.py` |

`Full_Image` 是冻结编码器整图训练对照，不是最终选中的 Layer4 微调模型。其 training history 在 epoch 11、固定阈值 0.73 的最佳完整验证 Dice 为 0.8582769393；统一阈值搜索后为 0.81、Dice 0.8590458603。原自动报告中的“最终训练输入：原始384×544图像”应结合这一区别理解。

## 中心裁剪研究指标的定义

第 42～48 课相关研究结果使用 60 张验证图的 256×256 中心裁剪，不能与最终完整验证/测试集 Dice 合并。骨架 Recall 与像素 Recall 也属于不同指标。

- GT 骨架由 `skimage.morphology.skeletonize` 提取；局部宽度代理为 `2 × distance_transform_edt(GT)`，细裂纹定义为宽度 ≤3 px。
- 对比度代理为 `max(21×21 局部平均灰度 − 当前灰度, 0)`；验证分析中低对比度阈值为 0.0779。
- 2 px 容差采用 3×3 全 1 邻域膨胀预测两次实现，属于 8 邻域容差，不是严格的欧氏距离 2 px。
- `width_contrast_statistics.csv` 来自 lesson42 baseline、阈值 0.73：细低对比骨架 1,127/20,483 点，占 5.50%；未覆盖 489/1,426 点，占 34.29%；覆盖 Recall=0.5661047028。
- lesson43 的 `low_recall` 指低对比度真实裂纹**像素** Recall，不是细低对比度骨架 Recall。固定阈值 0.73 下 baseline/增强的 Precision 为 0.845002/0.809314，Recall 为 0.863641/0.887687；将增强阈值调至 0.84 后 Precision 0.846368、Recall 0.853713，低对比像素 Recall 0.662962，低于 baseline 的 0.682885。

## 困难采样与多种子

[hard_sampling_summary.csv](hard_sampling_summary.csv) 只提取固定阈值 0.73 下各 history 的最佳验证 epoch Dice：

| 训练种子 | 普通随机采样 Dice | 困难采样 Dice | 变化 |
| --- | ---: | ---: | ---: |
| 42 | 0.8494 | 0.8542 | +0.0049 |
| 43 | 0.8457 | 0.8551 | +0.0095 |
| 44 | 0.8439 | 0.8469 | +0.0030 |

三组属于相同的中心裁剪指标，变化基于原始精度计算。seed42 来源 `outputs/lesson46/`，seed43/44 来源 `outputs/lesson47/seed_43/` 与 `seed_44/`。原始 history 路径及最佳 epoch 已写入汇总 CSV，没有统计显著性或置信区间。

[seed 43](hard_sampling_seed_43_summary.csv) 和 [seed 44](hard_sampling_seed_44_summary.csv) 的 `best_dice` 则是扫描阈值后的最优 Dice；`matched_*` 是 Precision≥0.845 条件下 Recall 最高的比较点。`thin_low_recall` 在该比较点计算，分别从 0.6574977817→0.6424134871、0.5785270630→0.4525288376，不能与最优 Dice 的阈值混用。seed42 未保存对应 geometry 汇总，不补造该局部数值。

困难区域候选仅由训练图标注和灰度分析得到，以 0.5 概率引导裁剪。采样内部使用每图对比度 0.67 分位数判定相对低对比区域，与验证分析的分组阈值不同。候选不足时回退至细骨架或全部骨架。

## soft-clDice 的证据边界

[soft_cldice_summary.csv](soft_cldice_summary.csv) 为 split seed42、train seed42 的单次对照，冻结 ImageNet 编码器训练 15 epoch。损失为 `BCE(pos_weight=5) + 0.5 Dice Loss + 0.25 soft-clDice`，soft-skeleton 迭代 10 次；无 GT 裂纹的裁剪块不计算骨架损失。

在 Precision 约 0.846539/0.846781、阈值分别 0.73/0.83 时，细低骨架 Recall 从 0.5989352263 提高到 0.6779059450；`missing_segments` 从 66 降至 58。后者是 GT 骨架未被容差预测覆盖区域中，8 邻域连通分量大小≥5点的候选片段数，不是人工确认的断点或真实路径条数。`far_skeleton_ratio` 同时从 0.0272134303 升至 0.0326194057，故不能将结果表述为所有拓扑指标均改善。该策略尚未完成多随机种子验证。

## 研究实验的运行依赖

以下命令均在项目根目录运行，会训练模型或更新同名本地输出，需要先准备数据和依赖。它们不是仅查看报告的命令。

```bash
python lesson42_contrast_augmentation.py
python lesson43_precision_recall.py
python lesson44_skeleton_analysis.py
python lesson45_width_contrast.py
python lesson46_hard_sampling.py
python lesson47_multi_seed.py --seed 43
python lesson47_multi_seed.py --seed 44
python lesson48_cldice_experiment.py
```

lesson43/44 依赖 lesson42 生成的 `best_baseline.pth` 和 `best_contrast_aug.pth`；lesson45 仅依赖 `best_baseline.pth`。lesson46 与 lesson47 各自完成普通/困难采样训练，lesson48 完成 baseline/soft-clDice 对照。其他结构实验见 README 的真实脚本列表；没有独立的 lesson49 脚本。研究脚本大多在顶层执行，不应将 `--help` 当作无副作用的查看方式。

## 公开汇总的来源

| 公开文件 | 原始本地来源（未纳入 Git） |
| --- | ---: |
| [full_validation_comparison.csv](full_validation_comparison.csv) | `outputs/final_review/full_validation_comparison.csv` |
| [test_summary.csv](test_summary.csv) | `outputs/final_review/test_summary.csv` |
| [test_per_image.csv](test_per_image.csv) | `outputs/final_review/test_per_image.csv` |
| [full_image_training_history.csv](full_image_training_history.csv) | `outputs/final_full_image/training_history.csv` |
| [width_contrast_statistics.csv](width_contrast_statistics.csv) | `outputs/lesson45/joint_statistics.csv` |
| [hard_sampling_seed_43_summary.csv](hard_sampling_seed_43_summary.csv) | `outputs/lesson47/seed_43/seed_summary.csv` |
| [hard_sampling_seed_44_summary.csv](hard_sampling_seed_44_summary.csv) | `outputs/lesson47/seed_44/seed_summary.csv` |
| [soft_cldice_summary.csv](soft_cldice_summary.csv) | `outputs/lesson48/summary.csv` |
| [contrast_baseline_metrics.csv](contrast_baseline_metrics.csv) | `outputs/lesson43/baseline_metrics.csv` |
| [contrast_aug_metrics.csv](contrast_aug_metrics.csv) | `outputs/lesson43/contrast_aug_metrics.csv` |

`hard_sampling_summary.csv` 的六份输入为 `outputs/lesson46/{baseline,hard_sampling}_history.csv`、`outputs/lesson47/seed_43/{baseline,hard_sampling}_history.csv`、`outputs/lesson47/seed_44/{baseline,hard_sampling}_history.csv`。所有来源只读提取，原始本地实验输出没有移动或修改。

旧文件 [model_comparison.csv](model_comparison.csv)、[MiniUNet_test_results.csv](MiniUNet_test_results.csv)、[ResNet18UNet_test_results.csv](ResNet18UNet_test_results.csv) 继续保留，属于 lesson25 的历史测试协议；它们不是本次最终选中模型的结果。历史 ONNX 性能记录来自原 README，导出与 benchmark 均使用 `best_resnet18_frozen.pth`；本次未复测性能。

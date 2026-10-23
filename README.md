# Road Crack Semantic Segmentation

基于PyTorch与ONNX Runtime的道路裂纹语义分割项目。

## 1. 项目简介

本项目基于DeepCrack数据集，实现道路裂纹的像素级语义分割，并构建从数据处理、模型训练、性能评价到ONNX部署的完整计算机视觉工作流。

实现了两种分割模型：

- Mini U-Net：轻量级语义分割模型。
- ResNet18 U-Net：采用ImageNet预训练编码器的语义分割模型。

项目支持对不同尺寸的道路图像进行批量预测，输出裂纹二值Mask、概率图及可视化叠加结果。

## 2. 数据集

使用DeepCrack数据集：

- 训练来源：300张图像。
- 训练集：240张。
- 验证集：60张。
- 独立测试集：237张。

数据预处理包括随机裁剪、中心裁剪、图像与标签同步增强，以及像素值归一化。

原始数据集不包含在代码仓库中。

## 3. 模型测试结果

在237张独立测试图像上进行评价：

| 指标 | Mini U-Net | ResNet18 U-Net |
|---|---:|---:|
| Precision | 0.8510 | 0.8486 |
| Recall | 0.7420 | 0.7560 |
| Micro Dice | 0.7928 | 0.7997 |
| IoU | 0.6567 | 0.6662 |
| Mean Image Dice | 0.7510 | 0.7732 |

模型阈值由验证集确定，测试时未重新优化阈值。

测试结果采用PyTorch整图推理；ONNX滑动窗口部署的分割精度需要单独评价。

## 4. 模型推理性能

测试硬件：NVIDIA GeForce RTX 3050 Laptop GPU。

输入尺寸：256×256，Batch Size为1。

| 指标 | Mini U-Net | ResNet18 U-Net |
|---|---:|---:|
| 参数量 | 29,465 | 12,973,801 |
| 模型权重大小 | 0.12 MB | 49.58 MB |
| GPU平均推理延迟 | 1.426 ms | 4.134 ms |
| PyTorch显存峰值 | 17.56 MiB | 122.17 MiB |

ResNet18 U-Net的CPU推理性能：

- PyTorch CPU：44.703 ms。
- ONNX Runtime CPU：22.441 ms。
- ONNX Runtime加速比：1.99×。

以上为对应设备和测试配置下的实测结果。

## 5. 环境安装

推荐使用Python虚拟环境。

安装ONNX推理依赖：

```bash
python -m pip install -r requirements.txt
```

开发与训练环境：

```bash
python -m pip install -r requirements-dev.txt
```

使用NVIDIA GPU进行训练时，需要安装与本机CUDA环境兼容的PyTorch版本。

## 6. ONNX批量推理

将待预测图片放入`demo_images/`，并将导出的模型权重文件放置于：

`outputs/resnet18_unet.onnx`

在项目根目录运行：

```bash
python -m app.infer --input demo_images
```

指定输出目录：

```bash
python -m app.infer --input demo_images --output outputs/demo_v2
```

修改预测阈值：

```bash
python -m app.infer --input demo_images --threshold 0.73
```

查看全部命令参数：

```bash
python -m app.infer --help
```

当前ONNX推理使用256×256滑动窗口、64像素重叠，支持不同尺寸的RGB图片。

输出文件包括裂纹Mask、概率图和红色预测叠加图。

## 7. 实验内容

- U-Net语义分割网络实现。
- Weighted BCE与Dice Loss组合。
- 数据增强与训练采样实验。
- 验证集阈值优化。
- ImageNet预训练与ResNet18迁移学习。
- 编码器冻结与微调。
- BatchNorm训练策略实验。
- 模型独立测试与错误案例分析。
- GPU与CPU推理性能测试。
- PyTorch模型导出ONNX。
- ONNX Runtime批量滑动窗口推理。

## 8. 项目局限

模型在部分低对比度裂纹和复杂路面纹理图像中仍存在漏检或误检。

在实际工程应用之前，还需要进一步验证跨场景泛化能力，以及滑动窗口推理策略对预测结果的影响。

## 9. 模型与数据说明

原始数据集、训练权重及本地实验输出默认不纳入Git仓库。

ONNX模型文件需要单独提供。后续可通过GitHub Releases发布经过验证的模型文件。
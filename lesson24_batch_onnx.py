
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image


# =====================================
# 1. 配置
# =====================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "outputs" / "resnet18_unet.onnx"

INPUT_DIR = ROOT / "demo_images"

OUTPUT_DIR = ROOT / "outputs" / "onnx_batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 256
OVERLAP = 64
STRIDE = TILE_SIZE - OVERLAP

# 冻结编码器模型的验证集最佳阈值
THRESHOLD = 0.73

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =====================================
# 2. 加载ONNX模型
# =====================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX模型加载成功")
print("推理设备:", session.get_providers())


# =====================================
# 3. 单个图像块预测
# =====================================

def predict_tile(tile):

    # tile: H×W×3，uint8
    # 模型内部已包含ImageNet归一化
    x = tile.astype(np.float32) / 255.0

    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)

    x = np.ascontiguousarray(x)

    logits = session.run(
        [output_name],
        {input_name: x}
    )[0]

    logits = logits[0, 0]

    # Sigmoid，转为裂纹概率
    logits = np.clip(logits, -40, 40)

    probabilities = 1.0 / (1.0 + np.exp(-logits))

    return probabilities


# =====================================
# 4. 生成滑动窗口的起点
# =====================================

def get_positions(length):

    if length <= TILE_SIZE:
        return [0]

    positions = list(
        range(0, length - TILE_SIZE + 1, STRIDE)
    )

    # 确保覆盖最后一段
    last = length - TILE_SIZE

    if positions[-1] != last:
        positions.append(last)

    return positions


# =====================================
# 5. 任意尺寸图像推理
# =====================================

def predict_full_image(image_np):

    h, w = image_np.shape[:2]

    # 对尺寸不足256的图像进行边缘填充
    padded_h = max(h, TILE_SIZE)
    padded_w = max(w, TILE_SIZE)

    pad_bottom = padded_h - h
    pad_right = padded_w - w

    padded = np.pad(
        image_np,
        (
            (0, pad_bottom),
            (0, pad_right),
            (0, 0)
        ),
        mode="edge"
    )

    prob_sum = np.zeros(
        (padded_h, padded_w),
        dtype=np.float32
    )

    count_map = np.zeros(
        (padded_h, padded_w),
        dtype=np.float32
    )

    ys = get_positions(padded_h)
    xs = get_positions(padded_w)

    # 逐块推理
    for y in ys:
        for x in xs:

            tile = padded[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            tile_prob = predict_tile(tile)

            prob_sum[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += tile_prob

            count_map[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += 1

    # 对重叠区域的概率求平均
    full_prob = prob_sum / count_map

    # 裁掉之前填充的部分
    full_prob = full_prob[:h, :w]

    return full_prob, len(xs) * len(ys)


# =====================================
# 6. 保存预测结果
# =====================================

def save_results(image_np, prob_map, filename):

    pred_mask = prob_map > THRESHOLD

    # 二值Mask
    mask_u8 = pred_mask.astype(np.uint8) * 255

    # 概率图
    prob_u8 = np.clip(
        prob_map * 255,
        0, 255
    ).astype(np.uint8)

    # 红色预测叠加
    overlay = image_np.copy()

    red = np.array([255, 0, 0])

    overlay[pred_mask] = (
        0.55 * image_np[pred_mask]
        + 0.45 * red
    ).astype(np.uint8)

    # 文件名包含原扩展名，避免同名JPG/PNG覆盖
    stem = filename.replace(".", "_")

    Image.fromarray(mask_u8).save(
        OUTPUT_DIR / f"{stem}_mask.png"
    )

    Image.fromarray(prob_u8).save(
        OUTPUT_DIR / f"{stem}_probability.png"
    )

    Image.fromarray(overlay).save(
        OUTPUT_DIR / f"{stem}_overlay.png"
    )


# =====================================
# 7. 批量处理
# =====================================

INPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    print("没有找到图片，请检查:", INPUT_DIR)

else:
    print(f"共找到 {len(image_paths)} 张图片")

    start_all = time.perf_counter()

    for index, image_path in enumerate(image_paths, 1):

        start = time.perf_counter()

        with Image.open(image_path) as img:
            image_np = np.array(img.convert("RGB"))

        h, w = image_np.shape[:2]

        prob_map, tile_count = predict_full_image(
            image_np
        )

        save_results(
            image_np,
            prob_map,
            image_path.name
        )

        elapsed = time.perf_counter() - start

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"{w}×{h} | "
            f"{tile_count}块 | "
            f"{elapsed:.3f}s"
        )

    total_time = time.perf_counter() - start_all

    print("\n批量预测完成")
    print(f"总耗时: {total_time:.2f}s")
    print("输出目录:", OUTPUT_DIR)

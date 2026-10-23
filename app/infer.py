
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


# 项目根目录：app文件夹的上一级
ROOT = Path(__file__).resolve().parents[1]

TILE_SIZE = 256
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_positions(length, overlap):
    """计算滑动窗口起始位置。"""

    if length <= TILE_SIZE:
        return [0]

    stride = TILE_SIZE - overlap

    positions = list(
        range(0, length - TILE_SIZE + 1, stride)
    )

    last = length - TILE_SIZE

    if positions[-1] != last:
        positions.append(last)

    return positions


class CrackPredictor:
    """基于ONNX Runtime的裂纹分割器。"""

    def __init__(self, model_path, threshold=0.73,
                 overlap=64):

        if not 0 <= threshold <= 1:
            raise ValueError("threshold必须在0~1之间")

        if not 0 <= overlap < TILE_SIZE:
            raise ValueError("overlap必须在0~255之间")

        self.threshold = threshold
        self.overlap = overlap

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_name = (
            self.session.get_outputs()[0].name
        )

    def predict_tile(self, tile):
        """预测一个256×256图像块。"""

        x = tile.astype(np.float32) / 255.0

        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        x = np.ascontiguousarray(x)

        logits = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )[0][0, 0]

        logits = np.clip(logits, -40, 40)

        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, image):
        """预测任意尺寸RGB图像。"""

        h, w = image.shape[:2]

        padded_h = max(h, TILE_SIZE)
        padded_w = max(w, TILE_SIZE)

        padded = np.pad(
            image,
            (
                (0, padded_h - h),
                (0, padded_w - w),
                (0, 0)
            ),
            mode="edge"
        )

        prob_sum = np.zeros(
            (padded_h, padded_w),
            dtype=np.float32
        )

        count = np.zeros_like(prob_sum)

        ys = get_positions(padded_h, self.overlap)
        xs = get_positions(padded_w, self.overlap)

        for y in ys:
            for x in xs:

                tile = padded[
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE
                ]

                prob = self.predict_tile(tile)

                prob_sum[
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE
                ] += prob

                count[
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE
                ] += 1

        full_prob = (prob_sum / count)[:h, :w]

        mask = full_prob > self.threshold

        return full_prob, mask

    def save_results(self, image, prob, mask,
                     output_dir, name):
        """保存Mask、概率图和叠加图。"""

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_u8 = mask.astype(np.uint8) * 255

        prob_u8 = np.clip(
            prob * 255, 0, 255
        ).astype(np.uint8)

        overlay = image.copy()

        overlay[mask] = (
            0.55 * image[mask]
            + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)

        Image.fromarray(mask_u8).save(
            output_dir / f"{name}_mask.png"
        )

        Image.fromarray(prob_u8).save(
            output_dir / f"{name}_probability.png"
        )

        Image.fromarray(overlay).save(
            output_dir / f"{name}_overlay.png"
        )


def main():

    parser = argparse.ArgumentParser(
        description="ONNX路面裂纹语义分割"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "demo_images",
        help="输入图片或文件夹"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "onnx_batch",
        help="预测结果保存目录"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "outputs" / "resnet18_unet.onnx",
        help="ONNX模型文件路径"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.73,
        help="裂纹概率阈值"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="图像块重叠像素"
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"输入路径不存在: {args.input}")

    if not args.model.is_file():
        parser.error(f"模型不存在: {args.model}")

    # 支持单张图片或整个文件夹
    if args.input.is_file():
        image_paths = [args.input]
    else:
        image_paths = sorted(
            p for p in args.input.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        )

    if not image_paths:
        print("未找到可处理的图片")
        return

    predictor = CrackPredictor(
        model_path=args.model,
        threshold=args.threshold,
        overlap=args.overlap
    )

    print("模型加载成功")
    print("图片数量:", len(image_paths))
    print("预测阈值:", args.threshold)

    start_all = time.perf_counter()

    for index, image_path in enumerate(
        image_paths, 1
    ):

        start = time.perf_counter()

        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"))

        prob, mask = predictor.predict(image)

        # 保留原扩展名信息，避免同名文件覆盖
        name = (
            image_path.stem
            + "_"
            + image_path.suffix.lower().lstrip(".")
        )

        predictor.save_results(
            image,
            prob,
            mask,
            args.output,
            name
        )

        elapsed = time.perf_counter() - start

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"{elapsed:.3f}s"
        )

    print(
        f"全部完成，总耗时:"
        f"{time.perf_counter()-start_all:.2f}s"
    )

    print("输出目录:", args.output)


if __name__ == "__main__":
    main()

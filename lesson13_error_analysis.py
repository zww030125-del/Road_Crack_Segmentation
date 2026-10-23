
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent

csv_path = ROOT / "outputs" / "test_results.csv"

# 读取每张测试图像的评价指标
with open(csv_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    results = list(reader)

# 将指标转换成数值
for row in results:
    for key in ["precision", "recall", "dice", "iou"]:
        row[key] = float(row[key])

# 按Dice从小到大排序
results.sort(key=lambda x: x["dice"])

print("===== Dice最低的10张图像 =====")

for row in results[:10]:
    print(
        f"{row['image']:25s} "
        f"Dice={row['dice']:.4f} | "
        f"Precision={row['precision']:.4f} | "
        f"Recall={row['recall']:.4f}"
    )

print("\n===== Dice最高的5张图像 =====")

for row in results[-5:][::-1]:
    print(
        f"{row['image']:25s} "
        f"Dice={row['dice']:.4f}"
    )

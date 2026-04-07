"""
把data/raw里的parquet展开成图片和标注
每个split会在原文件夹生成png和json
给01准备可直接读取的原始数据
"""

import json
import re
import sys
from pathlib import Path

from datasets import load_dataset


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from config import DATASET_DIR, DIGIT_NUM


# parquet列表
def list_parquet():
    files = []
    for split_dir in sorted(DATASET_DIR.iterdir()):
        if not split_dir.is_dir():
            continue
        files.extend(sorted(split_dir.glob("*.parquet")))
    return files


# 读数文本
def gt_text(text):
    digits = re.findall(r"\d+", str(text))
    if not digits:
        return ""
    return digits[0].zfill(DIGIT_NUM)


# 目录清理
def clean_dir(split_dir):
    for path in split_dir.iterdir():
        if path.suffix.lower() in [".png", ".json"]:
            path.unlink()


# 单个parquet
def unpack_one(parquet_path):
    split_dir = parquet_path.parent
    split_name = split_dir.name
    data = load_dataset("parquet", data_files={"data": str(parquet_path)})["data"]
    stem = parquet_path.stem

    for i, row in enumerate(data):
        base_name = f"{stem}_{i:05d}"
        image_path = split_dir / f"{base_name}.png"
        json_path = split_dir / f"{base_name}.json"

        row["image"].save(image_path)
        json_path.write_text(
            json.dumps(
                {
                    "split": split_name,
                    "ground_truth": row["ground_truth"],
                    "meter_reading": gt_text(row["ground_truth"]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if (i + 1) % 200 == 0 or i == len(data) - 1:
            print(f"{split_name} {parquet_path.name} {i + 1}/{len(data)}")


# 主流程
def main():
    parquet_files = list_parquet()
    if not parquet_files:
        raise FileNotFoundError("data/raw里没有parquet文件")

    done_dirs = set()
    for parquet_path in parquet_files:
        split_dir = parquet_path.parent
        if split_dir not in done_dirs:
            clean_dir(split_dir)
            done_dirs.add(split_dir)
        unpack_one(parquet_path)


if __name__ == "__main__":
    main()

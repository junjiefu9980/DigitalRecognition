"""
把原始图像和读数整理成训练前要用的数据
比较不同阈值模式的切分情况
输出单字符数字样本和统计结果
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from config import (
    DATASET_DIR,
    DATASET_STATS_FILE,
    DIGIT_NUM,
    IMG_EXTS,
    PARSED_FILE,
    PREPARE_TH_MODE,
    SEG_SUMMARY_FILE,
    SINGLE_DIR,
)
from tools import ensure_dir, make_dirs, normalize_read, auto_seg, read_image, save_csv


THRESHOLD_MODES = ["otsu_threshold", "adaptive_threshold"]
TRAIN_TH_MODES = ["otsu_threshold"]


# 图像列表
def list_images(root_dir):
    root_dir = Path(root_dir)
    image_files = []
    for ext in IMG_EXTS:
        image_files.extend(root_dir.rglob(f"*{ext}"))
        image_files.extend(root_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(image_files))


# 标注列表
def list_ann(root_dir):
    root_dir = Path(root_dir)
    annotation_files = []
    for suffix in [".txt", ".json", ".xml", ".csv"]:
        annotation_files.extend(root_dir.rglob(f"*{suffix}"))
        annotation_files.extend(root_dir.rglob(f"*{suffix.upper()}"))
    return sorted(set(annotation_files))


# 数据目录
def find_data_dir(start_dir):
    start_dir = Path(start_dir)
    if not start_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{start_dir}")

    if list_images(start_dir):
        return start_dir

    for candidate in start_dir.rglob("*"):
        if candidate.is_dir() and list_images(candidate):
            return candidate

    raise FileNotFoundError("在data/raw里没有找到图像文件")


# 图像对应
def find_ann(image_path, annotation_files):
    image_path = Path(image_path)
    image_stem = image_path.stem.lower()
    split_name = image_path.parent.name.lower()

    same_split = [
        annotation_path
        for annotation_path in annotation_files
        if annotation_path.parent.name.lower() == split_name and annotation_path.stem.lower() == image_stem
    ]
    if same_split:
        return same_split[0]

    exact_list = [annotation_path for annotation_path in annotation_files if annotation_path.stem.lower() == image_stem]
    if exact_list:
        return exact_list[0]

    for annotation_path in annotation_files:
        annotation_stem = annotation_path.stem.lower()
        if image_stem in annotation_stem or annotation_stem in image_stem:
            return annotation_path

    return None


# 数据划分
def read_split(image_path, dataset_root):
    image_path = Path(image_path)
    dataset_root = Path(dataset_root)
    try:
        rel_parts = image_path.relative_to(dataset_root).parts
    except ValueError:
        rel_parts = image_path.parts

    if not rel_parts:
        raise ValueError(f"无法读取数据划分 {image_path}")

    split_name = str(rel_parts[0]).lower()
    if split_name not in ["train", "valid", "test"]:
        raise ValueError(f"不支持的数据划分 {split_name}")
    return split_name


# 文件名
def clean_name(text):
    return re.sub(r"[^\w\-\.]+", "_", str(text)).strip("_")


# 读数提取
def read_from_text(text, expected_length=DIGIT_NUM):
    text = str(text)
    patterns = [
        r"ground[_\s]?truth\s*[:=]\s*\"?([0-9]{1,5})\"?",
        r"reading\s*[:=]\s*([0-9]{1,5})",
        r"meter[_\s]?reading\s*[:=]\s*([0-9]{1,5})",
        r"gt[_\s]?parse\s*[:=]\s*([0-9]{1,5})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).zfill(expected_length)

    candidates = re.findall(r"\b\d{1,5}\b", text)
    if candidates:
        candidates = sorted(candidates, key=lambda item: abs(len(item) - expected_length))
        return str(candidates[0]).zfill(expected_length)
    return ""


# 生成记录
def build_record(image_path="", split="", annotation_path=""):
    record = {
        "image_path": str(image_path),
        "image_name": Path(image_path).name if image_path else "",
        "split": str(split),
        "annotation_path": str(annotation_path),
        "meter_reading": "",
    }
    for index in range(DIGIT_NUM):
        record[f"digit_{index}_label"] = -1
    return record


# 标签写回
def fill_labels(record, reading):
    reading = normalize_read(reading)
    record["meter_reading"] = reading
    if reading:
        for index, char in enumerate(reading[:DIGIT_NUM]):
            record[f"digit_{index}_label"] = int(char)
    return record


# 文本读取
def read_text(file_path):
    file_path = Path(file_path)
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1")


# 单个标注
def parse_ann(annotation_path, image_path=""):
    annotation_path = Path(annotation_path)
    record = build_record(image_path=image_path, split="", annotation_path=annotation_path)
    reading = read_from_text(read_text(annotation_path))
    return fill_labels(record, reading)


# 记录修正
def fix_record(record):
    digit_list = []
    for index in range(DIGIT_NUM):
        value = record.get(f"digit_{index}_label", -1)
        if value != -1:
            digit_list.append(str(int(value)))

    if not record["meter_reading"] and len(digit_list) == DIGIT_NUM:
        record["meter_reading"] = "".join(digit_list)

    reading = normalize_read(record["meter_reading"])
    if reading:
        record = fill_labels(record, reading)
    return record


# 读数整理
def build_parsed():
    dataset_root = find_data_dir(DATASET_DIR)
    image_files = list_images(dataset_root)
    annotation_files = list_ann(dataset_root)

    if not image_files:
        raise FileNotFoundError("未找到原始图像")

    records = []
    print("整理原始读数")

    for index, image_path in enumerate(image_files):
        split_name = read_split(image_path, dataset_root)
        annotation_path = find_ann(image_path, annotation_files)

        if annotation_path is None:
            record = build_record(image_path=image_path, split=split_name, annotation_path="")
        else:
            record = parse_ann(annotation_path, image_path=image_path)
            record["split"] = split_name

        records.append(fix_record(record))

        if (index + 1) % 100 == 0 or index == len(image_files) - 1:
            print(f"已处理 {index + 1}/{len(image_files)}")

    return records


# 读数保存
def save_parsed(records):
    dataframe = pd.DataFrame(records)
    preferred_columns = [
        "image_path",
        "image_name",
        "meter_reading",
        "split",
        "annotation_path",
    ] + [f"digit_{index}_label" for index in range(DIGIT_NUM)]
    extra_columns = [column for column in dataframe.columns.tolist() if column not in preferred_columns]
    dataframe = dataframe[preferred_columns + extra_columns]

    save_csv(dataframe, PARSED_FILE)
    return dataframe


# 读数表
def load_parsed():
    return pd.read_csv(
        PARSED_FILE,
        dtype={
            "image_path": str,
            "image_name": str,
            "split": str,
            "annotation_path": str,
            "meter_reading": str,
        },
    )


# 单图切分
def single_image_seg(image_path, threshold_mode):
    image = read_image(image_path)
    pipeline_result = auto_seg(image, threshold_mode=threshold_mode)

    success = bool(pipeline_result["success"])
    fail_reason = pipeline_result["fail_reason"]

    return {
        "image_path": str(image_path),
        "image_name": Path(image_path).name,
        "threshold_mode": threshold_mode,
        "success": success,
        "num_digits": len(pipeline_result["digit_images"]),
        "raw_region_count": int(pipeline_result["raw_region_count"]),
        "used_center_crop": bool(pipeline_result["used_center_crop"]),
        "used_even_split": bool(pipeline_result["used_even_split"]),
        "fail_reason": fail_reason,
        "debug_dir": "",
    }


# 批量切分
def run_seg_set(threshold_mode, split="all", max_images=None):
    parsed_df = load_parsed()
    if split != "all":
        parsed_df = parsed_df[parsed_df["split"] == split].copy()
    if max_images is not None:
        parsed_df = parsed_df.head(max_images).copy()

    rows = []
    total_count = len(parsed_df)
    print(f"开始切分检查 模式={threshold_mode} 样本数={total_count}")

    for i, row in parsed_df.iterrows():
        try:
            result = single_image_seg(row["image_path"], threshold_mode)
            result["meter_reading"] = normalize_read(row["meter_reading"])
            result["split"] = str(row["split"])
        except Exception as error:
            result = {
                "image_path": row["image_path"],
                "image_name": Path(row["image_path"]).name,
                "threshold_mode": threshold_mode,
                "success": False,
                "num_digits": 0,
                "raw_region_count": 0,
                "used_center_crop": False,
                "used_even_split": False,
                "fail_reason": str(error),
                "debug_dir": "",
                "meter_reading": normalize_read(row["meter_reading"]),
                "split": str(row["split"]),
            }

        rows.append(result)

        if len(rows) % 100 == 0 or len(rows) == total_count:
            print(f"已完成 {len(rows)}/{total_count}")

    return pd.DataFrame(rows)


# 切分对比
def save_seg(logs_map):
    summary_rows = []

    for threshold_mode, logs_df in logs_map.items():
        total_images = len(logs_df)
        success_count = int(logs_df["success"].sum()) if not logs_df.empty else 0
        success_rate = success_count / total_images if total_images > 0 else 0.0

        summary_rows.append(
            {
                "threshold_mode": threshold_mode,
                "total_images": total_images,
                "success_count": success_count,
                "success_rate": success_rate,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("success_rate", ascending=False)
    save_csv(summary_df, SEG_SUMMARY_FILE)
    return summary_df


# 目录重建
def reset_digit_dir():
    if SINGLE_DIR.exists():
        shutil.rmtree(SINGLE_DIR)
    for split_name in ["train", "valid", "test"]:
        for label in range(10):
            ensure_dir(SINGLE_DIR / split_name / str(label))


# 数据集生成
def build_digit_set(threshold_mode):
    reset_digit_dir()
    parsed_df = load_parsed()

    metadata_rows = []
    success_image_count = 0

    for i, row in parsed_df.iterrows():
        meter_reading = normalize_read(row["meter_reading"])
        split_name = str(row["split"])
        mode_list = TRAIN_TH_MODES if split_name == "train" else [threshold_mode]

        if len(meter_reading) != DIGIT_NUM or not meter_reading.isdigit():
            continue

        try:
            image = read_image(row["image_path"])
        except Exception as error:
            print(f"跳过读取失败样本 {row['image_path']} 原因 {error}")
            continue

        image_stem = Path(row["image_path"]).stem
        saved_image = False
        for mode in mode_list:
            pipeline_result = auto_seg(image, threshold_mode=mode)
            digit_images = pipeline_result["digit_images"]

            if len(digit_images) != DIGIT_NUM:
                continue

            for digit_index, digit_image in enumerate(digit_images):
                digit_label = int(meter_reading[digit_index])
                save_dir = SINGLE_DIR / split_name / str(digit_label)
                file_name = (
                    f"{clean_name(image_stem)}"
                    f"_split-{split_name}"
                    f"_th-{clean_name(mode)}"
                    f"_label-{digit_label}"
                    f"_idx-{digit_index}.png"
                )
                save_path = save_dir / file_name
                cv2.imwrite(str(save_path), digit_image)
                metadata_rows.append(
                    {
                        "image_name": Path(row["image_path"]).name,
                        "split": split_name,
                        "digit_index": digit_index,
                        "digit_label": digit_label,
                        "saved_path": str(save_path),
                        "threshold_mode": mode,
                    }
                )
            saved_image = True

        if saved_image:
            success_image_count += 1
            if success_image_count % 100 == 0:
                print(f"已生成 {success_image_count} 张单字符样本")

    metadata_df = pd.DataFrame(metadata_rows)
    stats_df = (
        metadata_df.groupby(["split", "digit_label"])
        .size()
        .reset_index(name="sample_count")
        .sort_values(["split", "digit_label"])
    )

    save_csv(stats_df, DATASET_STATS_FILE)
    return metadata_df, stats_df


# 主流程
def main():
    parser = argparse.ArgumentParser(description="整理读数 检查切分 生成单字符数据集")
    parser.add_argument(
        "--threshold-mode",
        default=PREPARE_TH_MODE,
        choices=THRESHOLD_MODES,
        help="生成单字符数据集时使用的阈值模式",
    )
    parser.add_argument(
        "--compare-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否比较三种阈值模式",
    )
    parser.add_argument("--compare-split", default="test", choices=["all", "train", "valid", "test"], help="切分检查使用的数据划分")
    parser.add_argument("--compare-max-images", type=int, default=100, help="切分检查最多处理多少张图 填0表示全部")
    args = parser.parse_args()

    compare_max_images = None if args.compare_max_images <= 0 else args.compare_max_images

    # 目录
    make_dirs()

    # 读数
    records = build_parsed()
    parsed_df = save_parsed(records)
    print(f"结构化读数已保存到 {PARSED_FILE.parent}")
    print(parsed_df["split"].value_counts(dropna=False))

    # 切分
    if args.compare_all:
        logs_map = {}
        for mode in THRESHOLD_MODES:
            logs_map[mode] = run_seg_set(mode, split=args.compare_split, max_images=compare_max_images)
        compare_df = save_seg(logs_map)
    else:
        logs_df = run_seg_set(args.threshold_mode, split=args.compare_split, max_images=compare_max_images)
        compare_df = save_seg({args.threshold_mode: logs_df})

    # 数据集
    metadata_df, stats_df = build_digit_set(args.threshold_mode)

    print("数据准备完成")
    print(f"单字符样本数 {len(metadata_df)}")
    print(f"数字数据集目录 {SINGLE_DIR}")
    print("切分对比")
    print(compare_df)
    print("类别统计")
    print(stats_df)


if __name__ == "__main__":
    main()

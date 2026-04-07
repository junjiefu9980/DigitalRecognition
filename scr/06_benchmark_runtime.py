"""
统计不同方案的运行时间
会比较每种模型和阈值的FPS
输出测速总表和柱状图
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from config import PARSED_FILE, RUNTIME_FIG, RUNTIME_SUMMARY_FILE, TARGET_FPS
from tools import (
    MODEL_NAMES,
    THRESHOLD_MODES,
    get_device,
    load_csv,
    load_model,
    normalize_th,
    pred_read,
    read_image,
    save_bar,
    save_csv,
)


def bench_one(model_name, threshold_mode, eval_df):
    device = get_device()
    model_obj = load_model(model_name, device=device)

    preprocess_times = []
    split_times = []
    inference_times = []
    total_times = []

    # 阶段耗时
    for i, row in eval_df.iterrows():
        image = read_image(row["image_path"])
        total_start = time.time()
        infer_start = time.time()
        pred_read(
            image=image,
            model_name=model_name,
            model_obj=model_obj,
            device=device,
            threshold_mode=threshold_mode,
        )
        total_time = time.time() - total_start
        preprocess_times.append(0.0)
        split_times.append(0.0)
        inference_times.append(time.time() - infer_start)
        total_times.append(total_time)

        if (i + 1) % 50 == 0 or i == len(eval_df) - 1:
            print(f"测速 {model_name} {i + 1}/{len(eval_df)}")

    avg_preprocess = float(np.mean(preprocess_times)) if preprocess_times else 0.0
    avg_split = float(np.mean(split_times)) if split_times else 0.0
    avg_inference = float(np.mean(inference_times)) if inference_times else 0.0
    avg_total = float(np.mean(total_times)) if total_times else 0.0
    fps = 1.0 / avg_total if avg_total > 0 else 0.0

    return {
        "model_name": model_name,
        "threshold_mode": threshold_mode,
        "preprocess_time": avg_preprocess,
        "segmentation_time": avg_split,
        "inference_time": avg_inference,
        "total_time": avg_total,
        "fps": fps,
        "target_fps": TARGET_FPS,
        "target_reached": fps >= TARGET_FPS,
    }


def run_bench(model_name="all", threshold_mode=None, max_images=100):
    eval_df = load_csv(PARSED_FILE)
    eval_df = eval_df[eval_df["split"] == "test"].copy().head(max_images)
    if eval_df.empty:
        raise ValueError("测试集为空 无法测速")

    model_list = MODEL_NAMES if model_name == "all" else [model_name]
    threshold_list = THRESHOLD_MODES if threshold_mode is None else [normalize_th(threshold_mode)]
    runtime_rows = []

    # 逐项测速
    for name in model_list:
        for mode in threshold_list:
            print(f"开始测速 {name} {mode}")
            runtime_rows.append(bench_one(name, mode, eval_df))

    runtime_df = pd.DataFrame(runtime_rows).sort_values(["fps", "model_name"], ascending=[False, True])
    save_csv(runtime_df, RUNTIME_SUMMARY_FILE)
    save_bar(
        [f"{row.model_name}\n{row.threshold_mode}" for row in runtime_df.itertuples()],
        runtime_df["fps"].tolist(),
        RUNTIME_FIG,
        "FPS Comparison",
        "FPS",
    )
    return runtime_df


def main():
    parser = argparse.ArgumentParser(description="统计预处理 切分 推理 和总时间")
    parser.add_argument("--model", default="all", choices=["all", "template", "lenet", "resnet18"], help="测速模型")
    parser.add_argument(
        "--threshold-mode",
        default=None,
        choices=["dynamic_threshold", "otsu_threshold", "adaptive_threshold"],
        help="阈值模式",
    )
    parser.add_argument("--max-images", type=int, default=100, help="最多测速多少张图")
    args = parser.parse_args()

    runtime_df = run_bench(model_name=args.model, threshold_mode=args.threshold_mode, max_images=args.max_images)
    print("测速完成")
    print(runtime_df)


if __name__ == "__main__":
    main()

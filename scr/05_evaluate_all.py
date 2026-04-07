"""
把模板匹配LeNet和ResNet18放到一起评估
会统计字符级结果
输出总表和一张对比图
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from config import DIGIT_NUM, EVAL_CHAR_FIG, EVAL_SUMMARY_FILE, PARSED_FILE, TARGET_ACC, TARGET_OUTPUT_RATE
from tools import (
    MODEL_NAMES,
    THRESHOLD_MODES,
    calc_metrics,
    get_device,
    load_csv,
    load_model,
    normalize_read,
    pred_read,
    read_image,
    save_bar,
    save_csv,
)


def empty_metrics():
    return {
        "accuracy": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_macro": 0.0,
        "confusion_matrix": np.zeros((10, 10), dtype=int),
        "classification_report": {},
    }


def eval_one(model_name, threshold_mode, eval_df):
    device = get_device()
    model_obj = load_model(model_name, device=device)

    total_images = len(eval_df)
    total_chars = total_images * DIGIT_NUM
    region_success_count = 0
    correct_char_count = 0
    valid_output_count = 0
    valid_correct_char_count = 0
    valid_char_total = 0

    y_true = []
    y_pred = []

    # 逐张评估
    for i, row in eval_df.iterrows():
        image = read_image(row["image_path"])
        meter_reading = normalize_read(row["meter_reading"])
        infer_result = pred_read(
            image=image,
            model_name=model_name,
            model_obj=model_obj,
            device=device,
            threshold_mode=threshold_mode,
        )

        pipeline_result = infer_result["pipeline_result"]
        pred_labels = infer_result["pred_labels"]
        if pipeline_result["success"]:
            region_success_count += 1

        is_valid_output = bool(infer_result["is_valid_output"])
        if is_valid_output:
            valid_output_count += 1

        if len(pred_labels) == DIGIT_NUM:
            for digit_index in range(DIGIT_NUM):
                true_label = int(meter_reading[digit_index])
                pred_label = int(pred_labels[digit_index])
                y_true.append(true_label)
                y_pred.append(pred_label)
                if true_label == pred_label:
                    correct_char_count += 1
                    if is_valid_output:
                        valid_correct_char_count += 1
                if is_valid_output:
                    valid_char_total += 1

        if (i + 1) % 50 == 0 or i == len(eval_df) - 1:
            print(f"评估 {model_name} {threshold_mode} {i + 1}/{len(eval_df)}")

    if y_true:
        metrics = calc_metrics(y_true, y_pred, class_labels=list(range(10)))
    else:
        metrics = empty_metrics()

    digit_accuracy = metrics["accuracy"]
    char_level_accuracy = correct_char_count / total_chars if total_chars > 0 else 0.0
    effective_output_rate = valid_output_count / total_images if total_images > 0 else 0.0
    effective_char_accuracy = valid_correct_char_count / valid_char_total if valid_char_total > 0 else 0.0
    region_success_rate = region_success_count / total_images if total_images > 0 else 0.0

    return {
        "model_name": model_name,
        "threshold_mode": threshold_mode,
        "digit_accuracy": digit_accuracy,
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "char_level_accuracy": char_level_accuracy,
        "effective_char_accuracy": effective_char_accuracy,
        "effective_output_rate": effective_output_rate,
        "region_success_rate": region_success_rate,
        "target_accuracy": TARGET_ACC,
        "target_output_rate": TARGET_OUTPUT_RATE,
        "target_reached": (effective_char_accuracy >= TARGET_ACC) and (effective_output_rate >= TARGET_OUTPUT_RATE),
    }


def save_eval(summary_df):
    save_csv(summary_df, EVAL_SUMMARY_FILE)

    labels = [f"{row.model_name}\n{row.threshold_mode}" for row in summary_df.itertuples()]

    # 字符准确率
    save_bar(
        labels,
        summary_df["effective_char_accuracy"].tolist(),
        EVAL_CHAR_FIG,
        "Effective Character Accuracy",
        "Accuracy",
    )


def run_eval():
    eval_df = load_csv(PARSED_FILE)
    eval_df = eval_df[eval_df["split"] == "test"].copy()
    if eval_df.empty:
        raise ValueError("测试集为空 无法评估")

    summary_rows = []
    for threshold_mode in THRESHOLD_MODES:
        for model_name in MODEL_NAMES:
            print(f"开始评估 {model_name} {threshold_mode}")
            summary_rows.append(eval_one(model_name, threshold_mode, eval_df))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["effective_char_accuracy", "effective_output_rate", "char_level_accuracy"],
        ascending=False,
    )
    save_eval(summary_df)
    return summary_df


def main():
    summary_df = run_eval()
    print("统一评估完成")
    print(summary_df)


if __name__ == "__main__":
    main()

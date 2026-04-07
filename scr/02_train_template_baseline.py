"""
用模板匹配做传统方法对比
输入是已经生成好的单字符数字样本
输出模板方法的准确率结果
"""

import sys
from pathlib import Path

import cv2
import pandas as pd
from torchvision import datasets


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from config import SINGLE_DIR, TARGET_ACC, TEMPLATE_SUMMARY_FILE
from tools import calc_metrics, make_templates, save_csv, template_pred


def eval_template():
    test_dataset = datasets.ImageFolder(str(SINGLE_DIR / "test"))
    test_samples = test_dataset.samples
    if not test_samples:
        raise ValueError("测试集为空 请先准备数据集")

    templates = make_templates(SINGLE_DIR / "train")

    y_true = []
    y_pred = []

    # 模板匹配
    for index, sample in enumerate(test_samples):
        image_path, label_index = sample
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        pred_label, score_map = template_pred(image, templates)
        y_true.append(int(label_index))
        y_pred.append(pred_label)

        if (index + 1) % 500 == 0 or index == len(test_samples) - 1:
            print(f"模板匹配 {index + 1}/{len(test_samples)}")

    metrics = calc_metrics(y_true, y_pred, class_labels=list(range(10)))
    summary_df = pd.DataFrame(
        [
            {
                "model_name": "template",
                "digit_accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "target_accuracy": TARGET_ACC,
                "target_reached": metrics["accuracy"] >= TARGET_ACC,
            }
        ]
    )
    save_csv(summary_df, TEMPLATE_SUMMARY_FILE)
    return summary_df


def main():
    summary_df = eval_template()
    print("模板匹配完成")
    print(summary_df)


if __name__ == "__main__":
    main()

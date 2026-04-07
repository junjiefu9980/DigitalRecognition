"""
对一张完整仪表图做识别
自动切分5位数字并给出报警结果
输出终端里的识别结果
"""

import argparse
import sys
from pathlib import Path


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from config import DEMO_IMAGE, LOW_LIM, UP_LIM
from tools import (
    get_device,
    load_model,
    pred_read,
    read_image,
    pick_runtime,
)


def main():
    parser = argparse.ArgumentParser(description="单张整图推理")
    parser.add_argument("--image", default=str(DEMO_IMAGE), help="输入图像路径")
    parser.add_argument("--model", default=None, choices=["template", "lenet", "resnet18"], help="识别方法")
    parser.add_argument(
        "--threshold-mode",
        default=None,
        choices=["dynamic_threshold", "otsu_threshold", "adaptive_threshold"],
        help="阈值模式",
    )
    parser.add_argument("--lower-limit", default=LOW_LIM, type=int, help="报警下限")
    parser.add_argument("--upper-limit", default=UP_LIM, type=int, help="报警上限")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"输入图像不存在：{image_path}")

    model_name, threshold_mode = pick_runtime(args.model, args.threshold_mode)
    original_image = read_image(image_path)
    device = get_device()
    model_obj = load_model(model_name, device=device)

    # 推理结果
    infer_result = pred_read(
        image=original_image,
        model_name=model_name,
        model_obj=model_obj,
        device=device,
        threshold_mode=threshold_mode,
        lower_limit=args.lower_limit,
        upper_limit=args.upper_limit,
    )

    summary = {
        "image_path": str(image_path),
        "model_name": model_name,
        "threshold_mode": threshold_mode,
        "reading_text": infer_result["reading_text"],
        "raw_reading_text": infer_result["raw_reading_text"],
        "digit_predictions": infer_result["pred_labels"],
        "digit_scores": infer_result["pred_scores"],
        "mean_score": infer_result["mean_score"],
        "is_valid_output": infer_result["is_valid_output"],
        "is_alarm": infer_result["is_alarm"],
        "alarm_text": infer_result["alarm_text"],
        "region_success": infer_result["pipeline_result"]["success"],
        "fail_reason": infer_result["pipeline_result"]["fail_reason"],
    }

    print("单图推理完成")
    print(f"图像路径 {summary['image_path']}")
    print(f"模型 {summary['model_name']}")
    print(f"阈值模式 {summary['threshold_mode']}")
    print(f"读数 {summary['reading_text']}")
    print(f"原始读数 {summary['raw_reading_text']}")
    print(f"数字结果 {summary['digit_predictions']}")
    print(f"数字分数 {summary['digit_scores']}")
    print(f"平均分数 {summary['mean_score']:.4f}")
    print(f"有效输出 {summary['is_valid_output']}")
    print(f"报警 {summary['is_alarm']}")
    print(f"报警状态 {summary['alarm_text']}")
    print(f"区域成功 {summary['region_success']}")
    print(f"失败原因 {summary['fail_reason']}")



if __name__ == "__main__":
    main()

"""
打开摄像头做实时识别
画面里会显示读数 报警 和FPS
输出实时窗口
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from config import LOW_LIM, UP_LIM
from tools import draw_result, get_device, load_model, pred_read, pick_runtime


def main():
    # 窗口名
    window_name = "Realtime Meter Demo"

    parser = argparse.ArgumentParser(description="摄像头实时识别")
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

    model_name, threshold_mode = pick_runtime(args.model, args.threshold_mode)
    device = get_device()
    model_obj = load_model(model_name, device=device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("实时识别已启动 按q或esc退出")

    last_logged_reading = ""

    while True:
        success, frame = cap.read()
        if not success:
            continue

        loop_start = time.time()
        reading_text = ""
        alarm_text = "未检测到有效数字区域"

        try:
            infer_result = pred_read(
                image=frame,
                model_name=model_name,
                model_obj=model_obj,
                device=device,
                threshold_mode=threshold_mode,
                lower_limit=args.lower_limit,
                upper_limit=args.upper_limit,
            )
            reading_text = infer_result["reading_text"]
            alarm_text = infer_result["alarm_text"]

            if reading_text and reading_text != last_logged_reading:
                last_logged_reading = reading_text
        except Exception as error:
            alarm_text = f"识别失败: {error}"

        fps = 1.0 / max(time.time() - loop_start, 1e-6)
        display_frame = draw_result(frame, model_name, threshold_mode, reading_text, alarm_text, fps=fps)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q"), 27]:
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("实时识别结束")


if __name__ == "__main__":
    main()

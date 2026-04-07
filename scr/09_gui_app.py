"""
打开图形界面做展示
界面里可以看当前读数 报警 历史记录 和趋势图
输出交互界面
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from config import LOW_LIM, UP_LIM
from tools import draw_result, get_device, load_model, pred_read, read_image, pick_runtime


class MeterRecognitionGUI:
    def __init__(self, root, default_model=None, default_threshold=None):
        # 默认方案
        default_model, default_threshold = pick_runtime(default_model, default_threshold)
        self.root = root
        self.root.title("工业仪表数字识别系统")
        self.root.geometry("1220x820")

        self.device = get_device()
        self.current_model = None
        self.cap = None
        self.camera_running = False
        self.photo_image = None
        self.history_rows = []
        self.last_logged_reading = ""
        self.default_model = default_model
        self.default_threshold = default_threshold

        self.build_widgets()
        self.load_model()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_widgets(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(control_frame, text="选择图片", width=12, command=self.select_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="打开摄像头", width=12, command=self.open_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="停止", width=10, command=self.stop_camera).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="模型：").pack(side=tk.LEFT, padx=(20, 5))
        self.model_var = tk.StringVar(value=self.default_model)
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, values=["template", "lenet", "resnet18"], state="readonly", width=14)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        tk.Label(control_frame, text="阈值模式：").pack(side=tk.LEFT, padx=(20, 5))
        self.threshold_var = tk.StringVar(value=self.default_threshold)
        self.threshold_combo = ttk.Combobox(
            control_frame,
            textvariable=self.threshold_var,
            values=["dynamic_threshold", "otsu_threshold", "adaptive_threshold"],
            state="readonly",
            width=18,
        )
        self.threshold_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="报警下限：").pack(side=tk.LEFT, padx=(20, 5))
        self.lower_entry = tk.Entry(control_frame, width=8)
        self.lower_entry.insert(0, str(LOW_LIM))
        self.lower_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="报警上限：").pack(side=tk.LEFT, padx=(10, 5))
        self.upper_entry = tk.Entry(control_frame, width=8)
        self.upper_entry.insert(0, str(UP_LIM))
        self.upper_entry.pack(side=tk.LEFT, padx=5)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(main_frame, width=340)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_label = tk.Label(left_frame, text="图像显示区域", bg="black", width=80, height=32)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        info_frame = tk.LabelFrame(right_frame, text="当前结果", padx=10, pady=10)
        info_frame.pack(fill=tk.X, pady=5)

        self.reading_var = tk.StringVar(value="-----")
        self.alarm_var = tk.StringVar(value="正常")
        self.mode_var = tk.StringVar(value=self.default_model)
        self.threshold_show_var = tk.StringVar(value=self.default_threshold)

        tk.Label(info_frame, text="当前读数：").pack(anchor="w")
        self.reading_label = tk.Label(info_frame, textvariable=self.reading_var, font=("Arial", 24, "bold"), fg="blue")
        self.reading_label.pack(anchor="w", pady=4)

        tk.Label(info_frame, text="报警状态：").pack(anchor="w")
        self.alarm_label = tk.Label(info_frame, textvariable=self.alarm_var, font=("Arial", 18, "bold"), fg="green")
        self.alarm_label.pack(anchor="w", pady=4)

        tk.Label(info_frame, text="当前模型：").pack(anchor="w")
        tk.Label(info_frame, textvariable=self.mode_var, font=("Arial", 12)).pack(anchor="w", pady=2)

        tk.Label(info_frame, text="当前阈值：").pack(anchor="w")
        tk.Label(info_frame, textvariable=self.threshold_show_var, font=("Arial", 12)).pack(anchor="w", pady=2)

        history_frame = tk.LabelFrame(right_frame, text="历史记录", padx=10, pady=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.history_listbox = tk.Listbox(history_frame, height=16)
        self.history_listbox.pack(fill=tk.BOTH, expand=True)

        trend_frame = tk.LabelFrame(right_frame, text="读数趋势", padx=5, pady=5)
        trend_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.figure = Figure(figsize=(3.7, 2.4), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title("Reading Trend")
        self.axis.set_xlabel("Index")
        self.axis.set_ylabel("Value")
        self.canvas = FigureCanvasTkAgg(self.figure, master=trend_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_model(self):
        try:
            self.current_model = load_model(self.model_var.get(), device=self.device)
            self.mode_var.set(self.model_var.get())
        except Exception as error:
            self.current_model = None
            messagebox.showerror("模型加载失败", str(error))

    def on_model_change(self, event=None):
        self.load_model()

    def get_limits(self):
        try:
            lower_limit = int(self.lower_entry.get().strip())
            upper_limit = int(self.upper_entry.get().strip())
            return lower_limit, upper_limit
        except Exception:
            raise ValueError("报警阈值必须是整数。")

    def select_image(self):
        self.root.update_idletasks()
        self.root.lift()
        self.root.attributes("-topmost", True)
        try:
            image_path = filedialog.askopenfilename(
                parent=self.root,
                title="选择图片",
                initialdir=str((PRO_ROOT / "data" / "raw" / "test").resolve()),
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
            )
        finally:
            self.root.attributes("-topmost", False)
        if not image_path:
            return

        try:
            image = read_image(image_path)
            self.process_and_show(image, source_name=Path(image_path).name, show_popup=True)
        except Exception as error:
            messagebox.showerror("图片识别失败", str(error))

    def open_camera(self):
        if self.camera_running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头。")
            return

        self.camera_running = True
        self.update_camera_frame()

    def stop_camera(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_camera_frame(self):
        if not self.camera_running or self.cap is None:
            return

        success, frame = self.cap.read()
        if success:
            self.process_and_show(frame, source_name="camera", show_popup=False)

        self.root.after(30, self.update_camera_frame)

    def process_and_show(self, image, source_name, show_popup):
        if self.current_model is None:
            raise RuntimeError("当前模型未加载成功。")

        lower_limit, upper_limit = self.get_limits()
        threshold_mode = self.threshold_var.get()
        self.threshold_show_var.set(threshold_mode)

        infer_result = pred_read(
            image=image,
            model_name=self.model_var.get(),
            model_obj=self.current_model,
            device=self.device,
            threshold_mode=threshold_mode,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

        reading_text = infer_result["reading_text"] if infer_result["reading_text"] else "-----"
        alarm_text = infer_result["alarm_text"]
        display_image = draw_result(image, self.model_var.get(), threshold_mode, infer_result["reading_text"], alarm_text)

        self.show_image(display_image)
        self.reading_var.set(reading_text)
        self.alarm_var.set(alarm_text)
        self.alarm_label.config(fg="red" if alarm_text == "报警" else "green")

        # 历史趋势
        self.append_history(source_name, infer_result["reading_text"], alarm_text, threshold_mode)

        if show_popup and alarm_text == "报警":
            messagebox.showwarning("报警提示", f"当前读数 {reading_text} 超出阈值范围。")

    def show_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        pil_image.thumbnail((820, 720))
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.photo_image, text="")

    def append_history(self, source_name, reading_text, alarm_text, threshold_mode):
        if not reading_text:
            return
        if reading_text == self.last_logged_reading and source_name == "camera":
            return

        self.last_logged_reading = reading_text
        self.history_rows.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_name": source_name,
                "model_name": self.model_var.get(),
                "threshold_mode": threshold_mode,
                "reading_text": reading_text,
                "alarm_text": alarm_text,
            }
        )
        self.refresh_history_list()
        self.refresh_trend()

    def refresh_history_list(self):
        self.history_listbox.delete(0, tk.END)
        for row in self.history_rows[-30:]:
            text = f"{row['timestamp']} | {row['model_name']} | {row['threshold_mode']} | {row['reading_text']} | {row['alarm_text']}"
            self.history_listbox.insert(tk.END, text)

    def refresh_trend(self):
        values = []
        for row in self.history_rows:
            if str(row["reading_text"]).isdigit():
                values.append(int(row["reading_text"]))

        self.axis.clear()
        self.axis.set_title("Reading Trend")
        self.axis.set_xlabel("Index")
        self.axis.set_ylabel("Value")

        if values:
            self.axis.plot(values, marker="o")
        self.canvas.draw()

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="图形界面展示")
    parser.add_argument("--model", default=None, choices=["template", "lenet", "resnet18"], help="识别方法")
    parser.add_argument(
        "--threshold-mode",
        default=None,
        choices=["dynamic_threshold", "otsu_threshold", "adaptive_threshold"],
        help="阈值模式",
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = MeterRecognitionGUI(root, default_model=args.model, default_threshold=args.threshold_mode)
    root.mainloop()


if __name__ == "__main__":
    main()

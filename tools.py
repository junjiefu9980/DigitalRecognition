"""
放多个脚本都会用到的公共函数
包括切分 训练 推理 和显示
给00到09提供共享代码
"""

import json
import os
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import (
    AUG_BLUR_PROB,
    AUG_BRIGHTNESS,
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    AUG_NOISE_PROB,
    AUG_NOISE_STD,
    AUG_ROTATE_DEG,
    AUG_SCALE_RANGE,
    AUG_SHADOW_PROB,
    AUG_TRANSLATE,
    BATCH_SIZE,
    CKPT_DIR,
    CONF_SCORE_TH,
    DATA_DIR,
    DATASET_DIR,
    DIGIT_NUM,
    DIGIT_SIZE,
    EVAL_SUMMARY_FILE,
    LENET_EPOCHS,
    LENET_SUMMARY_FILE,
    LEARNING_RATE,
    LOW_LIM,
    PARSE_DIR,
    PARSED_FILE,
    PREPARE_TH_MODE,
    PRO_ROOT,
    RANDOM_SEED,
    READING_SIZE,
    RESULT_DIR,
    RESNET18_PRETRAIN,
    RESNET18_EPOCHS,
    RESNET18_SUMMARY_FILE,
    RUNTIME_SUMMARY_FILE,
    SINGLE_DIR,
    TARGET_ACC,
    TARGET_OUTPUT_RATE,
    TARGET_FPS,
    TEMP_NUM,
    UP_LIM,
)
from models.lenet import LeNet5
from models.resnet18 import build_resnet18

MPL_CONFIG_DIR = PRO_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


THRESHOLD_MODES = ["dynamic_threshold", "otsu_threshold", "adaptive_threshold"]
MODEL_NAMES = ["template", "lenet", "resnet18"]


#--------- 基础

def ensure_dir(dir_path):
    # 创建目录
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_dirs():
    # 常用目录
    for path in [
        DATA_DIR,
        DATASET_DIR,
        PARSE_DIR,
        SINGLE_DIR,
        CKPT_DIR,
        RESULT_DIR,
    ]:
        ensure_dir(path)


def save_csv(dataframe, file_path):
    # 保存CSV
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    dataframe.to_csv(file_path, index=False, encoding="utf-8-sig")


def load_csv(file_path):
    # 读取CSV
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")
    return pd.read_csv(file_path)


@lru_cache(maxsize=1)
def get_pos_prior():
    # 位置先验
    parsed_df = load_csv(PARSED_FILE)
    parsed_df = parsed_df[parsed_df["split"] == "train"].copy()
    if parsed_df.empty:
        return None

    prior_map = {}
    for index in range(DIGIT_NUM):
        col_name = f"digit_{index}_label"
        counts = np.ones(10, dtype=np.float32)
        for label, count in parsed_df[col_name].value_counts().items():
            label = int(label)
            if 0 <= label <= 9:
                counts[label] += float(count)
        prior_map[index] = counts / counts.sum()
    return prior_map


plt.rcParams["axes.unicode_minus"] = False


def save_bar(labels, values, save_path, title, y_label):
    # 柱状图
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#4C72B0")
    plt.title(title)
    plt.ylabel(y_label)
    max_value = max(values) if values else 0.0
    pad = max(0.02, max_value * 0.06)
    plt.ylim(0, max_value + pad * 2 if max_value > 0 else 1)
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y + pad * 0.2, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def calc_metrics(y_true, y_pred, class_labels=None):
    # 分类指标
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_labels),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def normalize_read(value, expected_length=DIGIT_NUM):
    # 读数规范
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    return digits[:expected_length].zfill(expected_length)


#--------- 切分

def read_image(image_path):
    # 图像读取
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"图像读取失败：{image_path}")
    return image


def to_gray(image):
    # 灰度图
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray_image, kernel_size=(5, 5)):
    # 高斯滤波
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


def adaptive_th(gray_image, inverse=True):
    # 自适应阈值
    threshold_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


def otsu_th(gray_image, inverse=True):
    # Otsu阈值
    threshold_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    ret, binary = cv2.threshold(gray_image, 0, 255, threshold_type | cv2.THRESH_OTSU)
    return binary


def normalize_th(threshold_mode):
    # 阈值模式
    threshold_mode = str(threshold_mode).strip().lower()
    mapping = {
        "dynamic": "dynamic_threshold",
        "dynamic_threshold": "dynamic_threshold",
        "otsu": "otsu_threshold",
        "otsu_threshold": "otsu_threshold",
        "adaptive": "adaptive_threshold",
        "adaptive_threshold": "adaptive_threshold",
    }
    return mapping.get(threshold_mode, "dynamic_threshold")


def apply_th(gray_image, threshold_mode="adaptive_threshold", inverse=True):
    # 阈值分割
    threshold_mode = normalize_th(threshold_mode)
    if threshold_mode == "otsu_threshold":
        return otsu_th(gray_image, inverse=inverse)
    return adaptive_th(gray_image, inverse=inverse)


def canny(image):
    # Canny
    return cv2.Canny(image, 50, 150)


def morph(binary_image):
    # 形态学
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    return image


def sort_points(points):
    # 四点排序
    points = np.array(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    value_sum = points.sum(axis=1)
    value_diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(value_sum)]
    rect[2] = points[np.argmax(value_sum)]
    rect[1] = points[np.argmin(value_diff)]
    rect[3] = points[np.argmax(value_diff)]
    return rect


def warp_perspective(image, points, output_size=READING_SIZE):
    # 透视校正
    rect = sort_points(points)
    width, height = output_size
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def clip_box(region, image_shape):
    # 区域限制
    x, y, w, h = [int(value) for value in region]
    image_h, image_w = image_shape[:2]
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(1, min(w, image_w - x))
    h = max(1, min(h, image_h - y))
    return x, y, w, h


def crop_box(image, region):
    # 区域裁剪
    x, y, w, h = clip_box(region, image.shape)
    return image[y : y + h, x : x + w].copy()


def find_read_boxes(image, edge_image):
    # 读数候选
    contours, info = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_h, image_w = image.shape[:2]
    image_area = image_h * image_w
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.005:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)
        area_ratio = area / max(image_area, 1)

        if aspect_ratio < 2.0 or aspect_ratio > 8.0:
            continue
        if h < image_h * 0.03 or w < image_w * 0.10:
            continue

        if len(approx) == 4:
            points = approx.reshape(4, 2)
            extra_score = 4
        else:
            rect = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rect)
            extra_score = 0

        score = area_ratio * 100 + aspect_ratio + extra_score
        candidates.append(
            {
                "region": (x, y, w, h),
                "points": np.array(points, dtype=np.float32),
                "score": score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def locate_read(image, threshold_mode="adaptive_threshold"):
    # 读数区域
    gray = to_gray(image)
    blur = gaussian_blur(gray)
    threshold = apply_th(blur, threshold_mode=threshold_mode, inverse=True)
    edges = canny(threshold)
    contour_vis = image.copy()
    image_h, image_w = image.shape[:2]
    aspect_ratio = image_w / max(image_h, 1)

    # 整图直用
    if 2.0 <= aspect_ratio <= 8.0:
        best = {
            "region": (0, 0, image_w, image_h),
            "points": np.array(
                [
                    [0, 0],
                    [image_w - 1, 0],
                    [image_w - 1, image_h - 1],
                    [0, image_h - 1],
                ],
                dtype=np.float32,
            ),
        }
        rectified = cv2.resize(image, READING_SIZE)
        used_center_crop = False
    else:
        # 轮廓区域
        candidates = find_read_boxes(image, edges)
        for candidate in candidates[:5]:
            x, y, w, h = candidate["region"]
            cv2.rectangle(contour_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if candidates:
            best = candidates[0]
            rectified = warp_perspective(image, best["points"])
            used_center_crop = False
        else:
            image_h, image_w = image.shape[:2]
            region = (int(image_w * 0.2), int(image_h * 0.35), int(image_w * 0.6), int(image_h * 0.2))
            best = {
                "region": region,
                "points": np.array(
                    [
                        [region[0], region[1]],
                        [region[0] + region[2], region[1]],
                        [region[0] + region[2], region[1] + region[3]],
                        [region[0], region[1] + region[3]],
                    ],
                    dtype=np.float32,
                ),
            }
            rectified = cv2.resize(crop_box(image, region), READING_SIZE)
            used_center_crop = True

    return {
        "gray": gray,
        "blur": blur,
        "threshold": threshold,
        "edges": edges,
        "contour_vis": contour_vis,
        "reading_region": best["region"],
        "reading_points": best["points"],
        "rectified": rectified,
        "threshold_mode": normalize_th(threshold_mode),
        "used_center_crop": used_center_crop,
    }


def split_even(region_image, expected_digits=DIGIT_NUM):
    # 等宽切分
    if len(region_image.shape) == 3:
        height, width = region_image.shape[:2]
    else:
        height, width = region_image.shape

    regions = []
    part_width = width / expected_digits
    for index in range(expected_digits):
        x = int(round(index * part_width))
        next_x = int(round((index + 1) * part_width))
        w = max(1, next_x - x)
        y = int(height * 0.1)
        h = int(height * 0.8)
        regions.append((x, y, w, h))
    return regions


def split_by_projection(binary_image, expected_digits=DIGIT_NUM):
    # 投影切分
    image_h, image_w = binary_image.shape[:2]
    if image_w < expected_digits * 8:
        return None

    col_sum = binary_image.sum(axis=0).astype(np.float32)
    kernel = np.ones(7, dtype=np.float32) / 7.0
    col_sum = np.convolve(col_sum, kernel, mode="same")

    part_width = image_w / expected_digits
    min_width = max(10, int(part_width * 0.45))
    search_width = max(6, int(part_width * 0.22))
    cuts = [0]

    for index in range(1, expected_digits):
        center = int(round(index * part_width))
        left = max(cuts[-1] + min_width, center - search_width)
        right = min(image_w - min_width, center + search_width)
        if right <= left:
            return None
        local_sum = col_sum[left : right + 1]
        cut = left + int(np.argmin(local_sum))
        if cut - cuts[-1] < min_width:
            return None
        cuts.append(cut)

    cuts.append(image_w)

    regions = []
    y = int(image_h * 0.1)
    h = int(image_h * 0.8)
    for index in range(expected_digits):
        x = cuts[index]
        next_x = cuts[index + 1]
        w = next_x - x
        if w < min_width:
            return None
        regions.append((x, y, w, h))
    return regions


def find_fg_box(binary_image):
    # 前景区域
    contours, info = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    boxes = []
    image_h, image_w = binary_image.shape[:2]
    min_area = max(4, int(image_h * image_w * 0.01))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return None

    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[0] + box[2] for box in boxes)
    y_max = max(box[1] + box[3] for box in boxes)
    return x_min, y_min, x_max - x_min, y_max - y_min


def refine_digit(gray_image, binary_image):
    # 单格裁边
    box = find_fg_box(binary_image)
    if box is None:
        return gray_image, False

    x, y, w, h = box
    pad_x = max(1, int(w * 0.10))
    pad_y = max(1, int(h * 0.08))
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(gray_image.shape[1] - x, w + pad_x * 2)
    h = min(gray_image.shape[0] - y, h + pad_y * 2)
    return crop_box(gray_image, (x, y, w, h)), True


def normalize_digit(digit_image, target_size=DIGIT_SIZE):
    # 单字符尺寸
    if digit_image is None or digit_image.size == 0:
        raise ValueError("单字符图为空。")

    if len(digit_image.shape) == 3:
        digit_image = to_gray(digit_image)

    gray = digit_image.copy()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = otsu_th(blur, inverse=True)
    box = find_fg_box(binary)
    if box is not None:
        gray = crop_box(gray, box)
        binary = crop_box(binary, box)

    digit_image = cv2.bitwise_and(gray, gray, mask=binary)

    target_w, target_h = target_size
    image_h, image_w = digit_image.shape[:2]
    scale = min(target_w / max(image_w, 1), target_h / max(image_h, 1))
    new_w = max(1, int(image_w * scale))
    new_h = max(1, int(image_h * scale))

    resized = cv2.resize(digit_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def split_digits(region_image, expected_digits=DIGIT_NUM, threshold_mode="adaptive_threshold"):
    # 数字切分
    gray = to_gray(region_image) if len(region_image.shape) == 3 else region_image.copy()
    blur = gaussian_blur(gray)
    threshold = apply_th(blur, threshold_mode=threshold_mode, inverse=True)
    morph_img = morph(threshold)

    # 投影边界
    digit_regions = split_by_projection(morph_img, expected_digits)
    used_even_split = False
    if digit_regions is None:
        digit_regions = split_even(morph_img, expected_digits)
        used_even_split = True

    digit_images = []
    found_count = 0

    for region in digit_regions:
        gray_crop = crop_box(gray, region)
        binary_crop = crop_box(morph_img, region)
        refined_crop, found_foreground = refine_digit(gray_crop, binary_crop)
        digit_images.append(normalize_digit(refined_crop))
        if found_foreground:
            found_count += 1

    return {
        "gray": gray,
        "blur": blur,
        "threshold": threshold,
        "morph": morph_img,
        "digit_regions": digit_regions,
        "digit_images": digit_images,
        "raw_region_count": found_count,
        "used_even_split": used_even_split,
        "success": found_count == expected_digits,
        "threshold_mode": normalize_th(threshold_mode),
    }


def auto_seg(image, expected_digits=DIGIT_NUM, threshold_mode="adaptive_threshold"):
    # 自动切分整图
    region_result = locate_read(image, threshold_mode=threshold_mode)
    digit_result = split_digits(
        region_result["rectified"],
        expected_digits=expected_digits,
        threshold_mode=threshold_mode,
    )
    if len(region_result["rectified"].shape) == 2:
        segmented_vis = cv2.cvtColor(region_result["rectified"], cv2.COLOR_GRAY2BGR)
    else:
        segmented_vis = region_result["rectified"].copy()

    for index, region in enumerate(digit_result["digit_regions"]):
        x, y, w, h = region
        cv2.rectangle(segmented_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(segmented_vis, str(index), (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    fail_reason = ""
    if not digit_result["success"]:
        if digit_result["used_even_split"]:
            fail_reason = "自动切分没有直接得到5位数字，已使用等宽切分"
        elif region_result["used_center_crop"]:
            fail_reason = "读数区域定位不稳定，已使用中心区域保底裁剪"
        else:
            fail_reason = "自动切分没有直接得到5位数字"

    return {
        "mode": "auto_segmentation",
        "gray": region_result["gray"],
        "blur": region_result["blur"],
        "threshold": region_result["threshold"],
        "edges": region_result["edges"],
        "contour_vis": region_result["contour_vis"],
        "reading_region": region_result["reading_region"],
        "rectified": region_result["rectified"],
        "segmented_vis": segmented_vis,
        "digit_regions": digit_result["digit_regions"],
        "digit_images": digit_result["digit_images"],
        "threshold_mode": normalize_th(threshold_mode),
        "used_center_crop": region_result["used_center_crop"],
        "used_even_split": digit_result["used_even_split"],
        "raw_region_count": digit_result["raw_region_count"],
        "success": digit_result["success"],
        "fail_reason": fail_reason,
    }


#--------- 显示

def draw_result(image, model_name, threshold_mode, reading_text, alarm_text, fps=None):
    # 结果文字
    canvas = image.copy()
    alarm_map = {
        "正常": "OK",
        "报警": "ALARM",
        "低置信度": "LOW_CONF",
        "未识别": "NO_READ",
        "未检测到有效数字区域": "NO_REGION",
    }
    show_alarm = alarm_map.get(alarm_text, "ERROR")
    color = (0, 255, 0) if alarm_text == "正常" else (0, 0, 255)
    cv2.putText(canvas, f"Model: {model_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(canvas, f"Threshold: {threshold_mode}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(canvas, f"Reading: {reading_text if reading_text else '-----'}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(canvas, f"Alarm: {show_alarm}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if fps is not None:
        cv2.putText(canvas, f"FPS: {fps:.2f}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


#--------- 训练

class AddShadow:
    # 阴影

    def __init__(self, prob=AUG_SHADOW_PROB):
        self.prob = prob

    def __call__(self, tensor):
        if random.random() > self.prob:
            return tensor

        channels, height, width = tensor.shape
        start_x = random.randint(0, max(width // 2, 1) - 1)
        end_x = random.randint(max(start_x + 1, width // 2), width)
        darkness = random.uniform(0.35, 0.75)

        shadow = torch.ones((1, height, width), dtype=tensor.dtype)
        shadow[:, :, start_x:end_x] *= darkness
        if random.random() < 0.5:
            shadow = torch.flip(shadow, dims=[2])

        if channels > 1:
            shadow = shadow.repeat(channels, 1, 1)

        return torch.clamp(tensor * shadow, 0.0, 1.0)


class AddGaussianNoise:
    # 噪声

    def __init__(self, std=AUG_NOISE_STD, prob=AUG_NOISE_PROB):
        self.std = std
        self.prob = prob

    def __call__(self, tensor):
        if random.random() > self.prob:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class DigitPathSet(Dataset):
    # 路径数据

    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def set_seed(seed=RANDOM_SEED):
    # 随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cpu_thread_count = min(8, os.cpu_count() or 1)
    torch.set_num_threads(cpu_thread_count)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tf_stats(model_name):
    # 归一化
    if model_name == "resnet18":
        return [0.485, 0.485, 0.485], [0.229, 0.229, 0.229]
    return [0.5], [0.5]


def build_tf(model_name, train=True):
    # 图像变换
    channel_num = 3 if model_name == "resnet18" else 1
    mean, std = get_tf_stats(model_name)
    transform_list = [
        transforms.Resize((DIGIT_SIZE[1], DIGIT_SIZE[0])),
        transforms.Grayscale(num_output_channels=channel_num),
    ]

    if train:
        transform_list.extend(
            [
                transforms.RandomApply([transforms.RandomRotation(degrees=AUG_ROTATE_DEG)], p=0.60),
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=AUG_TRANSLATE, scale=AUG_SCALE_RANGE)], p=0.50),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.3))], p=AUG_BLUR_PROB),
                transforms.RandomApply([transforms.ColorJitter(brightness=AUG_BRIGHTNESS)], p=0.60),
            ]
        )

    transform_list.append(transforms.ToTensor())

    if train:
        transform_list.append(AddShadow())
        transform_list.append(AddGaussianNoise())

    transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)


def make_loaders(model_name):
    # 数据加载
    train_dir = SINGLE_DIR / "train"
    val_dir = SINGLE_DIR / "valid"

    if not train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在：{train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"验证集目录不存在：{val_dir}")

    train_base = datasets.ImageFolder(str(train_dir))
    valid_base = datasets.ImageFolder(str(val_dir))
    class_names = train_base.classes
    train_samples = list(train_base.samples)
    holdout_samples = list(valid_base.samples)
    extra_valid_count = 0

    train_dataset = DigitPathSet(train_samples, transform=build_tf(model_name, train=True))
    val_dataset = DigitPathSet(holdout_samples, transform=build_tf(model_name, train=False))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, class_names, len(train_samples), len(holdout_samples), extra_valid_count


def make_model(model_name):
    # 模型创建
    if model_name == "lenet":
        return LeNet5(num_classes=10)
    if model_name == "resnet18":
        return build_resnet18(num_classes=10, pretrained=RESNET18_PRETRAIN)
    raise ValueError(f"不支持的模型：{model_name}")


def run_epoch(model, dataloader, criterion, optimizer=None, device="cpu"):
    # 单轮训练
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct_count = 0
    total_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct_count += (preds == labels).sum().item()
        total_count += labels.size(0)

    epoch_loss = running_loss / max(total_count, 1)
    epoch_acc = correct_count / max(total_count, 1)
    return epoch_loss, epoch_acc


def train_model(model_name):
    # 模型训练
    if model_name not in ["lenet", "resnet18"]:
        raise ValueError("只能训练lenet或resnet18")

    set_seed(RANDOM_SEED)
    make_dirs()

    # 训练设备
    device = get_device()
    print(f"训练设备 {device}")

    train_loader, val_loader, class_names, train_count, val_count, extra_valid_count = make_loaders(model_name)
    print(f"类别 {class_names}")
    print(f"训练样本 {train_count}")
    print(f"验证样本 {val_count}")
    print(f"并入训练的valid样本 {extra_valid_count}")

    # 优化器
    model = make_model(model_name).to(device)
    label_smoothing = 0.02 if model_name == "resnet18" else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if model_name == "resnet18":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch_count = LENET_EPOCHS if model_name == "lenet" else RESNET18_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epoch_count, 1))

    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epoch_count):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epoch_count}： "
            f"训练损失={train_loss:.4f}，训练准确率={train_acc:.4f}，"
            f"验证损失={val_loss:.4f}，验证准确率={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), CKPT_DIR / f"{model_name}_best.pth")
            print(f"已保存当前最优{model_name}模型")

    summary = {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_sample_count": train_count,
        "val_sample_count": val_count,
        "official_valid_in_train": False,
        "official_valid_count": extra_valid_count,
        "full_train_sample_count": train_count,
        "target_acc": TARGET_ACC,
        "target_reached": best_val_acc >= TARGET_ACC,
    }
    summary_path = LENET_SUMMARY_FILE if model_name == "lenet" else RESNET18_SUMMARY_FILE
    ensure_dir(Path(summary_path).parent)
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


#--------- 模板

def make_templates(train_dir, templates_per_class=TEMP_NUM):
    # 模板图
    dataset = datasets.ImageFolder(str(train_dir))
    samples = dataset.samples
    grouped = defaultdict(list)
    for image_path, label_index in samples:
        grouped[label_index].append(image_path)

    templates = {}
    for label_index, image_paths in grouped.items():
        template_images = []
        for image_path in image_paths[:templates_per_class]:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            template_images.append(cv2.resize(image, DIGIT_SIZE))
        templates[int(label_index)] = template_images
    return templates


def match_score(query_image, template_image):
    # 匹配分数
    result = cv2.matchTemplate(query_image, template_image, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def template_pred(query_image, templates):
    # 模板预测
    if query_image.shape != DIGIT_SIZE[::-1]:
        query_image = cv2.resize(query_image, DIGIT_SIZE)

    scores = {}
    for label, template_list in templates.items():
        class_scores = [match_score(query_image, template_image) for template_image in template_list]
        scores[int(label)] = float(max(class_scores)) if class_scores else -1.0

    pred_label = max(scores, key=scores.get)
    return int(pred_label), scores


#--------- 推理

def get_device():
    # 设备
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_runtime(model_name=None, threshold_mode=None):
    # 自动方案
    if model_name and threshold_mode:
        return str(model_name).lower(), normalize_th(threshold_mode)

    if not EVAL_SUMMARY_FILE.exists():
        raise RuntimeError(f"缺少评估结果：{EVAL_SUMMARY_FILE}")
    if not RUNTIME_SUMMARY_FILE.exists():
        raise RuntimeError(f"缺少测速结果：{RUNTIME_SUMMARY_FILE}")

    eval_df = load_csv(EVAL_SUMMARY_FILE)
    runtime_df = load_csv(RUNTIME_SUMMARY_FILE)

    eval_cols = {"model_name", "threshold_mode", "effective_char_accuracy", "effective_output_rate"}
    runtime_cols = {"model_name", "threshold_mode", "fps"}
    if eval_df.empty or not eval_cols.issubset(set(eval_df.columns)):
        raise RuntimeError("评估结果不完整，无法自动选择展示方案。")
    if runtime_df.empty or not runtime_cols.issubset(set(runtime_df.columns)):
        raise RuntimeError("测速结果不完整，无法自动选择展示方案。")

    eval_df = eval_df.copy()
    runtime_df = runtime_df.copy()
    eval_df["model_name"] = eval_df["model_name"].astype(str).str.lower()
    runtime_df["model_name"] = runtime_df["model_name"].astype(str).str.lower()
    eval_df["threshold_mode"] = eval_df["threshold_mode"].map(normalize_th)
    runtime_df["threshold_mode"] = runtime_df["threshold_mode"].map(normalize_th)

    merged_df = eval_df.merge(runtime_df, on=["model_name", "threshold_mode"], how="inner")

    # 条件补全
    if model_name:
        merged_df = merged_df[merged_df["model_name"] == str(model_name).lower()].copy()
    if threshold_mode:
        merged_df = merged_df[merged_df["threshold_mode"] == normalize_th(threshold_mode)].copy()

    # 达标组合
    merged_df = merged_df[
        (merged_df["effective_char_accuracy"] >= TARGET_ACC)
        & (merged_df["effective_output_rate"] >= TARGET_OUTPUT_RATE)
        & (merged_df["fps"] >= TARGET_FPS)
    ].copy()

    if merged_df.empty:
        raise RuntimeError(
            f"当前没有同时满足{TARGET_ACC:.0%}有效准确率"
            f"{TARGET_OUTPUT_RATE:.0%}有效输出率"
            f"{TARGET_FPS:.0f}FPS的方案 请手动指定"
        )

    merged_df = merged_df.sort_values(["effective_char_accuracy", "effective_output_rate", "fps"], ascending=False)
    best_row = merged_df.iloc[0]
    return str(best_row["model_name"]), normalize_th(best_row["threshold_mode"])


def load_lenet(device=None, checkpoint_path=None):
    # 加载LeNet
    if device is None:
        device = get_device()
    if checkpoint_path is None:
        checkpoint_path = CKPT_DIR / "lenet_best.pth"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到LeNet权重：{checkpoint_path}")
    model = LeNet5(num_classes=10).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_resnet18(device=None, checkpoint_path=None):
    # 加载ResNet18
    if device is None:
        device = get_device()
    if checkpoint_path is None:
        checkpoint_path = CKPT_DIR / "resnet18_best.pth"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到ResNet18权重：{checkpoint_path}")
    model = build_resnet18(num_classes=10, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_template(train_dir=None):
    # 加载模板
    if train_dir is None:
        train_dir = SINGLE_DIR / "train"
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"找不到模板目录：{train_dir}")
    return make_templates(train_dir)


def load_model(model_name, device=None):
    # 模型加载
    model_name = str(model_name).lower()
    if model_name == "template":
        return load_template()
    if model_name == "lenet":
        return load_lenet(device=device)
    if model_name == "resnet18":
        return load_resnet18(device=device)
    raise ValueError(f"不支持的模型：{model_name}")


def make_infer_batch(digit_images, model_name, device):
    # 推理批量
    if not digit_images:
        raise ValueError("数字列表为空")

    image_list = []
    for digit_image in digit_images:
        if digit_image is None or digit_image.size == 0:
            raise ValueError("单字符图为空。")
        resized = cv2.resize(digit_image, DIGIT_SIZE)
        tensor = torch.from_numpy(resized).float() / 255.0
        if model_name == "resnet18":
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        else:
            tensor = tensor.unsqueeze(0)
        image_list.append(tensor)

    batch = torch.stack(image_list, dim=0).to(device)
    mean_values, std_values = get_tf_stats(model_name)
    mean = torch.tensor(mean_values, dtype=batch.dtype, device=device).view(1, len(mean_values), 1, 1)
    std = torch.tensor(std_values, dtype=batch.dtype, device=device).view(1, len(std_values), 1, 1)
    return (batch - mean) / std


def prob_to_pred(prob_list):
    # 概率转结果
    pred_labels = []
    pred_scores = []
    for probs in prob_list:
        pred_label = int(np.argmax(probs))
        pred_score = float(np.max(probs))
        pred_labels.append(pred_label)
        pred_scores.append(pred_score)
    return pred_labels, pred_scores


def pred_probs(digit_images, model_name, model_obj, device=None, pos_prior=None):
    # 概率输出
    if model_name == "template":
        prob_list = []
        for digit_image in digit_images:
            pred_label, score_map = template_pred(digit_image, model_obj)
            probs = np.zeros(10, dtype=np.float32)
            for label, score in score_map.items():
                probs[int(label)] = float(score)
            total_score = float(probs.sum())
            if total_score > 0:
                probs = probs / total_score
            prob_list.append(probs)
        return prob_list

    if device is None:
        device = get_device()
    batch = make_infer_batch(digit_images, model_name, device)
    with torch.no_grad():
        logits = model_obj(batch)
        probs = torch.softmax(logits, dim=1)

    prob_list = []
    for index in range(probs.shape[0]):
        px = probs[index : index + 1]
        if pos_prior is not None and index in pos_prior:
            prior = torch.tensor(pos_prior[index], dtype=px.dtype, device=px.device).view(1, -1)
            px = px * prior
            px = px / px.sum(dim=1, keepdim=True).clamp_min(1e-8)
        prob_list.append(px.squeeze(0).detach().cpu().numpy())
    return prob_list


def pred_digits(digit_images, model_name, model_obj, device=None):
    # 批量预测
    pos_prior = None
    prob_list = pred_probs(digit_images, model_name, model_obj, device=device, pos_prior=pos_prior)
    return prob_to_pred(prob_list)


def merge_prob_list(primary_probs, extra_probs):
    # 概率融合
    merged_list = []
    for primary, extra in zip(primary_probs, extra_probs):
        primary_weight = float(np.max(primary))
        extra_weight = float(np.max(extra))
        merged = primary * primary_weight + extra * extra_weight
        merged = merged / max(float(merged.sum()), 1e-8)
        merged_list.append(merged)
    return merged_list


def run_single_mode(image, model_name, model_obj, device, threshold_mode):
    # 单模式推理
    pipeline_result = auto_seg(image, threshold_mode=threshold_mode)
    pred_labels = []
    pred_scores = []
    prob_list = []
    if len(pipeline_result["digit_images"]) == DIGIT_NUM:
        pos_prior = None
        prob_list = pred_probs(
            pipeline_result["digit_images"],
            model_name,
            model_obj,
            device=device,
            pos_prior=pos_prior,
        )
        pred_labels, pred_scores = prob_to_pred(prob_list)
    return {
        "pipeline_result": pipeline_result,
        "pred_labels": pred_labels,
        "pred_scores": pred_scores,
        "prob_list": prob_list,
        "threshold_mode": normalize_th(threshold_mode),
    }


def pred_read(image, model_name, model_obj=None, device=None, threshold_mode=PREPARE_TH_MODE, lower_limit=LOW_LIM, upper_limit=UP_LIM):
    # 整图预测
    if device is None:
        device = get_device()
    if model_obj is None:
        model_obj = load_model(model_name, device=device)
    threshold_mode = normalize_th(threshold_mode)

    pred_labels = []
    pred_scores = []
    pipeline_result = None
    reading_text = ""
    raw_reading_text = ""
    reading_value = None
    is_alarm = False
    alarm_text = "未识别"
    mean_score = 0.0
    is_valid_output = False

    if threshold_mode == "dynamic_threshold":
        primary_result = run_single_mode(image, model_name, model_obj, device, "otsu_threshold")
        extra_result = run_single_mode(image, model_name, model_obj, device, "adaptive_threshold")
        pipeline_result = primary_result["pipeline_result"]

        if len(primary_result["pred_labels"]) == DIGIT_NUM and len(extra_result["pred_labels"]) == DIGIT_NUM:
            if model_name in ["lenet", "resnet18"]:
                merged_probs = merge_prob_list(primary_result["prob_list"], extra_result["prob_list"])
                pred_labels, pred_scores = prob_to_pred(merged_probs)
            else:
                for index in range(DIGIT_NUM):
                    primary_score = primary_result["pred_scores"][index]
                    extra_score = extra_result["pred_scores"][index]
                    if extra_score > primary_score:
                        pred_labels.append(extra_result["pred_labels"][index])
                        pred_scores.append(extra_score)
                    else:
                        pred_labels.append(primary_result["pred_labels"][index])
                        pred_scores.append(primary_score)
        elif len(primary_result["pred_labels"]) == DIGIT_NUM:
            pred_labels = list(primary_result["pred_labels"])
            pred_scores = list(primary_result["pred_scores"])
        elif len(extra_result["pred_labels"]) == DIGIT_NUM:
            pipeline_result = extra_result["pipeline_result"]
            pred_labels = list(extra_result["pred_labels"])
            pred_scores = list(extra_result["pred_scores"])
        else:
            pipeline_result = primary_result["pipeline_result"]
    else:
        single_result = run_single_mode(image, model_name, model_obj, device, threshold_mode)
        pipeline_result = single_result["pipeline_result"]
        pred_labels = list(single_result["pred_labels"])
        pred_scores = list(single_result["pred_scores"])

    if len(pred_labels) == DIGIT_NUM:
        raw_reading_text = "".join(str(int(value)) for value in pred_labels[:DIGIT_NUM])
        mean_score = float(np.mean(pred_scores)) if pred_scores else 0.0
        is_valid_output = mean_score >= CONF_SCORE_TH
        if is_valid_output and raw_reading_text.isdigit():
            reading_text = raw_reading_text
            reading_value = int(reading_text)
            if reading_value < lower_limit or reading_value > upper_limit:
                is_alarm = True
                alarm_text = "报警"
            else:
                alarm_text = "正常"
        else:
            alarm_text = "低置信度"
    else:
        alarm_text = "未检测到有效数字区域"

    return {
        "pipeline_result": pipeline_result,
        "pred_labels": pred_labels,
        "pred_scores": pred_scores,
        "reading_text": reading_text,
        "raw_reading_text": raw_reading_text,
        "reading_value": reading_value,
        "mean_score": mean_score,
        "is_valid_output": is_valid_output,
        "is_alarm": is_alarm,
        "alarm_text": alarm_text,
    }

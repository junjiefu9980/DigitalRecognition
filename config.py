"""
统一项目里的路径和参数
包括数据目录 结果目录 训练参数和阈值
"""

from pathlib import Path


# 项目
PRO_ROOT = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = PRO_ROOT / "data"
DATASET_DIR = DATA_DIR / "raw"
PARSE_DIR = DATA_DIR / "parsed"
SINGLE_DIR = DATA_DIR / "digit_dataset"
PARSED_FILE = PARSE_DIR / "parsed_annotations.csv"
DATASET_STATS_FILE = PARSE_DIR / "digit_dataset_statistics.csv"
DEMO_IMAGE = DATASET_DIR / "test" / "00000_00000.png"

# 结果和模型
CKPT_DIR = PRO_ROOT / "checkpoints"
RESULT_DIR = PRO_ROOT / "results"

# 结果文件
SEG_SUMMARY_FILE = RESULT_DIR / "segmentation_compare.csv"
TEMPLATE_SUMMARY_FILE = RESULT_DIR / "template_summary.csv"
LENET_SUMMARY_FILE = RESULT_DIR / "lenet_training.json"
RESNET18_SUMMARY_FILE = RESULT_DIR / "resnet18_training.json"
EVAL_SUMMARY_FILE = RESULT_DIR / "evaluation_summary.csv"
EVAL_CHAR_FIG = RESULT_DIR / "evaluation_char_accuracy.png"
RUNTIME_SUMMARY_FILE = RESULT_DIR / "runtime_summary.csv"
RUNTIME_FIG = RESULT_DIR / "runtime_fps.png"

# 图像后缀
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# 读数，单字符尺寸
DIGIT_NUM = 5
DIGIT_SIZE = (32, 32)
READING_SIZE = (320, 96)

# 基础训练参数
BATCH_SIZE = 64
LENET_EPOCHS = 30
RESNET18_EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_SEED = 42
RESNET18_PRETRAIN = True

# 报警阈值
LOW_LIM = 0
UP_LIM = 99999

# 数据准备默认阈值
PREPARE_TH_MODE = "otsu_threshold"
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 5

# 模板匹配模板
TEMP_NUM = 5

# 指标
TARGET_ACC = 0.95
TARGET_OUTPUT_RATE = 0.95
TARGET_FPS = 20.0
CONF_SCORE_TH = 0.75

# 训练增强，旋转保留±30度，模拟倾斜，缩放和小幅平移
AUG_ROTATE_DEG = 30
AUG_SCALE_RANGE = (0.95, 1.05)
AUG_TRANSLATE = (0.03, 0.03)

# 亮度、模糊、阴影、噪声
AUG_BRIGHTNESS = (0.75, 1.25)
AUG_BLUR_PROB = 0.15
AUG_SHADOW_PROB = 0.12
AUG_NOISE_PROB = 0.08
AUG_NOISE_STD = 0.02

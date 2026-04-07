# 基于动态阈值与CNN的工业仪表数字识别系统

## 1. 项目说明

这个项目做5位工业仪表数字识别。

原始数据来自Hugging Face上的`UFPR-AMR`。官方数据本身已经分成`train`、`valid`、`test`三个划分，项目直接沿用这三个划分，不再自己重新拆分。

处理流程是：

1. 下载官方`parquet`
2. 把`parquet`展开成图片和读数标注
3. 做灰度化和高斯滤波
4. 做阈值分割、边缘检测、读数区域定位
5. 做透视校正
6. 把读数区域切成5个数字
7. 生成单字符数据集
8. 用模板匹配、LeNet、ResNet18做识别
9. 输出评估结果和测速结果

训练增强包括：

1. 旋转`±30°`
2. 缩放和平移
3. 亮度变化
4. 高斯模糊
5. 阴影
6. 高斯噪声

## 2. 目录结构

```text
DigitalRecognition/
├── README.md
├── requirements.txt
├── config.py
├── tools.py
├── models/
│   ├── __init__.py
│   ├── lenet.py
│   └── resnet18.py
├── scr/
│   ├── 00_unpack_parquet.py
│   ├── 01_prepare_dataset.py
│   ├── 02_train_template_baseline.py
│   ├── 03_train_lenet.py
│   ├── 04_train_resnet18.py
│   ├── 05_evaluate_all.py
│   ├── 06_benchmark_runtime.py
│   ├── 07_infer_image.py
│   ├── 08_realtime_demo.py
│   └── 09_gui_app.py
├── data/
│   ├── raw/
│   │   ├── train/
│   │   │   ├── 00000.parquet
│   │   │   ├── 00000_00000.png
│   │   │   ├── 00000_00000.json
│   │   │   └── ...
│   │   ├── valid/
│   │   │   ├── 00000.parquet
│   │   │   ├── 00000_00000.png
│   │   │   ├── 00000_00000.json
│   │   │   └── ...
│   │   └── test/
│   │       ├── 00000.parquet
│   │       ├── 00000_00000.png
│   │       ├── 00000_00000.json
│   │       └── ...
│   ├── parsed/
│   │   ├── parsed_annotations.csv
│   │   └── digit_dataset_statistics.csv
│   └── digit_dataset/
│       ├── train/
│       ├── valid/
│       └── test/
├── checkpoints/
│   ├── lenet_best.pth
│   └── resnet18_best.pth
└── results/
    ├── segmentation_compare.csv
    ├── template_summary.csv
    ├── lenet_training.json
    ├── resnet18_training.json
    ├── evaluation_summary.csv
    ├── evaluation_char_accuracy.png
    ├── runtime_summary.csv
    └── runtime_fps.png
```

说明：

1. `data/raw`里的`.parquet`是官方下载数据
2. 运行`00`之后，同目录会展开出`.png`和`.json`
3. `data/digit_dataset`是切好的单字符训练集
4. `results`里保留的是最后的总结果

## 3. 完整运行

完整重跑时，按`00`到`06`顺序依次运行。
运行顺序：

1. `python scr/00_unpack_parquet.py`
2. `python scr/01_prepare_dataset.py`
3. `python scr/02_train_template_baseline.py`
4. `python scr/03_train_lenet.py`
5. `python scr/04_train_resnet18.py`
6. `python scr/05_evaluate_all.py`
7. `python scr/06_benchmark_runtime.py --model all`

## 4. 展示运行

展示部分直接运行`07`、`08`、`09`即可。

`07`：单张图识别，不加参数时会直接使用`config.py`里的默认示例图。  
运行：`python scr/07_infer_image.py`

`08`：摄像头实时识别，会打开摄像头窗口，显示当前读数、报警状态和FPS。  
运行：`python scr/08_realtime_demo.py`

`09`：图形界面，会打开可视化窗口，支持图片识别、摄像头识别、历史记录和趋势图查看。  
运行：`python scr/09_gui_app.py`

## 5. 脚本和输出

`00_unpack_parquet.py`：把官方`parquet`展开成图片和标注  
输出物：`00000_00000.png`等展开后的原始图片；`00000_00000.json`等对应的原始读数标注

`01_prepare_dataset.py`：整理读数、比较切分、生成单字符数据集  
输出物：`parsed_annotations.csv`，整图样本和读数总表；`digit_dataset_statistics.csv`，单字符数量统计表；`00000_0.png`等训练用的单字符数字图片；`segmentation_compare.csv`，不同阈值模式的切分对比表

`02_train_template_baseline.py`：跑模板匹配baseline
输出物：`template_summary.csv`，模板匹配方法的结果表

`03_train_lenet.py`：训练LeNet  
输出物：`lenet_best.pth`，LeNet最优权重；`lenet_training.json`，LeNet训练结果

`04_train_resnet18.py`：训练ResNet18  
输出物：`resnet18_best.pth`，ResNet18最优权重；`resnet18_training.json`，ResNet18训练结果

`05_evaluate_all.py`：统一评估不同模型和阈值  
输出物：`evaluation_summary.csv`，统一评估结果表；`evaluation_char_accuracy.png`，有效字符准确率对比图

`06_benchmark_runtime.py`：统计不同方案的时间和FPS  
输出物：`runtime_summary.csv`，测速结果表；`runtime_fps.png`，FPS对比图

`07_infer_image.py`：对单张图做完整识别  
输出物：终端结果。包含读数、每位分数、平均置信度和报警状态

`08_realtime_demo.py`：打开摄像头做实时识别和报警展示  
输出物：实时窗口。显示当前读数、报警状态和FPS

`09_gui_app.py`：打开图形界面展示读数、报警、历史记录和趋势图  
输出物：图形界面。显示当前读数、报警状态、历史记录和趋势图

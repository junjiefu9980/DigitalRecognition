"""
用单字符数字样本训练LeNet
输入是已经准备好的训练集和验证集
输出最优权重和训练结果
"""

import sys
from pathlib import Path


PRO_ROOT = Path(__file__).resolve().parents[1]
if str(PRO_ROOT) not in sys.path:
    sys.path.insert(0, str(PRO_ROOT))

from tools import train_model


def main():
    summary = train_model("lenet")
    print("LeNet训练完成")
    print(summary)


if __name__ == "__main__":
    main()

"""
定义LeNet数字识别模型
输入是单通道灰度图
输出是0到9的分类结果
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    # LeNet主体

    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积层1
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)

        # 卷积层2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = LeNet5(num_classes=10)
    test_input = torch.randn(2, 1, 32, 32)
    test_output = model(test_input)
    print("模型测试输出形状：", test_output.shape)

"""
定义ResNet18数字识别模型
输入是三通道灰度图
输出是0到9的分类结果
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes=10, pretrained=False):
    # 预训练
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    try:
        model = models.resnet18(weights=weights)
    except Exception:
        weights = None
        model = models.resnet18(weights=weights)

    # 输入层
    old_conv = model.conv1
    new_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained and weights is not None:
        new_conv.weight.data.copy_(old_conv.weight.data[:, :, 2:5, 2:5])
    model.conv1 = new_conv

    # 下采样
    model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":
    model = build_resnet18(num_classes=10, pretrained=False)
    test_input = torch.randn(2, 3, 32, 32)
    test_output = model(test_input)
    print("模型测试输出形状：", test_output.shape)

"""
ResNet-20 model definition for CIFAR-10.

This module implements a standard ResNet-20 architecture suitable for
CIFAR-10 classification (32x32 input images, 10 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet-20 model for CIFAR-10 and MNIST.
    
    Architecture:
    - Initial conv: in_channels -> 16 channels
    - Layer 1: 3 blocks, 16 channels
    - Layer 2: 3 blocks, 32 channels
    - Layer 3: 3 blocks, 64 channels
    - Global average pool
    - Linear classifier: 64 -> num_classes
    
    Args:
        num_classes: Number of output classes (default: 10)
        in_channels: Number of input channels (default: 3 for RGB, 1 for grayscale)
    """
    
    def __init__(self, num_classes=10, in_channels=3):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Adaptive pooling for different input sizes
        self.linear = nn.Linear(64, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)  # Adaptive pooling works for both CIFAR-10 (8x8) and MNIST (7x7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



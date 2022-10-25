# coding: utf-8


import torch
import torch.nn as nn
import collections


class AlexNet1D(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3),
        )
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool = nn.MaxPool1d(2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet1d(num_classes):
    """ return a alexnet1d object
    """
    return AlexNet1D(num_classes)
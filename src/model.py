import torch as tr
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models


class Model(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(Model, self).__init__()
        self.resnet = models.resnext101_32x8d(pretrained=pretrained)
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.ReLU(),
                                nn.BatchNorm1d(1000),
                                nn.Linear(1000, n_classes),
                                nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def freeze_resnet(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def unfreeze_resnet(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

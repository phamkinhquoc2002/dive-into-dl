import torch
import numpy as numpy
from torch import nn
from d2l import torch as d2l
from config import device


class AlexNet(d2l.Classifier):
    def __init__(self, lr = 0.1, num_classes = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=4, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride = 2), nn.Flatten(),
            nn.LazyLinear(4096),
            nn.LazyLinear(4096),
            nn.LazyLinear(num_classes)
        )
        self.net.apply(d2l.init_cnn)
        self.to(device)
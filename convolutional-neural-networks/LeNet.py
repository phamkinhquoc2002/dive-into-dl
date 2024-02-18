import numpy as numpy
import matplotlib.pyplot as plt
plt.ion()
from torch import nn
import torch
from torchvision import datasets
from d2l import torch as d2l
from config import init_cnn


class LeNet(d2l.Classifier): #@save
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride =2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(100), nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )
        self.net.apply(init_cnn)





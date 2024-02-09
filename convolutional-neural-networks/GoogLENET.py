import torch
import numpy as numpy
from torch import nn
from d2l import torch as d2l
from torch.nn import F
from config import init_cnn

class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        #Branch 1
        self.b1 = nn.LazyConv2d(c1, kernel_size=1)
        #Branch 2
        self.b2_1 = nn.LazyConv2d(c2, kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2, kernel_size=3, padding=1)
        #Branch 3
        self.b3_1 = nn.LazyConv2d(c3, kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3, kernel_size=5, padding=2)
        #Branch 4
        self.b4_1 = nn.MaxPool2d(c4, kernel_size=3, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)
    def forward(self, x):
        b1 = F.relu(self.b1(x))
        b2 = F.relu(self.b2_2(self.b2_1(x)))
        b3 = F.relu(self.b3_2(self.b3_1(x)))
        b4 = F.relu(self.b4_2(self.b4_1))
        return torch.concat((b1, b2, b3, b4), dim=1)